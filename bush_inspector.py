import cv2
import numpy as np
import os
import math
import json
import time
import logging
from datetime import datetime

logger = logging.getLogger("Inspector")

# =========================================================================
# 클래스 정의
# =========================================================================
class BushInspectionResult:
    def __init__(self):
        self.image_name = ""
        self.bush_id = -1
        self.center_x_mm = 0.0
        self.center_y_mm = 0.0
        self.center_x_px = 0
        self.center_y_px = 0
        
        self.final_angle = 0.0      
        self.min_rect_angle = 0.0   
        self.corner_angle = 0.0     
        self.angle_diff = 0.0       
        
        self.direction_code = 0 
        self.surface_type = ""
        self.text_type = "" 
        self.contour_area = 0.0
        self.text_blob_box = None 
        self.blob_stats = {} 
        self.blob_distance = 0.0 
        self.edge_check_result = "" 
        self.edge_length_sum = 0    
        self.edge_map = None        
        self.failed_blobs = []
        self.min_rect_box = None

class BushInfo:
    def __init__(self, contour, center, area, corners):
        self.id = -1
        self.contour = contour
        self.center = center 
        self.area = area
        self.corners = corners 

# =========================================================================
# 메인 클래스: BushInspector
# =========================================================================
class BushInspector:
    def __init__(self, pixel_to_mm=0.028761, config_path=None, debug_mode=True):
        self.pixel_to_mm = pixel_to_mm
        self.debug_mode = debug_mode
        self.center_check_radius = 10
        
        # 1. Product Config 로드
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "product_config.json")
        self.product_configs = self.load_json(config_path)
        
        # 2. Global Config 로드
        config_dir = os.path.dirname(config_path)
        self.global_config_path = os.path.join(config_dir, "global_config.json")
        self.global_params = self.load_global_config()

        # 3. 로드된 Global 값 적용
        self.apply_global_params()

        self.default_params = {
            'outer_margin': 10, 'inner_margin': 10, 'min_area': 500, 'max_area': 5000,      
            'min_short_side': 40, 'max_short_side': 100, 'min_long_side': 40, 'max_long_side': 100,   
            'min_ratio': 0.5, 'max_ratio': 1.5, 'min_dist': 150, 'max_dist': 210,
            'min_edge_length_sum': 240, 'dilation_iter': 3     
        }
        
        self.current_product_id = "1"
        self.current_params = self.get_current_params()

    def load_json(self, path):
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def load_global_config(self):
        defaults = {
            "pixel_to_mm": 0.028761,
            "max_retry_count": 3,
            "bush_min_area": 150000,
            "bush_max_area": 170000,
            "min_center_dist_mm": 14.0,
            "thresh_block_size": 15,
            "thresh_c": 3,
            "simple_thresh_val": 125, # [추가] Simple Threshold 값 (기본 125)
            "canny_thresh1": 30,
            "canny_thresh2": 100,
            "canny_blur_size": 3 
        }
        if os.path.exists(self.global_config_path):
            try:
                with open(self.global_config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for k, v in defaults.items():
                        if k not in data: data[k] = v
                    return data
            except Exception as e:
                print(f"Global config load error: {e}")
                return defaults
        else:
            return defaults

    def save_global_config(self):
        try:
            with open(self.global_config_path, 'w', encoding='utf-8') as f:
                json.dump(self.global_params, f, indent=4)
            print("Global config saved.")
        except Exception as e:
            print(f"Global config save error: {e}")

    def apply_global_params(self):
        p = self.global_params
        self.pixel_to_mm = p.get("pixel_to_mm", 0.028761)
        self.max_retry_count = p.get("max_retry_count", 3)
        self.bush_min_area = p.get("bush_min_area", 150000)
        self.bush_max_area = p.get("bush_max_area", 170000)
        self.min_center_dist_mm = p.get("min_center_dist_mm", 14.0)
        self.thresh_block_size = p.get("thresh_block_size", 15)
        self.thresh_c = p.get("thresh_c", 3)
        self.simple_thresh_val = p.get("simple_thresh_val", 125) # [추가] 변수 적용
        
        self.canny_thresh1 = p.get("canny_thresh1", 30)
        self.canny_thresh2 = p.get("canny_thresh2", 100)
        self.canny_blur_size = p.get("canny_blur_size", 3)
        
        self.update_derived_params()

    def update_derived_params(self):
        if self.pixel_to_mm > 0:
            target_dist_mm = 3.6 * math.sqrt(2)
            self.target_dist_px = target_dist_mm / self.pixel_to_mm
            self.min_center_dist_px = self.min_center_dist_mm / self.pixel_to_mm
        else:
            self.target_dist_px = 0
            self.min_center_dist_px = 0

    def set_target_product(self, product_id):
        self.current_product_id = str(product_id)
        self.current_params = self.get_current_params()

    def get_current_params(self):
        if self.current_product_id in self.product_configs:
            return self.product_configs[self.current_product_id]
        else:
            return self.default_params

    def preprocess_image(self, image, debug_dir=None, debug_name=None , debug_mode=False):
        if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else: gray = image
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 1. Adaptive Threshold (기존 방식)
        block_size = int(self.thresh_block_size)
        if block_size % 2 == 0: block_size += 1
        
        binary_adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 
            block_size, self.thresh_c
        )

       # 2. Simple Threshold (신규 방식 - 덩어리 위주)
        # [수정] Config 변수(simple_thresh_val) 사용
        _, binary_simple = cv2.threshold(blurred, int(self.simple_thresh_val), 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # [A] Weak Merge: 기존 로직 (iter=1)
        binary_merged_weak = cv2.morphologyEx(binary_adaptive, cv2.MORPH_CLOSE, kernel, iterations=1)

        # [B] Strong Merge: 단순 Thresh + 높은 iter (iter=3) -> 겹침 확인용
        binary_merged_strong = cv2.morphologyEx(binary_simple, cv2.MORPH_CLOSE, kernel, iterations=3)

        if debug_mode:
            try:
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, f"{debug_name}_blurred.jpg"), blurred)
                cv2.imwrite(os.path.join(debug_dir, f"{debug_name}_binary_adaptive.jpg"), binary_adaptive)
                cv2.imwrite(os.path.join(debug_dir, f"{debug_name}_binary_simple.jpg"), binary_simple)
                cv2.imwrite(os.path.join(debug_dir, f"{debug_name}_binary_merged_weak.jpg"), binary_merged_weak)
                cv2.imwrite(os.path.join(debug_dir, f"{debug_name}_binary_merged_strong.jpg"), binary_merged_strong)
            except Exception as e:
                logger.error(f"Preprocess Debug Save Error: {e}")
        else :
            try:
                os.makedirs(debug_dir, exist_ok=True)
            except Exception as e:
                logger.error(f"Preprocess Image Save Error: {e}")

        return binary_merged_weak, binary_merged_strong

    def detect_bushes(self, image, output_dir=None, image_name=None, debug_mode=False):
        processed_weak, processed_strong = self.preprocess_image(image, output_dir, image_name , debug_mode)
        
        # Contour 검출에는 Strong 이미지 사용
        contours, _ = cv2.findContours(processed_strong, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # [신규] 이미지 전체 크기 가져오기 (경계 체크용)
        img_h, img_w = processed_strong.shape[:2]
        edge_margin = 5  # 가장자리 여유분 (픽셀)

        contours_with_area = []
        min_area = self.bush_min_area
        max_area = self.bush_max_area
        visual_min_area = min_area * 0.5
        visual_max_area = max_area * 2.0
        
        ignored_blobs = [] 
        candidates = []
        total_contour_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= 1.0: continue
            
            # [신규 로직] 1. 이미지 경계(Edge)에 닿은 객체 필터링
            # 바운딩 박스 추출
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 상하좌우 경계에 닿았는지 확인 (margin 이내)
            is_touching_edge = (x <= edge_margin) or \
                               (y <= edge_margin) or \
                               (x + w >= img_w - edge_margin) or \
                               (y + h >= img_h - edge_margin)

            if is_touching_edge:
                # 경계에 닿아 잘린 객체는 중심점/각도 계산이 부정확하므로 과감히 제외
                # 시각화(ignored_blobs)에만 추가하고 로직 건너뜀
                rect = cv2.minAreaRect(cnt)
                center = (int(rect[0][0]), int(rect[0][1]))
                ignored_blobs.append({
                    'type': 'Edge', # 시각화 시 Pink색으로 표시됨 (기존 로직상 else로 빠짐)
                    'contour': cnt, 
                    'area': area, 
                    'center': center
                })
                continue # Skip! (Total Area 합산 및 후보군 등록 안 함)

            # 2. 유효한(잘리지 않은) 객체만 면적 합산
            if area > 100:
                total_contour_area += area

            # 3. 면적 기반 필터링 (Small / Large / Valid)
            if area < min_area:
                if area > visual_min_area:
                    rect = cv2.minAreaRect(cnt)
                    center = (int(rect[0][0]), int(rect[0][1]))
                    ignored_blobs.append({
                        'type': 'Small', 'contour': cnt, 'area': area, 'center': center
                    })
                continue
            
            if area > max_area:
                if area <= visual_max_area:
                    rect = cv2.minAreaRect(cnt)
                    center = (int(rect[0][0]), int(rect[0][1]))
                    ignored_blobs.append({
                        'type': 'Large', 'contour': cnt, 'area': area, 'center': center
                    })
                continue
            
            contours_with_area.append({'contour': cnt, 'area': area})

        contours_with_area.sort(key=lambda x: x['area'], reverse=True)

        for item in contours_with_area:
            cnt = item['contour']
            area = item['area']
            center, corners = self.find_corner_center_simple(cnt)
            candidates.append({
                'contour': cnt, 'area': area, 'center': center, 'corners': corners
            })

        bushes = []
        overlap_indices = set()
        min_dist_sq = self.min_center_dist_px ** 2 
        
        count = len(candidates)
        if count > 0:
            centers = np.array([c['center'] for c in candidates])
            for i in range(count):
                if i in overlap_indices: continue
                diff = centers[i+1:] - centers[i]
                dist_sq = np.sum(diff**2, axis=1)
                close_indices = np.where(dist_sq < min_dist_sq)[0]
                for idx in close_indices:
                    overlap_indices.add(i + 1 + idx)

        for i, cand in enumerate(candidates):
            if i in overlap_indices: continue
            bush = BushInfo(cand['contour'], cand['center'], cand['area'], cand['corners'])
            bush.id = len(bushes)
            bushes.append(bush)
            
        #return bushes, processed_weak, ignored_blobs, total_contour_area
        return bushes, processed_strong, ignored_blobs, total_contour_area

    def find_corner_center_simple(self, contour):
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        search_cnt = approx if len(approx) > 10 else contour

        rect = cv2.minAreaRect(search_cnt)
        cx, cy = int(rect[0][0]), int(rect[0][1])
        center = (cx, cy)
        
        candidates = []
        for pt in search_cnt:
            px, py = pt[0]
            dist = math.sqrt((px - cx)**2 + (py - cy)**2)
            diff = abs(dist - self.target_dist_px)
            candidates.append(((px, py), diff))
            
        candidates.sort(key=lambda x: x[1])
        final_corners = []
        min_pixel_dist = 30 
        for pt_data in candidates:
            if len(final_corners) >= 4: break
            pt = pt_data[0]
            is_far = True
            for exist_pt in final_corners:
                d = math.sqrt((pt[0]-exist_pt[0])**2 + (pt[1]-exist_pt[1])**2)
                if d < min_pixel_dist:
                    is_far = False; break
            if is_far: final_corners.append(pt)
            
        final_corners.sort(key=lambda p: math.atan2(p[1]-cy, p[0]-cx))
        return center, final_corners

    def calculate_angles(self, contour, corners, center):
        rect = cv2.minAreaRect(contour)
        raw_angle = rect[2]
        mr_angle = raw_angle + 45.0
        while mr_angle < -45: mr_angle += 90
        while mr_angle >= 45: mr_angle -= 90
        min_rect_box = np.int32(cv2.boxPoints(rect))
        corner_angle = 0.0
        if len(corners) == 4:
            diffs = []
            for p in corners:
                deg = math.degrees(math.atan2(p[1]-center[1], p[0]-center[0]))
                candidates = [45, 135, -135, -45]
                errors = [((deg - ref + 180) % 360 - 180) for ref in candidates]
                min_err = min(errors, key=abs)
                diffs.append(min_err)
            if diffs: corner_angle = sum(diffs) / len(diffs)
            while corner_angle < -45: corner_angle += 90
            while corner_angle >= 45: corner_angle -= 90
        else: corner_angle = mr_angle
        return mr_angle, corner_angle, min_rect_box

    def determine_direction(self, center, blob_box, part_angle):
        if blob_box is None: return 0
        blob_center = blob_box[0]
        dx = blob_center[0] - center[0]; dy = blob_center[1] - center[1]
        raw_angle_rad = math.atan2(dy, dx)
        raw_angle_deg = math.degrees(raw_angle_rad)
        corrected_angle = raw_angle_deg - part_angle
        while corrected_angle <= -180: corrected_angle += 360
        while corrected_angle > 180: corrected_angle -= 360
        if -45 <= corrected_angle < 45: return 1 
        elif 45 <= corrected_angle < 135: return 2 
        elif -135 <= corrected_angle < -45: return 4 
        else: return 3 

    def check_surface_type(self, image, center):
        x, y = center; r = self.center_check_radius; h, w = image.shape[:2]
        x1 = max(0, x - r); y1 = max(0, y - r); x2 = min(w, x + r); y2 = min(h, y + r)
        roi = image[y1:y2, x1:x2]
        if len(roi.shape) == 3: roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else: roi_gray = roi
        mean_val = np.mean(roi_gray)
        
        return "TOP" if mean_val < 200 else "BOT"

    def check_text_like_edges(self, image, rect):
        try:
            (cx, cy), (w, h), angle = rect
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            img_h, img_w = image.shape[:2]
            
            if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else: gray = image.copy()
            
            rotated_img = cv2.warpAffine(gray, M, (img_w, img_h))
            crop = cv2.getRectSubPix(rotated_img, (int(w), int(h)), (cx, cy))
            
            if crop is None or crop.size == 0: return "none_edge", 0
            
            if self.canny_blur_size > 0:
                k = self.canny_blur_size
                if k % 2 == 0: k += 1
                crop = cv2.GaussianBlur(crop, (k, k), 0)

            blob_edges = cv2.Canny(crop, self.canny_thresh1, self.canny_thresh2)
            edge_contours, _ = cv2.findContours(blob_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            total_length = 0
            for ec in edge_contours:
                length = cv2.arcLength(ec, False)
                if length > 5: total_length += length
            
            threshold = self.current_params['min_edge_length_sum']
            if total_length >= threshold: return "edge_detect", total_length
            else: return "none_edge", total_length
        except Exception: return "none_edge", 0

    # [수정] corners 인자 추가 및 모서리 근접 필터링 로직 추가
    def detect_text_blob_region(self, image, contour, center, inner_r_base, corners=None, debug_dir=None, debug_name=None ,debug_mode=False):
        p = self.current_params
        x, y, w, h = cv2.boundingRect(contour)
        img_h, img_w = image.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        roi_w, roi_h = x2 - x1, y2 - y1
        
        if roi_w <= 0 or roi_h <= 0:
            return False, None, {}, 0.0, "none_edge", 0, [], None

        roi_img = image[y1:y2, x1:x2]
        if len(roi_img.shape) == 3: roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        else: roi_gray = roi_img.copy()

        if self.canny_blur_size > 0:
            k = self.canny_blur_size
            if k % 2 == 0: k += 1
            roi_gray = cv2.GaussianBlur(roi_gray, (k, k), 0)

        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        roi_contour = contour - [x1, y1]
        cv2.drawContours(mask, [roi_contour], -1, 255, -1)
        
        if p['outer_margin'] > 0:
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (p['outer_margin'], p['outer_margin']))
            mask = cv2.erode(mask, kernel_erode)
            
        cx_roi, cy_roi = int(center[0] - x1), int(center[1] - y1)
        safe_inner_r = inner_r_base + p['inner_margin'] 
        cv2.circle(mask, (cx_roi, cy_roi), safe_inner_r, 0, -1)

        edges = cv2.Canny(roi_gray, self.canny_thresh1, self.canny_thresh2)
        filtered_edges = cv2.bitwise_and(edges, edges, mask=mask)
        
        iter_count = p['dilation_iter']
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
        dilated = cv2.dilate(filtered_edges, kernel_dilate, iterations=iter_count)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if debug_mode:
            roi_debug_img = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
            try:
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, f"{debug_name}_ROI_Debug_Img.jpg"), roi_debug_img)
                cv2.imwrite(os.path.join(debug_dir, f"{debug_name}_ROI_Gray.jpg"), roi_gray)
                cv2.imwrite(os.path.join(debug_dir, f"{debug_name}_ROI_Mask.jpg"), mask)
                cv2.imwrite(os.path.join(debug_dir, f"{debug_name}_ROI_Edges.jpg"), edges)
                cv2.imwrite(os.path.join(debug_dir, f"{debug_name}_ROI_Dilated.jpg"), dilated)
            except Exception as e:
                logger.error(f"Debug Image Save Error: {e}")

        best_blob_box = None; best_stats = {}; best_distance = 0.0
        best_edge_result = "none_edge"; best_edge_len = 0
        max_area = 0; has_text_blob = False; failed_blobs_data = [] 

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 50: 
                if debug_mode: logger.info(f"  [Blob #{i}] Skipped: Area {area} < 50")
                continue
            
            rect = cv2.minAreaRect(cnt)
            (rect_w, rect_h) = rect[1]; (rcx, rcy) = rect[0]
            short_side = min(rect_w, rect_h); long_side = max(rect_w, rect_h)
            aspect_ratio = float(short_side) / long_side if long_side > 0 else 0
            
            dist_from_center = math.sqrt((cx_roi - rcx)**2 + (cy_roi - rcy)**2)
            
            current_stats = {'area': int(area), 'short': int(short_side), 'long': int(long_side), 'ratio': round(aspect_ratio, 2)}
            global_center = (rcx + x1, rcy + y1)
            global_rect = (global_center, rect[1], rect[2])

            if p['min_area'] <= area <= p['max_area']:
                failed_blobs_data.append({'rect': global_rect, 'dist': dist_from_center, 'stats': current_stats})

            is_valid_area = p['min_area'] < area < p['max_area']
            is_valid_ratio = p['min_ratio'] < aspect_ratio < p['max_ratio']
            is_valid_size = (p['min_short_side'] < short_side < p['max_short_side']) and \
                            (p['min_long_side'] < long_side < p['max_long_side'])
            is_valid_dist = p['min_dist'] <= dist_from_center <= p['max_dist']

            # [신규] 모서리 근접 여부 확인 (거리 < 20px)
            is_corner_interference = False
            closest_corner_dist = 9999
            if corners:
                for c_pt in corners:
                    # global_center와 모서리 점 사이 거리
                    c_dist = math.sqrt((global_center[0] - c_pt[0])**2 + (global_center[1] - c_pt[1])**2)
                    if c_dist < 40:
                        is_corner_interference = True
                        closest_corner_dist = c_dist
                        break

            if debug_mode and not (is_valid_area and is_valid_ratio and is_valid_size and is_valid_dist and not is_corner_interference):
                fail_reasons = []
                if not is_valid_area: fail_reasons.append("Area")
                if not is_valid_ratio: fail_reasons.append("Ratio")
                if not is_valid_size: fail_reasons.append("Size")
                if not is_valid_dist: fail_reasons.append("Dist")
                if is_corner_interference: fail_reasons.append(f"CornerProx({int(closest_corner_dist)})")
                
                reason_str = ",".join(fail_reasons)
                logger.info(f"  [Blob #{i}] Rejected: {reason_str} | Stats: Area={int(area)}, Dist={int(dist_from_center)}")
                try:
                    box = cv2.boxPoints(rect); box = np.int32(box)
                    cv2.drawContours(roi_debug_img, [box], 0, (0, 0, 255), 1)
                    cv2.putText(roi_debug_img, reason_str, (int(box[0][0]), int(box[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                except Exception: pass

            # [수정] 모서리 간섭이 없을 때만 통과
            if is_valid_area and is_valid_ratio and is_valid_size and is_valid_dist and not is_corner_interference:
                if area > max_area:
                    max_area = area
                    best_blob_box = global_rect
                    has_text_blob = True
                    best_distance = dist_from_center
                    best_stats = current_stats
                    e_res, e_len = self.check_text_like_edges(image, global_rect)
                    best_edge_result = e_res
                    best_edge_len = int(e_len)
                    
                    if debug_mode: 
                        logger.info(f"  [Blob #{i}] Selected! (Edge: {e_len})")
                        try:
                            box = cv2.boxPoints(rect); box = np.int32(box)
                            cv2.drawContours(roi_debug_img, [box], 0, (0, 255, 0), 2)
                        except: pass
        
        if debug_mode:
            try:
                cv2.imwrite(os.path.join(debug_dir, f"{debug_name}_ROI_Analysis.jpg"), roi_debug_img)
            except Exception as e:
                logger.error(f"Analysis Image Save Error: {e}")

        return has_text_blob, best_blob_box, best_stats, best_distance, \
               best_edge_result, best_edge_len, failed_blobs_data, None

    # [수정] corners 인자 전달
    def detect_text_region(self, image, contour, center, bush_id, output_dir, image_name, corners=None, debug_dir=None , debug_mode=False):
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0: return "back", None, {}, 0.0, "", 0, [], None
        bush_rect = cv2.minAreaRect(contour)
        bush_radius = min(bush_rect[1]) / 2
        estimated_inner_r = int(bush_radius * 0.5) 
        
        debug_name = f"{image_name}_Bush{bush_id}" if debug_dir else None

        has_blob, blob_box, blob_stats, blob_dist, edge_result, edge_len, failed_blobs, edge_map = \
            self.detect_text_blob_region(image, contour, center, estimated_inner_r, corners, debug_dir, debug_name , debug_mode)
            
        if has_blob:
             return "front", blob_box, blob_stats, blob_dist, edge_result, edge_len, failed_blobs, edge_map
        else:
            return "back", None, {}, 0.0, "", 0, failed_blobs, edge_map

    def visualize_result(self, image, results, bushes, ignored_blobs=[]):
        vis_img = image.copy()
        img_h, img_w = image.shape[:2]
        p = self.current_params

        for blob in ignored_blobs:
            cnt = blob['contour']
            area = blob['area']
            center = blob['center']
            
            if blob['type'] == 'Small': color = (0, 165, 255) 
            else: color = (255, 0, 255) 
                
            cv2.drawContours(vis_img, [cnt], -1, color, 2)
            text_x, text_y = center[0] - 20, center[1]
            label = f"{blob['type']}:{int(area)}"
            cv2.putText(vis_img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for i, (result, bush) in enumerate(zip(results, bushes)):
            contour = bush.contour; center = bush.center; corners = bush.corners
            cv2.drawContours(vis_img, [contour], -1, (0, 255, 0), 2)
            if result.min_rect_box is not None:
                cv2.drawContours(vis_img, [result.min_rect_box], 0, (0, 80, 200), 3)
            
            if result.text_blob_box is not None:
                box = cv2.boxPoints(result.text_blob_box); box = np.int32(box)
                edge_res = result.edge_check_result
                box_color = (255, 255, 0) if edge_res == "edge_detect" else (0, 255, 255)
                text_color = box_color

                cv2.drawContours(vis_img, [box], 0, box_color, 2) 
                blob_center = result.text_blob_box[0]
                cv2.line(vis_img, (int(center[0]), int(center[1])), 
                         (int(blob_center[0]), int(blob_center[1])), (255, 255, 0), 2)
                
                stats = result.blob_stats
                if stats:
                    line1 = f"{stats['short']}x{stats['long']} {edge_res} : {result.edge_length_sum}"
                    text_x = int(box[0][0]); text_y = int(box[0][1]) - 10
                    cv2.putText(vis_img, line1, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

            cv2.circle(vis_img, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
            ang_rad = math.radians(result.min_rect_angle) 
            end_x = int(center[0] + 40 * math.cos(ang_rad))
            end_y = int(center[1] + 40 * math.sin(ang_rad))
            cv2.line(vis_img, (int(center[0]), int(center[1])), (end_x, end_y), (0, 165, 255), 2)

            target_x_mm = (img_w - center[0]) * self.pixel_to_mm
            target_y_mm = center[1] * self.pixel_to_mm
            dir_map = {0: "-", 1: "E", 2: "S", 3: "W", 4: "N"}
            dir_str = dir_map.get(result.direction_code, "-")
            
            info_text = f"ID:{result.bush_id} {dir_str} {result.surface_type}:({target_x_mm:.1f}, {target_y_mm:.1f}) Angle :{result.min_rect_angle}"
            cv2.putText(vis_img, info_text, (int(center[0]) - 40, int(center[1]) - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(vis_img, info_text, (int(center[0]) - 40, int(center[1]) - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return vis_img

    def highlight_best_objects(self, image, best_top_res, best_bot_res):
        vis_img = image.copy()
        targets = [best_top_res, best_bot_res]
        for res in targets:
            if res is None: continue
            if res.min_rect_box is not None:
                cv2.drawContours(vis_img, [res.min_rect_box], 0, (0, 255, 0), 4)
                cx, cy = res.center_x_px, res.center_y_px
                cv2.putText(vis_img, "SELECTED", (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return vis_img

    def process_image(self, image_or_path, output_dir, top_id, bot_id, image_name=None , debug_mode=False):
        start_time = time.time()
        
        image = None
        if isinstance(image_or_path, str):
            if image_name is None:
                image_name = os.path.splitext(os.path.basename(image_or_path))[0]
            try:
                stream = open(image_or_path.encode("utf-8"), "rb")
                bytes = bytearray(stream.read())
                numpy_array = np.asarray(bytes, dtype=np.uint8)
                image = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
                stream.close()
            except Exception as e:
                logger.error(f"Image Load Error: {e}")
                return [], None
        elif isinstance(image_or_path, np.ndarray):
            image = image_or_path
            if image_name is None:
                image_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            return [], None
        if image is None: return [], None
            
        t1 = time.time()
        img_w = image.shape[1]
        
        bushes, binary_img, ignored_blobs, total_contour_area = self.detect_bushes(image, output_dir, image_name , debug_mode)
        
        t2 = time.time()
        logger.info(f"[TIME] Detect Bushes: {t2 - t1:.4f}s (Cnt: {len(bushes)})")
        
        results = []
        for bush in bushes:
            res = BushInspectionResult()
            res.image_name = image_name
            res.bush_id = bush.id
            res.contour_area = bush.area
            res.center_x_px, res.center_y_px = bush.center
            res.center_x_mm = round((img_w - bush.center[0]) * self.pixel_to_mm, 3)
            res.center_y_mm = round(bush.center[1] * self.pixel_to_mm, 3)
            mr_ang, cn_ang, mr_box = self.calculate_angles(bush.contour, bush.corners, bush.center)
            res.min_rect_angle = round(mr_ang, 2)
            res.corner_angle = round(cn_ang, 2)
            res.min_rect_box = mr_box
            res.angle_diff = round(abs(res.min_rect_angle - res.corner_angle), 2)
            res.rotation_angle = res.min_rect_angle
            res.final_angle = res.rotation_angle 
            
            #res.surface_type = self.check_surface_type(image, bush.center)
            res.surface_type = self.check_surface_type(binary_img, bush.center)
            
            if res.surface_type == "TOP":
                self.set_target_product(top_id)
            else: 
                self.set_target_product(bot_id)
            
            # [수정] corners 전달
            res.text_type, res.text_blob_box, res.blob_stats, res.blob_distance, \
            res.edge_check_result, res.edge_length_sum, res.failed_blobs, res.edge_map = \
                self.detect_text_region(image, bush.contour, bush.center, bush.id, output_dir, image_name, corners=bush.corners, debug_dir=None , debug_mode=debug_mode)
                
            res.direction_code = self.determine_direction(bush.center, res.text_blob_box, res.rotation_angle)
            results.append(res)
            
        t3 = time.time()
        vis_img = self.visualize_result(image, results, bushes, ignored_blobs)
        t4 = time.time()
        logger.info(f"[TIME] Visualization: {t4 - t3:.4f}s | TOTAL: {t4 - start_time:.4f}s")
        
        return results, vis_img, total_contour_area

    def process_image_manual(self, image_or_path, output_dir, image_name=None):
        logger.info("=== Manual Inspection Start (Debug Mode ON) ===")
        start_time = time.time()
        
        image = None
        if isinstance(image_or_path, str):
            if image_name is None:
                image_name = os.path.splitext(os.path.basename(image_or_path))[0]
            try:
                stream = open(image_or_path.encode("utf-8"), "rb")
                bytes = bytearray(stream.read())
                numpy_array = np.asarray(bytes, dtype=np.uint8)
                image = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
                stream.close()
            except Exception as e:
                logger.error(f"Image Load Error: {e}")
                return [], None
        elif isinstance(image_or_path, np.ndarray):
            image = image_or_path
            if image_name is None:
                image_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            return [], None
        if image is None: return [], None
            
        img_w = image.shape[1]
        
        bushes, binary_img, ignored_blobs, total_contour_area = self.detect_bushes(image, output_dir, image_name , debug_mode=True)
        
        results = []
        for bush in bushes:
            res = BushInspectionResult()
            res.image_name = image_name
            res.bush_id = bush.id
            res.contour_area = bush.area
            res.center_x_px, res.center_y_px = bush.center
            res.center_x_mm = round((img_w - bush.center[0]) * self.pixel_to_mm, 3)
            res.center_y_mm = round(bush.center[1] * self.pixel_to_mm, 3)
            mr_ang, cn_ang, mr_box = self.calculate_angles(bush.contour, bush.corners, bush.center)
            res.min_rect_angle = round(mr_ang, 2)
            res.corner_angle = round(cn_ang, 2)
            res.min_rect_box = mr_box
            res.angle_diff = round(abs(res.min_rect_angle - res.corner_angle), 2)
            res.rotation_angle = res.min_rect_angle
            res.final_angle = res.rotation_angle 
            #res.surface_type = self.check_surface_type(image, bush.center)
            res.surface_type = self.check_surface_type(binary_img, bush.center)
            
            logger.info(f"Analyzing Bush ID: {bush.id} (Area: {int(bush.area)})")
            
            # [수정] corners 전달
            res.text_type, res.text_blob_box, res.blob_stats, res.blob_distance, \
            res.edge_check_result, res.edge_length_sum, res.failed_blobs, res.edge_map = \
                self.detect_text_region(image, bush.contour, bush.center, bush.id, output_dir, image_name, corners=bush.corners, debug_dir=output_dir)
                
            res.direction_code = self.determine_direction(bush.center, res.text_blob_box, res.rotation_angle)
            results.append(res)
            
        vis_img = self.visualize_result(image, results, bushes, ignored_blobs)
        
        logger.info(f"=== Manual Inspection Done (Total: {time.time() - start_time:.4f}s) ===")
        return results, vis_img