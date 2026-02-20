import cv2
import numpy as np
import os
import math
import csv
import json
from datetime import datetime

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
            'outer_margin': 10, 'inner_margin': 10, 'min_area': 1050, 'max_area': 5000,      
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
            "thresh_c": 3
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

    def preprocess_image(self, image, output_dir=None, image_name=None):
        if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else: gray = image.copy()
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        block_size = int(self.thresh_block_size)
        if block_size % 2 == 0: block_size += 1
        
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 
            block_size, self.thresh_c
        )
        return binary

    def detect_bushes(self, image, output_dir=None, image_name=None):
        processed = self.preprocess_image(image, output_dir, image_name)
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contours_with_area = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1.0:
                contours_with_area.append({'contour': cnt, 'area': area})
        contours_with_area.sort(key=lambda x: x['area'], reverse=True)
        
        bushes = []
        small_blobs = []
        
        # 1차 필터링
        candidates = []
        min_area = self.bush_min_area
        max_area = self.bush_max_area
        visual_min_area = min_area * 0.5

        for item in contours_with_area:
            cnt = item['contour']
            area = item['area']
            
            if area < min_area:
                if area > visual_min_area:
                    rect = cv2.minAreaRect(cnt)
                    center = (int(rect[0][0]), int(rect[0][1]))
                    small_blobs.append({'contour': cnt, 'area': area, 'center': center})
                    # print(f"  [SKIP] Small Candidate: Area={area:.1f} (Min={min_area})")
                continue
            
            if area > max_area:
                # print(f"  [SKIP] Too Large: Area={area:.1f}")
                continue
                
            center, corners = self.find_corner_center_simple(cnt)
            candidates.append({
                'contour': cnt, 'area': area, 'center': center, 'corners': corners
            })

        # 2차 필터링: 중심점 거리 기반 겹침 확인 (NMS)
        overlap_indices = set()
        min_dist_px = self.min_center_dist_px
        
        count = len(candidates)
        for i in range(count):
            if i in overlap_indices: continue
            
            for j in range(i + 1, count):
                c1 = candidates[i]['center']
                c2 = candidates[j]['center']
                dist = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                
                if dist < min_dist_px:
                    overlap_indices.add(i)
                    overlap_indices.add(j)
                    print(f"  [OVERLAP] Dist={dist:.1f}px. Removing both.")

        for i, cand in enumerate(candidates):
            if i in overlap_indices:
                continue
            
            bush = BushInfo(cand['contour'], cand['center'], cand['area'], cand['corners'])
            bush.id = len(bushes)
            bushes.append(bush)
            
        return bushes, processed, small_blobs

    def find_corner_center_simple(self, contour):
        rect = cv2.minAreaRect(contour)
        cx, cy = int(rect[0][0]), int(rect[0][1])
        center = (cx, cy)
        candidates = []
        for pt in contour:
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
        return "TOP" if mean_val > 200 else "BOT"

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
            blob_edges = cv2.Canny(crop, 50, 150)
            edge_contours, _ = cv2.findContours(blob_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_length = 0
            for ec in edge_contours:
                length = cv2.arcLength(ec, False)
                if length > 5: total_length += length
            
            threshold = self.current_params['min_edge_length_sum']
            if total_length >= threshold: return "edge_detect", total_length
            else: return "none_edge", total_length
        except Exception: return "none_edge", 0

    def detect_text_blob_region(self, image, contour, center, inner_r_base):
        p = self.current_params
        cx, cy = int(center[0]), int(center[1])
        h, w = image.shape[:2]
        if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else: gray = image.copy()

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        if p['outer_margin'] > 0:
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (p['outer_margin'], p['outer_margin']))
            mask = cv2.erode(mask, kernel_erode)
        safe_inner_r = inner_r_base + p['inner_margin'] 
        cv2.circle(mask, (cx, cy), safe_inner_r, 0, -1)

        edges = cv2.Canny(gray, 50, 150)
        filtered_edges = cv2.bitwise_and(edges, edges, mask=mask)
        
        iter_count = p['dilation_iter']
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
        dilated = cv2.dilate(filtered_edges, kernel_dilate, iterations=iter_count)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_blob_box = None; best_stats = {}; best_distance = 0.0
        best_edge_result = "none_edge"; best_edge_len = 0
        max_area = 0; has_text_blob = False; failed_blobs_data = [] # 변수 선언 확인

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50: continue
            rect = cv2.minAreaRect(cnt)
            (rect_w, rect_h) = rect[1]; (rcx, rcy) = rect[0]
            short_side = min(rect_w, rect_h); long_side = max(rect_w, rect_h)
            aspect_ratio = float(short_side) / long_side if long_side > 0 else 0
            dist_from_center = math.sqrt((cx - rcx)**2 + (cy - rcy)**2)
            current_stats = {'area': int(area), 'short': int(short_side), 'long': int(long_side), 'ratio': round(aspect_ratio, 2)}

            if p['min_area'] <= area <= p['max_area']:
                failed_blobs_data.append({'rect': rect, 'dist': dist_from_center, 'stats': current_stats})

            is_valid_area = p['min_area'] < area < p['max_area']
            is_valid_ratio = p['min_ratio'] < aspect_ratio < p['max_ratio']
            is_valid_size = (p['min_short_side'] < short_side < p['max_short_side']) and \
                            (p['min_long_side'] < long_side < p['max_long_side'])
            is_valid_dist = p['min_dist'] <= dist_from_center <= p['max_dist']

            if is_valid_area and is_valid_ratio and is_valid_size and is_valid_dist:
                if area > max_area:
                    max_area = area
                    best_blob_box = rect
                    has_text_blob = True
                    best_distance = dist_from_center
                    best_stats = current_stats
                    e_res, e_len = self.check_text_like_edges(image, rect)
                    best_edge_result = e_res
                    best_edge_len = int(e_len)
        
        # [수정] 오타 수정 (failed_blobs -> failed_blobs_data)
        return has_text_blob, best_blob_box, best_stats, best_distance, \
               best_edge_result, best_edge_len, failed_blobs_data, filtered_edges

    def detect_text_region(self, image, contour, center, bush_id, output_dir, image_name):
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0: return "back", None, {}, 0.0, "", 0, [], None
        bush_rect = cv2.minAreaRect(contour)
        bush_radius = min(bush_rect[1]) / 2
        estimated_inner_r = int(bush_radius * 0.5) 
        
        has_blob, blob_box, blob_stats, blob_dist, edge_result, edge_len, failed_blobs, edge_map = \
            self.detect_text_blob_region(image, contour, center, estimated_inner_r)
            
        if has_blob:
             return "front", blob_box, blob_stats, blob_dist, edge_result, edge_len, failed_blobs, edge_map
        else:
            return "back", None, {}, 0.0, "", 0, failed_blobs, edge_map

    def visualize_result(self, image, results, bushes, small_blobs=[]):
        vis_img = image.copy()
        img_h, img_w = image.shape[:2]
        p = self.current_params

        # 1. 작아서 탈락한 후보군 (Red)
        for sb in small_blobs:
            cnt = sb['contour']
            area = sb['area']
            center = sb['center']
            cv2.drawContours(vis_img, [cnt], -1, (0, 0, 255), 2)
            text_x, text_y = center[0] - 20, center[1]
            label = f"Small: {int(area)}"
            cv2.putText(vis_img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 2. 정상 검출 Bush 시각화
        for i, (result, bush) in enumerate(zip(results, bushes)):
            contour = bush.contour; center = bush.center; corners = bush.corners
            cv2.drawContours(vis_img, [contour], -1, (0, 255, 0), 2)
            if result.min_rect_box is not None:
                cv2.drawContours(vis_img, [result.min_rect_box], 0, (0, 165, 255), 1)
            if corners and len(corners) == 4:
                for pt in corners:
                    cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 6, (255, 0, 255), -1)
            for f_data in result.failed_blobs:
                f_rect = f_data['rect']; f_dist = f_data['dist']; f_stats = f_data['stats']
                box_pts = cv2.boxPoints(f_rect); box_pts = np.int32(box_pts)
                cond1 = p['min_dist'] <= f_dist <= p['max_dist']
                cond2 = p['min_area'] <= f_stats['area'] <= p['max_area']
                if cond1 and cond2:
                    cv2.drawContours(vis_img, [box_pts], 0, (0, 0, 255), 1)
                    line1 = f"Sz:{f_stats['short']}x{f_stats['long']}"
                    text_x = int(box_pts[0][0]); text_y = int(box_pts[0][1]) - 5
                    cv2.putText(vis_img, line1, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            if result.text_blob_box is not None:
                box = cv2.boxPoints(result.text_blob_box); box = np.int32(box)
                edge_res = result.edge_check_result; edge_len = result.edge_length_sum
                if edge_res == "edge_detect":
                    box_color = (255, 255, 0); text_color = (255, 255, 0)
                    if result.edge_map is not None:
                        mask_box = np.zeros_like(result.edge_map)
                        cv2.drawContours(mask_box, [box], 0, 255, -1)
                        masked_edges = cv2.bitwise_and(result.edge_map, result.edge_map, mask=mask_box)
                        y_idxs, x_idxs = np.where(masked_edges > 0)
                        vis_img[y_idxs, x_idxs] = [0, 255, 0] 
                else:
                    box_color = (0, 255, 255); text_color = (0, 255, 255)
                cv2.drawContours(vis_img, [box], 0, box_color, 2) 
                blob_center = result.text_blob_box[0]
                cv2.line(vis_img, (int(center[0]), int(center[1])), 
                         (int(blob_center[0]), int(blob_center[1])), (255, 255, 0), 2)
                stats = result.blob_stats; dist_val = result.blob_distance
                if stats:
                    line1 = f"Sz:{stats['short']}x{stats['long']}"
                    line2 = f"A:{stats['area']} Dst:{int(dist_val)}"
                    line3 = f"{edge_res} L:{edge_len}" 
                    text_x = int(box[0][0]); text_y = int(box[0][1]) - 35
                    cv2.putText(vis_img, line1, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                    cv2.putText(vis_img, line2, (text_x, text_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                    cv2.putText(vis_img, line3, (text_x, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
            cv2.circle(vis_img, (int(center[0]), int(center[1])), 8, (0, 0, 255), -1)
            ang_rad = math.radians(result.min_rect_angle) 
            end_x = int(center[0] + 60 * math.cos(ang_rad))
            end_y = int(center[1] + 60 * math.sin(ang_rad))
            cv2.line(vis_img, (int(center[0]), int(center[1])), (end_x, end_y), (0, 165, 255), 2)
            cv2.line(vis_img, (int(center[0]), int(center[1])), (int(center[0]) + 60, int(center[1])), (255, 0, 0), 2)
            target_x_mm = (img_w - center[0]) * self.pixel_to_mm
            target_y_mm = center[1] * self.pixel_to_mm
            dir_map = {0: "N/A", 1: "E", 2: "S", 3: "W", 4: "N"}
            dir_str = dir_map.get(result.direction_code, "N/A")
            text_lines = [
                f"ID:{result.bush_id} {result.text_type}",
                f"XY:({target_x_mm:.1f}, {target_y_mm:.1f})",
                f"MA:{result.min_rect_angle:.1f} CA:{result.corner_angle:.1f}", 
                f"Diff:{result.angle_diff:.1f} Dir:{dir_str}"
            ]
            text_y = int(center[1]) - 140
            for j, line in enumerate(text_lines):
                y_offset = text_y + j * 25
                cv2.putText(vis_img, line, (int(center[0]) + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
                cv2.putText(vis_img, line, (int(center[0]) + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return vis_img

    def process_image(self, image_or_path, output_dir, image_name=None):
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
                print(f"Image Load Error: {e}")
                return [], None
        elif isinstance(image_or_path, np.ndarray):
            image = image_or_path
            if image_name is None:
                image_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            return [], None
        if image is None: return [], None
            
        print(f"Processing: {image_name}")
        img_w = image.shape[1]
        
        bushes, binary_img, small_blobs = self.detect_bushes(image, output_dir, image_name)
        
        os.makedirs(output_dir, exist_ok=True)
        binary_save_path = os.path.join(output_dir, f"binary_{image_name}.jpg")
        try:
            res, encoded_bin = cv2.imencode(".jpg", binary_img)
            if res:
                with open(binary_save_path, mode='w+b') as f:
                    encoded_bin.tofile(f)
        except Exception as e:
            print(f"Binary Save Error: {e}")

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
            res.surface_type = self.check_surface_type(image, bush.center)
            res.text_type, res.text_blob_box, res.blob_stats, res.blob_distance, \
            res.edge_check_result, res.edge_length_sum, res.failed_blobs, res.edge_map = \
                self.detect_text_region(image, bush.contour, bush.center, bush.id, output_dir, image_name)
            res.direction_code = self.determine_direction(bush.center, res.text_blob_box, res.rotation_angle)
            results.append(res)
            
        vis_img = self.visualize_result(image, results, bushes, small_blobs)
        return results, vis_img 

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