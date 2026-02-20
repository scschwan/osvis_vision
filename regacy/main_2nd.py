import cv2
import numpy as np
import os
import math
import json
import csv

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
        
        # [핵심 수정 1] default_params를 가장 먼저 정의 (순서 변경)
        # 이걸 먼저 선언해야 get_current_params()에서 에러가 안 납니다.
        self.default_params = {
            'outer_margin': 10, 'inner_margin': 10, 'min_area': 1050, 'max_area': 5000,      
            'min_short_side': 40, 'max_short_side': 100, 'min_long_side': 40, 'max_long_side': 100,   
            'min_ratio': 0.5, 'max_ratio': 1.5, 'min_dist': 150, 'max_dist': 210,
            'min_edge_length_sum': 240, 'dilation_iter': 3     
        }

        # 도면 기반 거리 (3.6 * sqrt(2) mm)
        target_dist_mm = 3.6 * math.sqrt(2)
        self.target_dist_px = target_dist_mm / self.pixel_to_mm

        # [핵심 수정 2] 절대 경로로 변환하여 파일 찾기 (경로 문제 해결)
        if config_path is None:
            # 현재 실행 중인 main.py 파일이 있는 폴더 경로를 구함
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "product_config.json")
            
        self.product_configs = self.load_config(config_path)
        
        # 현재 검사할 타겟 제품 ID (기본값 "1")
        self.current_product_id = "1"
        
        # 이제 default_params가 정의되어 있으므로 안전하게 호출 가능
        self.current_params = self.get_current_params()

    def load_config(self, path):
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    print(f"Loaded config from: {path}")
                    return json.load(f)
            except Exception as e:
                print(f"Config load failed: {e}")
                return {}
        else:
            print(f"Config file not found: {path}")
            return {}

    def set_target_product(self, product_id):
        """UI 등 외부에서 제품 ID를 변경할 때 호출"""
        self.current_product_id = str(product_id)
        self.current_params = self.get_current_params()
        print(f"Target Product Set to: {self.current_product_id}")

    def get_current_params(self):
        """현재 ID에 맞는 파라미터 반환 (없으면 1번이나 디폴트 반환)"""
        if self.current_product_id in self.product_configs:
            return self.product_configs[self.current_product_id]
        else:
            # 설정이 없으면 디폴트 리턴
            print(f"Warning: No config for ID {self.current_product_id}, using defaults.")
            return self.default_params

    def preprocess_image(self, image, output_dir=None, image_name=None):
        if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else: gray = image.copy()
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
        return binary

    def detect_bushes(self, image, output_dir=None, image_name=None):
        processed = self.preprocess_image(image, output_dir, image_name)
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_with_area = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            contours_with_area.append({'contour': cnt, 'area': area})
        contours_with_area.sort(key=lambda x: x['area'], reverse=True)
        bushes = []
        used_centers = []
        min_center_dist = 50
        
        # params에서 min/max area 가져오기 (전체 컨투어 필터링용)
        # Bush 전체 크기는 blob_params의 min_area와 다를 수 있지만, 
        # 여기서는 Bush 자체를 찾는 로직이므로 기존 하드코딩값(150000~170000) 유지 혹은 별도 설정 필요
        # 일단 기존 로직 유지 (Bush 자체 크기)
        BUSH_MIN_AREA = 150000
        BUSH_MAX_AREA = 170000

        for idx, item in enumerate(contours_with_area):
            cnt = item['contour']
            area = item['area']
            
            if area < BUSH_MIN_AREA or area > BUSH_MAX_AREA: continue
            
            center, corners = self.find_corner_center_simple(cnt)
            
            is_duplicate = False
            for uc in used_centers:
                dist = math.sqrt((center[0] - uc[0])**2 + (center[1] - uc[1])**2)
                if dist < min_center_dist:
                    is_duplicate = True
                    break
            if is_duplicate: continue
            
            used_centers.append(center)
            bush = BushInfo(cnt, center, area, corners)
            bush.id = len(bushes)
            bushes.append(bush)
        return bushes, processed

    # ... (find_corner_center_simple, calculate_diagonal_intersection, calculate_angles 등 기존 로직 유지) ...
    # 코드 길이상 중복되는 수학 함수들은 위와 동일하다고 가정하고 생략합니다. 
    # 실제 사용시엔 이전 답변의 함수들을 그대로 포함시켜야 합니다.
    
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

    def detect_rotation_angle(self, contour): # 구버전 함수 유지
        rect = cv2.minAreaRect(contour); return rect[2]

    # =========================================================================
    # [수정] Config 값 사용 (self.current_params)
    # =========================================================================
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
            
            # [수정] Config 값 사용
            threshold = self.current_params['min_edge_length_sum']
            
            if total_length >= threshold: return "edge_detect", total_length
            else: return "none_edge", total_length
        except Exception: return "none_edge", 0

    def detect_text_blob_region(self, image, contour, center, inner_r_base):
        # 파라미터 로딩 (Config)
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
        max_area = 0; has_text_blob = False; failed_blobs_data = [] 

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

    def visualize_result(self, image, results, bushes):
        vis_img = image.copy()
        img_h, img_w = image.shape[:2]
        
        # Config 값 사용
        p = self.current_params

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

    def process_image(self, image_path, output_dir):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        try: image = cv2.imread(image_path)
        except Exception: image = None
        if image is None: print(f"Error loading {image_path}"); return []
        print(f"Processing: {image_name}")
        img_w = image.shape[1]
        bushes, _ = self.detect_bushes(image, output_dir, image_name)
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

            res.surface_type = self.check_surface_type(image, bush.center)
            
            res.text_type, res.text_blob_box, res.blob_stats, res.blob_distance, \
            res.edge_check_result, res.edge_length_sum, res.failed_blobs, res.edge_map = \
                self.detect_text_region(image, bush.contour, bush.center, bush.id, output_dir, image_name)
            
            res.direction_code = self.determine_direction(bush.center, res.text_blob_box, res.rotation_angle)
            results.append(res)
            
        vis_img = self.visualize_result(image, results, bushes)
        cv2.imwrite(os.path.join(output_dir, f"result_{image_name}.jpg"), vis_img)
        vis_img = None
        for r in results:
            r.edge_map = None 
            r.failed_blobs = [] 
        return results

    def process_directory(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        valid_ext = ['.jpg', '.jpeg', '.png', '.bmp']
        files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_ext]
        all_results = []
        for f in files:
            path = os.path.join(input_dir, f)
            results = self.process_image(path, output_dir)
            all_results.extend(results)
        csv_path = os.path.join(output_dir, "inspection_results.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Image", "Bush_ID", "X_mm", "Y_mm", "MinRectAng", "CornerAng", "Diff", "Dir", "Type", "Edge_Len"])
            for r in all_results:
                writer.writerow([
                    r.image_name, r.bush_id, 
                    r.center_x_mm, r.center_y_mm, 
                    r.min_rect_angle, r.corner_angle, r.angle_diff,
                    r.direction_code, r.text_type, 
                    r.edge_length_sum
                ])
        print("Processing Completed.")

if __name__ == "__main__":
    input_path = r"images"
    output_path = r"result_python"
    
    config_path = r"product_config.json"
    
    # 1. Config 파일이 없으면 먼저 생성해야 함 (create_config.py 실행)
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}. Please run create_config.py first.")
    
    inspector = BushInspector(config_path=config_path)
    
    # [중요] 여기서 검사할 제품 번호를 설정합니다. (UI에서 받아올 값)
    # 예를 들어 10번 제품을 검사한다면:
    inspector.set_target_product(10)
    
    inspector.process_directory(input_path, output_path)