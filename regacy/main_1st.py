import cv2
import numpy as np
import os
import math
import json
import csv
# from paddleocr import PaddleOCR # [삭제] OCR 불필요

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
        self.rotation_angle = 0.0
        self.surface_type = ""
        self.text_type = "" 
        self.confidence = 0.0 # OCR 제거로 의미 없어짐 (0.0 고정)
        self.contour_area = 0.0
        
        self.text_blob_box = None 
        self.blob_stats = {} 
        self.blob_distance = 0.0 
        
        self.edge_check_result = "" 
        self.edge_length_sum = 0    
        self.edge_map = None        
        self.failed_blobs = [] 

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
    def __init__(self, pixel_to_mm=0.028761, min_area=150000, max_area=170000, debug_mode=True):
        self.pixel_to_mm = pixel_to_mm
        self.min_area = min_area
        self.max_area = max_area
        self.debug_mode = debug_mode
        self.center_check_radius = 10
        
        # [삭제] PaddleOCR 로딩 제거
        # self.ocr = ... 

        # ▼▼▼ [설정] 파라미터 튜닝 영역 ▼▼▼
        self.blob_params = {
            #'outer_margin': 15,    
            'outer_margin': 10,    
            'inner_margin': 10,    
            #'min_area': 1950,       
            'min_area': 1050,       
            'max_area': 5000,      
            
            # 길이 필터링
            'min_short_side': 40,  
            'max_short_side': 100,  
            'min_long_side': 40,   
            'max_long_side': 100,   
            
            # 비율 필터링
            'min_ratio': 0.5,      
            'max_ratio': 1.5, 
            
            # 거리 필터링 (중심점 ~ 글자 중심 거리)
            'min_dist': 150,
            'max_dist': 210,

            # [수정] 에지 길이 필터링 (400 미만이면 none_edge)
            'min_edge_length_sum': 240, 
            
            #'dilation_iter': 3     
            'dilation_iter': 3     
        }

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
        for idx, item in enumerate(contours_with_area):
            cnt = item['contour']
            area = item['area']
            if area < self.min_area or area > self.max_area: continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity < 0.3: continue
            center, corners = self.find_corner_center(cnt)
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

    def find_corner_center(self, contour):
        # [수정] MinAreaRect 기반의 안정적인 중심점 찾기
        M = cv2.moments(contour)
        if M["m00"] != 0: 
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else: 
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
        
        initial_center = (cx, cy)
        corners = []

        try:
            # MinAreaRect로 "이상적인 사각형"의 틀을 잡음
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box) 
            
            # 기하학적 중심
            geo_center = (int(rect[0][0]), int(rect[0][1]))
            
            # 모서리 (시각화에서는 제거하겠지만 데이터는 유지)
            corners = [tuple(p) for p in box]
            
            # 중심점 보정 (모멘트가 너무 튀면 기하학 중심 사용)
            diff = math.sqrt((initial_center[0] - geo_center[0])**2 + (initial_center[1] - geo_center[1])**2)
            
            if diff > 3.0: 
                return geo_center, corners
            else:
                # 차이가 작으면 모멘트보다 기하학 중심이 더 정밀할 확률이 높음 (원형 대칭이므로)
                return geo_center, corners

        except Exception: 
            pass
            
        return initial_center, corners

    def detect_rotation_angle(self, contour):
        rect = cv2.minAreaRect(contour)
        angle = rect[2]
        if angle < -45: angle = 90 + angle
        elif angle > 45: angle = angle - 90
        return angle

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
            threshold = self.blob_params['min_edge_length_sum']
            if total_length >= threshold: return "edge_detect", total_length
            else: return "none_edge", total_length
        except Exception: return "none_edge", 0

    def detect_text_blob_region(self, image, contour, center, inner_r_base):
        cx, cy = int(center[0]), int(center[1])
        h, w = image.shape[:2]
        if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else: gray = image.copy()

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        outer_margin = self.blob_params['outer_margin']
        if outer_margin > 0:
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (outer_margin, outer_margin))
            mask = cv2.erode(mask, kernel_erode)
        inner_margin = self.blob_params['inner_margin']
        safe_inner_r = inner_r_base + inner_margin 
        cv2.circle(mask, (cx, cy), safe_inner_r, 0, -1)

        edges = cv2.Canny(gray, 50, 150)
        filtered_edges = cv2.bitwise_and(edges, edges, mask=mask)
        
        iter_count = self.blob_params['dilation_iter']
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
        dilated = cv2.dilate(filtered_edges, kernel_dilate, iterations=iter_count)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_blob_box = None; best_stats = {}; best_distance = 0.0
        best_edge_result = "none_edge"; best_edge_len = 0
        max_area = 0; has_text_blob = False; failed_blobs_data = [] 

        p_min_area = self.blob_params['min_area']
        p_max_area = self.blob_params['max_area']
        p_min_ratio = self.blob_params['min_ratio']
        p_max_ratio = self.blob_params['max_ratio']
        p_min_short = self.blob_params['min_short_side']
        p_max_short = self.blob_params['max_short_side']
        p_min_long = self.blob_params['min_long_side']
        p_max_long = self.blob_params['max_long_side']
        p_min_dist = self.blob_params['min_dist']
        p_max_dist = self.blob_params['max_dist']

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50: continue
            rect = cv2.minAreaRect(cnt)
            (rect_w, rect_h) = rect[1]; (rcx, rcy) = rect[0]
            short_side = min(rect_w, rect_h); long_side = max(rect_w, rect_h)
            aspect_ratio = float(short_side) / long_side if long_side > 0 else 0
            dist_from_center = math.sqrt((cx - rcx)**2 + (cy - rcy)**2)
            
            current_stats = {'area': int(area), 'short': int(short_side), 'long': int(long_side), 'ratio': round(aspect_ratio, 2)}

            if p_min_area <= area <= p_max_area:
                failed_blobs_data.append({'rect': rect, 'dist': dist_from_center, 'stats': current_stats})

            is_valid_area = p_min_area < area < p_max_area
            is_valid_ratio = p_min_ratio < aspect_ratio < p_max_ratio
            is_valid_size = (p_min_short < short_side < p_max_short) and (p_min_long < long_side < p_max_long)
            is_valid_dist = p_min_dist <= dist_from_center <= p_max_dist

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

    # =========================================================================
    # [수정] OCR 호출 제거, 오직 Blob 검출로만 판단
    # =========================================================================
    def detect_text_region(self, image, contour, center, bush_id, output_dir, image_name):
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0: return "back", None, {}, 0.0, "", 0, [], None

        bush_rect = cv2.minAreaRect(contour)
        bush_radius = min(bush_rect[1]) / 2
        estimated_inner_r = int(bush_radius * 0.5) 

        # Blob 검출 수행
        has_blob, blob_box, blob_stats, blob_dist, edge_result, edge_len, failed_blobs, edge_map = \
            self.detect_text_blob_region(image, contour, center, estimated_inner_r)

        # 최종 판단 (OCR 없이 Blob 존재 여부만으로)
        if has_blob:
             return "front", blob_box, blob_stats, blob_dist, edge_result, edge_len, failed_blobs, edge_map
        else:
            return "back", None, {}, 0.0, "", 0, failed_blobs, edge_map

    # =========================================================================
    # [수정] 시각화 (보라색 점 제거)
    # =========================================================================
    def visualize_result(self, image, results, bushes):
        vis_img = image.copy()
        
        p_min_dist = self.blob_params['min_dist']
        p_max_dist = self.blob_params['max_dist']
        p_min_area = self.blob_params['min_area']
        p_max_area = 5000 

        for i, (result, bush) in enumerate(zip(results, bushes)):
            contour = bush.contour; center = bush.center
            # 컨투어 그리기
            cv2.drawContours(vis_img, [contour], -1, (0, 255, 0), 2)
            
            # 1. 실패한 후보군 (Red)
            for f_data in result.failed_blobs:
                f_rect = f_data['rect']; f_dist = f_data['dist']; f_stats = f_data['stats']
                box_pts = cv2.boxPoints(f_rect); box_pts = np.int32(box_pts)
                
                cond1 = p_min_dist <= f_dist <= p_max_dist
                cond2 = p_min_area <= f_stats['area'] <= p_max_area
                
                if cond1 and cond2:
                    cv2.drawContours(vis_img, [box_pts], 0, (0, 0, 255), 1)
                    line1 = f"Sz:{f_stats['short']}x{f_stats['long']}"
                    text_x = int(box_pts[0][0]); text_y = int(box_pts[0][1]) - 5
                    cv2.putText(vis_img, line1, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # 2. 성공한 Blob (Green Edges + Cyan/Yellow Box)
            if result.text_blob_box is not None:
                box = cv2.boxPoints(result.text_blob_box); box = np.int32(box)
                edge_res = result.edge_check_result; edge_len = result.edge_length_sum
                
                if edge_res == "edge_detect":
                    box_color = (255, 255, 0) # Cyan
                    text_color = (255, 255, 0)
                    if result.edge_map is not None:
                        mask_box = np.zeros_like(result.edge_map)
                        cv2.drawContours(mask_box, [box], 0, 255, -1)
                        masked_edges = cv2.bitwise_and(result.edge_map, result.edge_map, mask=mask_box)
                        y_idxs, x_idxs = np.where(masked_edges > 0)
                        vis_img[y_idxs, x_idxs] = [0, 255, 0] # Green Pixel (내부 에지만 초록색)
                else:
                    box_color = (0, 255, 255) # Yellow
                    text_color = (0, 255, 255)

                cv2.drawContours(vis_img, [box], 0, box_color, 2) 
                
                stats = result.blob_stats; dist_val = result.blob_distance
                if stats:
                    line1 = f"Sz:{stats['short']}x{stats['long']}"
                    line2 = f"A:{stats['area']} Dst:{int(dist_val)}"
                    line3 = f"{edge_res} L:{edge_len}" 
                    text_x = int(box[0][0]); text_y = int(box[0][1]) - 35
                    cv2.putText(vis_img, line1, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                    cv2.putText(vis_img, line2, (text_x, text_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                    cv2.putText(vis_img, line3, (text_x, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

            # [수정] 보라색 점 삭제, 중심점만 표시
            cv2.circle(vis_img, (int(center[0]), int(center[1])), 8, (0, 0, 255), -1)
            
            text_lines = [f"ID:{result.bush_id}", f"Type:{result.text_type}"]
            text_y = int(center[1]) - 120
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
        bushes, _ = self.detect_bushes(image, output_dir, image_name)
        results = []
        for bush in bushes:
            res = BushInspectionResult()
            res.image_name = image_name
            res.bush_id = bush.id
            res.contour_area = bush.area
            res.center_x_px, res.center_y_px = bush.center
            res.center_x_mm = round(res.center_x_px * self.pixel_to_mm, 3)
            res.center_y_mm = round(res.center_y_px * self.pixel_to_mm, 3)
            res.rotation_angle = round(self.detect_rotation_angle(bush.contour), 2)
            res.surface_type = self.check_surface_type(image, bush.center)
            
            # [수정] 리턴값 간소화
            res.text_type, res.text_blob_box, res.blob_stats, res.blob_distance, \
            res.edge_check_result, res.edge_length_sum, res.failed_blobs, res.edge_map = \
                self.detect_text_region(image, bush.contour, bush.center, bush.id, output_dir, image_name)
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
            writer.writerow(["Image", "Bush_ID", "Type", "Blob_Detected", "Edge_Result", "Edge_Len"])
            for r in all_results:
                writer.writerow([r.image_name, r.bush_id, r.text_type, 
                                 "Yes" if r.text_blob_box else "No", r.edge_check_result, r.edge_length_sum])
        print("Processing Completed.")

if __name__ == "__main__":
    input_path = r"C:\workspace2025\osvis vision_python\images"
    output_path = r"C:\workspace2025\osvis vision_python\result_python"
    
    inspector = BushInspector()
    inspector.process_directory(input_path, output_path)