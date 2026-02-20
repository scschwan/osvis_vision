import asyncio
import time
import math
import logging
import threading
import os
import sys
import cv2
import numpy as np
import shutil
from datetime import datetime

# 사용자 모듈 임포트
from vision_server import VisionServer, GlobalState
from feeder_control import FeederController
from camera_module import CameraController
from bush_inspector import BushInspector

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][MAIN] %(message)s')
logger = logging.getLogger("MainApp")

class InspectionSystem:
    def __init__(self):
        # 1. 경로 설정
        if getattr(sys, 'frozen', False):
            self.base_dir = os.path.dirname(sys.executable)
        else:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        config_path = os.path.join(self.base_dir, "product_config.json")
        
        logger.info(f"Base Dir: {self.base_dir}")
        logger.info(f"Config Path: {config_path}")

        # 2. 하드웨어/모듈 초기화
        self.feeder = FeederController(ip='192.168.1.100', port=502)
        self.camera = CameraController()
        self.inspector = BushInspector(config_path=config_path, debug_mode=False)
        
        self.IMG_CENTER_X = 2736
        self.IMG_CENTER_Y = 1577
        
        self.last_sent_status = "EMPTY"
        self.retry_cnt = 0

        # [추가] 현재 검사 중인 제품 ID 저장 변수
        self.current_top_id = "-"
        self.current_bot_id = "-"
        
    def initialize(self):
        """시스템 초기화 (하드웨어 연결)"""
        logger.info("=== 시스템 초기화 시작 ===")
        
        if self.feeder.connect():
            logger.info("✓ Feeder 연결 성공")
            self.feeder.setup_formula_mode()
        else:
            logger.error("✗ Feeder 연결 실패")
            # return False # 테스트 시 주석 처리

        if self.camera.connect_camera():
            logger.info("✓ Camera 연결 성공")
            self.camera.start_capture()
        else:
            logger.error("✗ Camera 연결 실패")
            # return False # 테스트 시 주석 처리
            
        GlobalState.readyYN = 1
        logger.info("✓ 시스템 준비 완료")
        return True

    def _manage_disk_space(self):
        try:
            total, used, free = shutil.disk_usage(self.base_dir)
            usage_percent = (used / total) * 100
            
            if usage_percent >= 90.0:
                logger.warning(f"⚠️ Disk Usage High ({usage_percent:.2f}%). Cleaning up old data...")
                target_dirs = ["original", "result"]
                for sub_dir in target_dirs:
                    target_path = os.path.join(self.base_dir, sub_dir)
                    if not os.path.exists(target_path): continue
                    folders = sorted([f for f in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, f))])
                    delete_targets = folders[:10] 
                    if not delete_targets: continue
                    logger.info(f"[{sub_dir}] Deleting {len(delete_targets)} old folders...")
                    for folder_name in delete_targets:
                        full_path = os.path.join(target_path, folder_name)
                        try:
                            shutil.rmtree(full_path)
                            logger.info(f"  - Deleted: {folder_name}")
                        except Exception as e:
                            logger.error(f"  - Failed to delete {folder_name}: {e}")
        except Exception as e:
            logger.error(f"Disk Management Error: {e}")

    def process_inspection_scenario(self, top_prod_id, bot_prod_id):
        # [추가] 들어온 요청 ID를 멤버 변수에 저장 (UI 표기용)
        self.current_top_id = str(top_prod_id)
        self.current_bot_id = str(bot_prod_id)
        
        """제품 검출 시나리오 수행"""
        self.retry_cnt = 0
        max_retries = self.inspector.max_retry_count
        
        final_img = None
        all_candidates = {}
        msg = "#EMPTY,0,0,0,0,0,0,0,0;"
        
        while self.retry_cnt <= max_retries:
            logger.info(f"--- 검사 시도 {self.retry_cnt + 1}/{max_retries + 1} ---")
            cycle_start = time.time()
            
            # 1. Feeder 동작
            self.feeder.client.write_register(2, 2, slave=1) # Light ON
            self.feeder.client.write_register(0, 1, slave=1)
            time.sleep(0.3)
            
            # 2. 이미지 그랩
            image = self.camera.get_latest_image()
            
            # Feeder Light OFF
            self.feeder.client.write_register(2, 3, slave=1)
            self.feeder.client.write_register(0, 1, slave=1)

            if image is not None:
                final_img = image.copy() 
                self._manage_disk_space()

                now_dt = datetime.now()
                date_str = now_dt.strftime("%Y%m%d")
                time_str = now_dt.strftime("%H%M%S_%f")[:10]
                
                orig_dir = os.path.join(self.base_dir, "original", date_str)
                res_dir = os.path.join(self.base_dir, "result", date_str)
                os.makedirs(orig_dir, exist_ok=True)
                os.makedirs(res_dir, exist_ok=True)
                
                orig_filename = f"{time_str}_raw.jpg"
                cv2.imwrite(os.path.join(orig_dir, orig_filename), image)
                
                # [수정] total_contour_area 받아옴
                results_all, vis_img_all, total_contour_area = self.inspector.process_image(
                    image, res_dir, top_prod_id, bot_prod_id, image_name=f"{time_str}_Unified" , debug_mode=False
                )
                
                if vis_img_all is not None:
                    final_img = vis_img_all.copy()

                valid_tops = [
                    r for r in results_all 
                    if r.surface_type == "TOP" and r.text_type == "front" and r.edge_check_result == "edge_detect"
                ]
                valid_bots = [
                    r for r in results_all 
                    if r.surface_type == "BOT" and r.text_type == "front" and r.edge_check_result == "edge_detect"
                ]
                
                best_top = self._find_closest_to_center(valid_tops)
                best_bot = self._find_closest_to_center(valid_bots)
                
                # [추가] 1) 제품 수량 추정 (평균 면적 이용)
                # min_area, max_area의 중간값 사용
                min_a = self.inspector.bush_min_area
                max_a = self.inspector.bush_max_area
                product_avg_area = (min_a + max_a) / 2.0
                
                estimated_count = 0
                if product_avg_area > 0:
                    estimated_count = int(total_contour_area / product_avg_area)
                
                logger.info(f"Total ROI Area: {int(total_contour_area)}, Avg Area: {int(product_avg_area)} -> Est Count: {estimated_count}")

                # [수정] UI로 보낼 데이터 갱신 (추정 수량, Valid 개수 포함)
                all_candidates = {
                    "tops": valid_tops, 
                    "bots": valid_bots,
                    "est_count": estimated_count,  # UI 표시용
                    "valid_top_cnt": len(valid_tops), # UI 표시용
                    "valid_bot_cnt": len(valid_bots)  # UI 표시용
                }

                if best_top and best_bot:
                    final_img = self.inspector.highlight_best_objects(final_img, best_top, best_bot)
                    cv2.imwrite(os.path.join(res_dir, f"result_{time_str}_FINAL.jpg"), final_img)

                    total_sets = min(len(valid_tops), len(valid_bots))
                    header = "#OK2" if total_sets < 2 else "#OK"
                    self.last_sent_status = "OK2" if total_sets < 2 else "OK"
                    
                    if header == "#OK2":
                        logger.info(f"Low Density (Sets: {total_sets}). Status=OK2")

                    msg = (f"{header},"
                           f"{best_top.center_x_mm},{best_top.center_y_mm},{best_top.final_angle},{best_top.direction_code},"
                           f"{best_bot.center_x_mm},{best_bot.center_y_mm},{best_bot.final_angle},{best_bot.direction_code};")
                    
                    logger.info(f"검출 성공: {msg}")
                    self.retry_cnt = 0
                    return msg, final_img, all_candidates
                else:                     
                    bt_info = best_top.bush_id if best_top else "None"
                    bb_info = best_bot.bush_id if best_bot else "None"
                    logger.info(f"검출 실패 : valid_tops : {len(valid_tops)} => best_top : {bt_info}, valid_bots : {len(valid_bots)} => best_bot : {bb_info}")     

                    self.last_sent_status = "EMPTY"
                    msg = "#EMPTY,0,0,0,0,0,0,0,0;"
                    cv2.imwrite(os.path.join(res_dir, f"result_{time_str}_FAIL.jpg"), final_img)

            # 재시도 로직
            self.retry_cnt += 1
            if self.retry_cnt <= max_retries:
                logger.info(f"재시도 {self.retry_cnt}: 피더 진동")
                self.feeder.run_vibration(formula_number=1, duration_ms=1000) 
                time.sleep(1.3)
                continue
            
        logger.error("최대 재시도 초과. 검출 실패.")
        
        # [추가] 3) 재시도 초과 시, 상세 실패 메시지 분기 (#EMPTYTOP, #EMPTYBOT)
        # 마지막 시도의 valid_tops, valid_bots 개수 기준
        cnt_top = len(valid_tops)
        cnt_bot = len(valid_bots)
        
        if cnt_top < 3 and cnt_bot >= 3:
            msg = "#EMPTYTOP,0,0,0,0,0,0,0,0;"
            logger.info("Fail Reason: Lack of TOP -> #EMPTYTOP")
        elif cnt_bot < 3 and cnt_top >= 3:
            msg = "#EMPTYBOT,0,0,0,0,0,0,0,0;"
            logger.info("Fail Reason: Lack of BOT -> #EMPTYBOT")
        else:
            # 둘 다 부족하거나 기타 상황 -> #EMPTY (기존)
            msg = "#EMPTY,0,0,0,0,0,0,0,0;"
            logger.info("Fail Reason: Both Insufficient -> #EMPTY")

        self.retry_cnt = 0
        if final_img is None:
            final_img = np.zeros((480, 640, 3), dtype=np.uint8)

        return msg, final_img, all_candidates

    def _find_closest_to_center(self, results):
        if not results: return None
        best_res = None
        min_dist = float('inf')
        for res in results:
            dist = (res.center_x_px - self.IMG_CENTER_X)**2 + (res.center_y_px - self.IMG_CENTER_Y)**2
            if dist < min_dist:
                min_dist = dist
                best_res = res
        return best_res

    def handle_manual_trigger(self):
        logger.info(f"Manual Trigger Received. Last Status: {self.last_sent_status}")
        if self.last_sent_status == "OK2":
            logger.info("Executing Feeder Vibe (Reason: Previous #OK2)")
            #self.feeder.run_vibration(formula_number=1, duration_ms=1000)
            # [수정] 별도 스레드(비동기)로 진동 실행 (메인 로직 멈춤 방지)
            threading.Thread(target=self.feeder.run_vibration, args=(1, 1000), daemon=True).start()
        else:
            logger.info("No Action Required")

class IntegratedVisionServer(VisionServer):
    def __init__(self, system_logic):
        super().__init__(host='0.0.0.0', port=8000)
        self.system = system_logic

    def perform_inspection(self, p1, p2):
        if GlobalState.autoYN != 1:
            logger.warning("자동 모드 아님. 강제 수행.")
        
        try:
            msg, img, cands = self.system.process_inspection_scenario(p1, p2)
            return msg 
        except Exception as e:
            logger.error(f"Scenario Error: {e}")
            return "#ERR,SCENARIO;"

    def handle_manual_command(self):
        try:
            self.system.handle_manual_trigger()
            return True
        except Exception as e:
            logger.error(f"Manual Command Error: {e}")
            return False

async def main():
    system = InspectionSystem()
    if not system.initialize():
        logger.error("시스템 초기화 실패 (하드웨어 연결 확인 필요).")
    server = IntegratedVisionServer(system)
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("프로그램 종료 요청")
    except Exception as e:
        logger.error(f"서버 실행 중 치명적 오류: {e}")
    finally:
        system.feeder.disconnect()
        system.camera.stop()

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())