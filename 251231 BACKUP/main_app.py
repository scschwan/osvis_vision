import asyncio
import time
import math
import logging
import os
import sys
import cv2  # <--- [추가] 원본 저장을 위해 필요
from datetime import datetime # <--- [추가] 날짜 폴더 생성을 위해 필요

# 각 모듈 임포트
from vision_server import VisionServer, GlobalState
from feeder_control import FeederController
from camera_module import CameraController
from bush_inspector import BushInspector

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][MAIN] %(message)s')
logger = logging.getLogger("MainApp")

class InspectionSystem:
    def __init__(self):
        # 1. 경로 설정 (Base Directory 기준)
        if getattr(sys, 'frozen', False):
            self.base_dir = os.path.dirname(sys.executable)
        else:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Config 파일 경로
        config_path = os.path.join(self.base_dir, "product_config.json")
        
        # [수정] 디버그 폴더 대신 original/result 폴더 구조는 
        # process_inspection_scenario 내부에서 날짜별로 동적 생성합니다.
        
        logger.info(f"Base Dir: {self.base_dir}")
        logger.info(f"Config Path: {config_path}")

        # 2. 모듈 초기화
        self.feeder = FeederController(ip='192.168.1.100', port=502)
        self.camera = CameraController()
        
        # 검사기 초기화
        self.inspector = BushInspector(config_path=config_path, debug_mode=False)
        
        # 이미지 중심점 (고정값)
        self.IMG_CENTER_X = 2736
        self.IMG_CENTER_Y = 1577
        
        # 재시도 카운트
        self.retry_cnt = 0
        
    def initialize(self):
        """시스템 초기화 (연결 및 준비)"""
        logger.info("=== 시스템 초기화 시작 ===")
        
        if self.feeder.connect():
            logger.info("✓ Feeder 연결 성공")
            self.feeder.setup_formula_mode()
        else:
            logger.error("✗ Feeder 연결 실패")
            return False

        if self.camera.connect_camera():
            logger.info("✓ Camera 연결 성공")
            self.camera.start_capture()
        else:
            logger.error("✗ Camera 연결 실패")
            return False
            
        GlobalState.readyYN = 1
        logger.info("✓ 시스템 준비 완료 (GlobalState.readyYN = 1)")
        return True

    def process_inspection_scenario(self, top_prod_id, bot_prod_id):
        """제품 검출 시나리오 수행 (이미지 저장 로직 추가됨)"""
        self.retry_cnt = 0

        # [수정] Global Config의 Max Retry Count 사용
        max_retries = self.inspector.max_retry_count
        
        while self.retry_cnt <= 3:
            logger.info(f"--- 검사 시도 {self.retry_cnt + 1}/{max_retries + 1} (TOP:{top_prod_id}, BOT:{bot_prod_id}) ---")
            
            # 2-1) Feeder 동작
            self.feeder.client.write_register(2, 2, slave=1) # Formula 2 (Light ON)
            self.feeder.client.write_register(0, 1, slave=1) # Start
            time.sleep(0.3)
            
            # 이미지 그랩
            image = self.camera.get_latest_image()
            if image is None:
                logger.error("이미지 그랩 실패")
            
            # 2-3) Feeder 동작
            self.feeder.client.write_register(2, 3, slave=1) # Formula 3 (Light OFF)
            self.feeder.client.write_register(0, 1, slave=1) # Start

            if image is not None:
                # =========================================================
                # [추가] 이미지 저장 경로 생성 (original/날짜, result/날짜)
                # =========================================================
                now_dt = datetime.now()
                date_str = now_dt.strftime("%Y%m%d")
                time_str = now_dt.strftime("%H%M%S_%f")[:10] # 밀리초 앞부분까지만
                
                # 폴더 경로
                orig_dir = os.path.join(self.base_dir, "original", date_str)
                res_dir = os.path.join(self.base_dir, "result", date_str)
                
                os.makedirs(orig_dir, exist_ok=True)
                os.makedirs(res_dir, exist_ok=True)
                
                # 1. 원본 이미지 저장
                orig_filename = f"{time_str}_raw.jpg"
                cv2.imwrite(os.path.join(orig_dir, orig_filename), image)
                
                # 3) ~ 5) 검사 및 필터링
                self.inspector.set_target_product(top_prod_id)
                results_top, vis_img_top = self.inspector.process_image(image, res_dir, image_name=f"{time_str}_TOP")
                
                # 유효한 TOP 후보군
                valid_tops = [r for r in results_top if r.surface_type == "TOP" and r.text_type == "front"]
                
                self.inspector.set_target_product(bot_prod_id)
                results_bot, vis_img_bot = self.inspector.process_image(image, res_dir, image_name=f"{time_str}_BOT")
                
                # 유효한 BOT 후보군
                valid_bots = [r for r in results_bot if r.surface_type == "BOT" and r.text_type == "front"]
                
                # 6) ~ 7) 최적 대상 선정
                best_top = self._find_closest_to_center(valid_tops)
                best_bot = self._find_closest_to_center(valid_bots)
                
                # 시각화 및 저장 (기존과 동일)
                final_img = vis_img_bot.copy()
                if best_top or best_bot:
                    final_img = self.inspector.highlight_best_objects(final_img, best_top, best_bot)
                
                final_save_path = os.path.join(res_dir, f"result_{time_str}_FINAL.jpg")
                cv2.imwrite(final_save_path, final_img)
                
                # [수정] UI에 전달할 전체 후보군 데이터
                all_candidates = {
                    "tops": valid_tops,
                    "bots": valid_bots
                }

                if best_top and best_bot:
                    msg = (f"#OK,"
                           f"{best_top.center_x_mm},{best_top.center_y_mm},{best_top.final_angle},{best_top.direction_code},"
                           f"{best_bot.center_x_mm},{best_bot.center_y_mm},{best_bot.final_angle},{best_bot.direction_code};")
                    
                    logger.info(f"검출 성공: {msg}")
                    self.retry_cnt = 0
                    
                    # [수정] (메세지, 이미지, 전체후보군) 반환
                    return msg, final_img, all_candidates
                
                else:
                    self.retry_cnt += 1
                    logger.warning(f"{self.retry_cnt} 회 검출 실패")

            if self.retry_cnt <= max_retries:
                logger.info(f"재시도 {self.retry_cnt}: 피더 진동 요청")
                self.feeder.run_vibration(formula_number=1, duration_ms=500) 
                time.sleep(0.7)
                continue
            
        logger.error("최대 재시도 초과. 검출 실패.")
        self.retry_cnt = 0
        return "#EMPTY,0,0,0,0,0,0,0,0;" , final_img , all_candidates

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

# =========================================================
# Vision Server Wrapper (서버와 로직 연결)
# =========================================================
class IntegratedVisionServer(VisionServer):
    def __init__(self, system_logic):
        super().__init__()
        self.system = system_logic

    def perform_inspection(self, p1, p2):
        if GlobalState.autoYN != 1:
            logger.warning("자동 모드가 아닙니다 (#AUTOL 필요). 일단 수행합니다.")
        return self.system.process_inspection_scenario(p1, p2)
    
# =========================================================
# 메인 실행 부
# =========================================================
async def main():
    system = InspectionSystem()
    if not system.initialize():
        logger.error("시스템 초기화 실패. 종료합니다.")
        return
    server = IntegratedVisionServer(system)
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("프로그램 종료")
    finally:
        system.feeder.disconnect()
        system.camera.stop()

if __name__ == "__main__":
   if os.name == 'nt':
       asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
   asyncio.run(main())