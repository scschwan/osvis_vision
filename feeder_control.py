import time
import logging
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class FeederController:
    def __init__(self, ip='192.168.1.100', port=502):
        self.ip = ip
        self.port = port
        self.client = ModbusTcpClient(self.ip, port=self.port)
        self.is_connected = False
        self.unit_id = 1  # Station ID

    def connect(self):
        """피더 연결"""
        try:
            logger.info(f"연결 시도 중... ({self.ip}:{self.port})")
            self.is_connected = self.client.connect()
            if self.is_connected:
                logger.info("✓ Modbus TCP 연결 성공")
                return True
            else:
                logger.error("✗ 연결 실패")
                return False
        except Exception as e:
            logger.error(f"✗ 연결 중 오류 발생: {e}")
            return False

    def disconnect(self):
        """연결 해제"""
        if self.is_connected:
            # 안전하게 정지 신호 전송 후 연결 해제
            try:
                self.client.write_register(0, 0, slave=self.unit_id)
            except:
                pass
            self.client.close()
            self.is_connected = False
            logger.info("연결 해제됨")

    def _write_register(self, address, value, description=""):
        """레지스터 쓰기 헬퍼 함수"""
        if not self.is_connected:
            logger.error("연결되지 않았습니다.")
            return False

        try:
            # pymodbus의 write_register 사용
            result = self.client.write_register(address, value, slave=self.unit_id)
            if result.isError():
                logger.error(f"✗ {description} 실패 (Addr: {address}, Val: {value})")
                return False
            else:
                logger.info(f"  {description} 성공 (Addr: {address} -> {value})")
                time.sleep(0.05) # 기기 처리 대기 시간
                return True
        except ModbusException as e:
            logger.error(f"✗통신 오류: {e}")
            return False

    def setup_formula_mode(self):
        """기본 설정 (Formula Mode 초기화)"""
        logger.info("=== 설정 초기화 시작 ===")
        
        # P0.03 Control Mode = 1 (Formula)
        if not self._write_register(3, 1, "모드 설정(Formula)"): return False
        
        # P0.21 Signal Type = 0 (Switch)
        if not self._write_register(21, 0, "신호 타입 설정(Switch)"): return False
        
        # P0.99 Save = 1
        self._write_register(99, 1, "설정 저장")
        time.sleep(0.2)
        
        logger.info("✓ 설정 완료")
        return True

    # =========================================================
    # 요청 기능 1: 피더기 진동 (Formula 실행)
    # =========================================================
    def run_vibration(self, formula_number=1, duration_ms=3000):
        """
        피더 진동 실행 (Formula 1번 실행)
        :param formula_number: 실행할 Formula 번호 (기본 1)
        :param duration_ms: 진동 시간 (ms)
        """
        if not self.is_connected:
            logger.error("연결 필요")
            return False

        logger.info(f"=== 진동 시작 (Formula: {formula_number}, 시간: {duration_ms}ms) ===")

        # 1. Formula 번호 설정 (P0.02)
        if not self._write_register(2, formula_number, "Formula 번호 설정"): return False

        # 2. 시작 신호 (P0.00 = 1)
        if not self._write_register(0, 1, "진동 시작"): return False

        # 3. 지정된 시간만큼 대기
        time.sleep(duration_ms / 1000.0)

        # 4. 정지 신호 (P0.00 = 0)
        if not self._write_register(0, 0, "진동 정지"): return False
        
        logger.info("✓ 진동 완료")
        return True

    # =========================================================
    # 요청 기능 2: 백라이트 ON
    # =========================================================
    def light_on(self, brightness=100):
        """
        백라이트 켜기
        :param brightness: 밝기 0~100 (%)
        """
        logger.info(f"=== 백라이트 ON (밝기: {brightness}%) ===")
        
        # P0.10 Light Switch = 1 (ON)
        if not self._write_register(10, 1, "라이트 스위치 ON"): return False
        
        # P0.11 Brightness (0-1000 범위로 변환)
        val = int(brightness * 10)
        if not self._write_register(11, val, "밝기 설정"): return False
        
        return True

    # =========================================================
    # 요청 기능 3: 백라이트 OFF
    # =========================================================
    def light_off(self):
        """백라이트 끄기"""
        logger.info("=== 백라이트 OFF ===")
        
        # P0.10 Light Switch = 0 (OFF)
        return self._write_register(10, 0, "라이트 스위치 OFF")

# =========================================================
# 사용 예시 (Main)
# =========================================================
if __name__ == "__main__":
    # 피더 컨트롤러 생성
    feeder = FeederController(ip='192.168.1.100', port=502)

    if feeder.connect():
        try:
            # 1. 초기 1회 설정 (필요시)
            feeder.setup_formula_mode()

            while True:
                print("\n=== 메뉴 선택 ===")
                print("1. 피더 진동 (Formula 1)")
                print("2. 백라이트 ON")
                print("3. 백라이트 OFF")
                print("q. 종료")
                
                choice = input("선택: ").strip()

                if choice == '1':
                    # Formula 1번을 3초간 실행
                    feeder.run_vibration(formula_number=1, duration_ms=3000)
                
                elif choice == '2':
                    # 밝기 100%로 켜기
                    feeder.light_on(brightness=5)
                
                elif choice == '3':
                    # 끄기
                    feeder.light_off()
                
                elif choice.lower() == 'q':
                    break
                
        except KeyboardInterrupt:
            print("\n종료 중...")
        finally:
            feeder.disconnect()