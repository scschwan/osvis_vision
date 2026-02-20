import asyncio
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("VisionServer")

# =========================================================
# 1. 전역 상태 관리 (Global Flags)
# =========================================================
class GlobalState:
    readyYN = 1  # 1: 검사 가능, 0: 검사 불가능
    autoYN = 0   # 1: 자동 모드, 0: 수동 모드

# =========================================================
# 2. 비전 서버 클래스
# =========================================================
class VisionServer:
    def __init__(self, host='0.0.0.0', port=8000):
        self.host = host
        self.port = port

    async def start_server(self):
        """TCP 서버 시작"""
        server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )

        addr = server.sockets[0].getsockname()
        logger.info(f"Vision TCP Server 런칭... ({addr})")

        async with server:
            await server.serve_forever()

    async def handle_client(self, reader, writer):
        """클라이언트 접속 처리 핸들러"""
        addr = writer.get_extra_info('peername')
        logger.info(f"Client connected: {addr}")

        buffer = ""

        try:
            while True:
                data = await reader.read(1024)
                if not data:
                    break  # 연결 끊김

                # 데이터 수신 및 디코딩
                buffer += data.decode('utf-8')

                # 패킷 처리 루프 (Sticky Packet 처리)
                # 메시지 포맷: #COMMAND;
                while ';' in buffer:
                    # 시작 문자(#) 확인
                    if '#' not in buffer:
                        # #이 없는데 ;만 있다면 쓰레기 데이터이므로 버림
                        buffer = "" 
                        break

                    start_idx = buffer.find('#')
                    end_idx = buffer.find(';')

                    # 종료 문자가 시작 문자보다 앞에 있으면 (깨진 패킷), 앞부분 버림
                    if end_idx < start_idx:
                        buffer = buffer[start_idx:]
                        continue

                    # 완전한 메시지 추출
                    full_msg = buffer[start_idx : end_idx+1] # #CMD;
                    
                    # 버퍼에서 처리된 부분 제거
                    buffer = buffer[end_idx+1:]

                    logger.info(f"[RECV] {full_msg}")

                    # 명령어 처리 및 응답 생성
                    response = self.process_command(full_msg)

                    # 응답 전송
                    if response:
                        writer.write(response.encode('utf-8'))
                        await writer.drain()
                        logger.info(f"[SEND] {response}")

        except Exception as e:
            logger.error(f"Client Error: {e}")
        finally:
            logger.info(f"Client disconnected: {addr}")
            writer.close()
            await writer.wait_closed()

    def process_command(self, raw_msg):
        """명령어 파싱 및 로직 수행"""
        # #과 ; 제거
        trim_msg = raw_msg.strip().replace('#', '').replace(';', '')
        parts = trim_msg.split(',')

        if not parts:
            return ""

        command = parts[0].upper()

        if command == "READY":
            # 1) 검사 가능 여부 체크
            if GlobalState.readyYN == 1:
                return "#READY,,,,,,;" # 콤마 개수 유지
            else:
                return "#NG,,,,,,;"

        elif command == "MANUAL":
            # 2) 수동 모드 전환
            GlobalState.autoYN = 0
            logger.info("Mode Changed: MANUAL")
            return "#MANUAL,OK;"

        elif command == "AUTO":
            # 3) 자동 모드 전환
            GlobalState.autoYN = 1
            logger.info("Mode Changed: AUTO")
            return "#AUTO,OK;"

        elif command == "ALIGN":
            # 4) 제품 검사 요청 (#ALIGN,제품1,제품2;)
            if len(parts) < 3:
                return "#NG,ARGS_ERROR;"
            
            prod1 = parts[1]
            prod2 = parts[2]

            # 실제 검사 로직 수행 (Mock)
            return self.perform_inspection(prod1, prod2)

        else:
            return "#UNKNOWN_CMD;"

    def perform_inspection(self, p1, p2):
        """
        [Mock] 실제 비전 검사 및 피더 로직이 들어갈 자리
        추후 main.py의 BushInspector와 연동 필요
        """
        logger.info(f"Inspect Request -> Prod1: {p1}, Prod2: {p2}")

        # --- [TODO: 여기에 실제 비전 로직 연동] ---
        # inspector = BushInspector(...)
        # results = inspector.process_image(...) 
        # ---------------------------------------

        # 테스트용 플래그 (True: 검출됨, False: 미검출)
        products_found = True 

        if products_found:
            # 검출 성공 시 데이터 포맷
            # x1, y1, r1, dir1, x2, y2, r2, dir2
            # 예시 데이터
            result_data = "106.423,53,99.33,1,16.43,53,9.33,3"
            return f"#OK,{result_data};"
        else:
            # 검출 실패 (EMPTY)
            # 0으로 채워진 데이터 반환 (x1~dir2 총 8개)
            return "#EMPTY,0,0,0,0,0,0;"

# =========================================================
# 서버 실행 (Main)
# =========================================================
if __name__ == "__main__":
    vision_server = VisionServer()
    
    # Windows의 경우 SelectorEventLoop 정책 설정 필요할 수 있음
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(vision_server.start_server())
    except KeyboardInterrupt:
        logger.info("Server stopped manually.")