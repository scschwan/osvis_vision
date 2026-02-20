import asyncio
import logging
import os

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
                        buffer = "" 
                        break

                    start_idx = buffer.find('#')
                    end_idx = buffer.find(';')

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
        trim_msg = raw_msg.strip().replace('#', '').replace(';', '')
        parts = trim_msg.split(',')

        if not parts:
            return ""

        command = parts[0].upper()

        if command == "READY":
            if GlobalState.readyYN == 1:
                return "#READY,,,,,,;" 
            else:
                return "#NG,,,,,,;"

        elif command == "MANUAL":
            GlobalState.autoYN = 0
            logger.info("Mode Changed: MANUAL")
            
            # [수정] 자식 클래스 훅 호출
            if self.handle_manual_command():
                return "#MANUAL,OK;"
            else:
                return "#MANUAL,FAIL;"

        elif command == "AUTO":
            GlobalState.autoYN = 1
            logger.info("Mode Changed: AUTO")
            return "#AUTO,OK;"

        elif command == "ALIGN":
            if len(parts) < 3:
                return "#NG,ARGS_ERROR;"
            
            prod1 = int(parts[1])
            prod2 = int(parts[2])

            # [수정] 실제 검사 로직 수행 (자식 클래스 Override)
            return self.perform_inspection(prod1, prod2)

        else:
            return "#UNKNOWN_CMD;"

    # --- Virtual Methods ---
    def perform_inspection(self, p1, p2):
        logger.warning("perform_inspection not implemented")
        return "#ERR,NOT_IMPL;"

    def handle_manual_command(self):
        # 기본 동작 True
        return True

if __name__ == "__main__":
    vision_server = VisionServer()
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(vision_server.start_server())
    except KeyboardInterrupt:
        logger.info("Server stopped manually.")