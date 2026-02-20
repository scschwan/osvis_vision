from pypylon import pylon
import cv2
import numpy as np
import threading
import time

class CameraController:
    """
    Basler 카메라 제어 및 이미지 취득 모듈
    (딥러닝 기능 제거됨)
    """
    
    def __init__(self):
        self.camera = None
        self.converter = pylon.ImageFormatConverter()
        
        # Pylon 이미지 포맷 설정 (OpenCV 호환용 BGR8)
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        
        # 기본 ROI 설정 [x, y, w, h] (필요에 따라 수정)
        self.roi = [0, 0, 1920, 1080] 
        self.use_roi = False  # ROI 사용 여부 플래그

        self.lock = threading.Lock()
        self.current_frame = None
        self.running = False
        self.thread = None

    def connect_camera(self):
        """카메라 연결"""
        try:
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            
            if len(devices) == 0:
                print("✗ 연결된 Basler 카메라가 없습니다")
                return False
            
            # 첫 번째 카메라 연결
            self.camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
            self.camera.Open()
            
            model_name = self.camera.GetDeviceInfo().GetModelName()
            print(f"✓ 카메라 연결 성공: {model_name}")
            
            # 카메라 해상도에 맞춰 기본 ROI 설정
            width = self.camera.Width.GetValue()
            height = self.camera.Height.GetValue()
            self.roi = [0, 0, width, height]
            
            return True
            
        except Exception as e:
            print(f"✗ 카메라 연결 실패: {e}")
            return False

    

    def start_capture(self):
        """이미지 캡처 스레드 시작"""
        if not self.camera or not self.camera.IsOpen():
            print("✗ 카메라가 연결되지 않았습니다.")
            return False
        
        if self.running:
            print("⚠ 이미 캡처 중입니다.")
            return True

        self.running = True
        
        # 연속 그랩 모드 시작 (최신 이미지만 유지)
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        
        # 백그라운드 스레드 시작
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("✓ 실시간 캡처 시작")
        return True
    
    def _capture_loop(self):
        """내부 캡처 루프 (Thread용)"""
        while self.running and self.camera.IsGrabbing():
            try:
                # 5000ms 타임아웃으로 이미지 대기
                grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                
                if grab_result.GrabSucceeded():
                    # Pylon 이미지를 OpenCV 포맷(numpy array)으로 변환
                    image = self.converter.Convert(grab_result)
                    img = image.GetArray()
                    
                    # ROI 적용
                    if self.use_roi:
                        with self.lock:
                            x, y, w, h = self.roi
                            # 배열 범위 초과 방지 로직 필요 시 추가
                            img = img[y:y+h, x:x+w].copy()
                    
                    # 최신 프레임 업데이트
                    with self.lock:
                        self.current_frame = img
                else:
                    print(f"⚠ 그랩 실패: {grab_result.ErrorCode} {grab_result.ErrorDescription}")
                
                grab_result.Release()
                
            except Exception as e:
                print(f"✗ 캡처 루프 오류: {e}")
                # 오류 발생 시 루프 종료 여부 결정 (여기선 계속 시도)
                # break 
        
        print("ℹ 캡처 루프 종료")

    def get_latest_image(self):
        """
        가장 최신 이미지를 numpy array(OpenCV 포맷)로 반환
        :return: cv2 image (numpy array) or None
        """
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None

    def get_jpeg_bytes(self):
        """
        웹 스트리밍 등을 위한 JPEG 바이트 데이터 반환
        :return: bytes or None
        """
        frame = self.get_latest_image()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                return buffer.tobytes()
        return None
    
    def stop(self):
        """카메라 정지 및 연결 해제"""
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        if self.camera:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
            if self.camera.IsOpen():
                self.camera.Close()
            self.camera = None
            
        print("✓ 카메라 종료 완료")

# =========================================================
# 테스트용 메인 코드
# =========================================================
if __name__ == "__main__":
    cam = CameraController()
    
    if cam.connect_camera():
        cam.start_capture()
        
        # 10초 동안 이미지 취득 테스트
        start_time = time.time()
        frame_count = 0
        
        print("🎥 10초간 이미지 취득 테스트 중...")
        
        while time.time() - start_time < 10:
            frame = cam.get_latest_image()
            if frame is not None:
                frame_count += 1
                # 여기에서 cv2.imshow 등으로 확인 가능
                # cv2.imshow("Test", frame)
                # cv2.waitKey(1)
            time.sleep(0.01)
            
        print(f"✓ 테스트 완료: 총 {frame_count} 프레임 취득")
        cam.stop()