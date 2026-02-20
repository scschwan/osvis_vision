import sys
import os
import json
import logging
import asyncio
import threading
import cv2
import math
import numpy as np
from datetime import datetime

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QTabWidget, QLabel, QPushButton, QTableWidget, QTableWidgetItem,QCheckBox,
                               QGroupBox, QTextEdit, QLineEdit, QComboBox, QHeaderView, QSplitter,QDoubleSpinBox,
                               QFormLayout, QMessageBox, QFrame, QGridLayout, QFileDialog, QSpinBox,QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem) 
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThread, Slot
from PySide6.QtGui import QImage, QPixmap, QColor, QFont, QPainter

# 기존 모듈 임포트
from main_app import InspectionSystem, IntegratedVisionServer
from vision_server import GlobalState

# =========================================================
# [신규] 줌/팬 기능이 있는 이미지 뷰어 위젯
# =========================================================
class ImageGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        # 기본 설정
        self.setDragMode(QGraphicsView.ScrollHandDrag) # 드래그로 이동 가능
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(30, 30, 30)) # 배경색 (어두운 회색)
        self.setFrameShape(QFrame.NoFrame)
        
        self.zoom_factor = 1.15 # 줌 속도

    def set_image(self, cv_image):
        """OpenCV 이미지를 받아서 표출"""
        if cv_image is None: return

        # BGR -> RGB 변환 및 QPixmap 생성
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # 씬 업데이트
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(0, 0, w, h)
        
        # 이미지를 뷰에 맞게 축소/확대 (최초 로드 시)
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def set_pixmap(self, pixmap):
        """QPixmap을 직접 받아서 표출 (파일 로드 시 사용)"""
        if pixmap is None: return
        self.pixmap_item.setPixmap(pixmap)
        w = pixmap.width()
        h = pixmap.height()
        self.scene.setSceneRect(0, 0, w, h)
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        """마우스 휠로 줌 인/아웃"""
        if event.angleDelta().y() > 0:
            # Zoom In
            self.scale(self.zoom_factor, self.zoom_factor)
        else:
            # Zoom Out
            self.scale(1 / self.zoom_factor, 1 / self.zoom_factor)

# =========================================================
# 1. 로그 핸들러 (Python logging -> Qt Signal)
# =========================================================
class QtLogHandler(logging.Handler, QObject):
    log_signal = Signal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)

# =========================================================
# 2. 백그라운드 워커 (TCP 서버 및 로직 구동)
# =========================================================
class SystemWorker(QThread):
    result_signal = Signal(object, str, object) 
    stats_signal = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.system = InspectionSystem()
        self.loop = None
        self.server = None
        self.running = True
        self.total_inspect_count = 0
        self.total_sent_count = 0

    def run(self):
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        if not self.system.initialize():
            return

        self.server = IntegratedVisionServer(self.system)
        
        original_process = self.system.process_inspection_scenario
        
        def hooked_process(top_id, bot_id):
            self.total_inspect_count += 1
            # process_inspection_scenario가 (msg, image)를 리턴하도록 수정됨
            # [수정] main_app에서 3개를 리턴받음
            result_msg, result_img, all_candidates = original_process(top_id, bot_id)
            
            if "#OK" in result_msg:
                self.total_sent_count += 1
            self.stats_signal.emit(self.total_inspect_count, self.total_sent_count)
            
            if result_img is None:
                current_img = self.system.camera.get_latest_image()
            else:
                current_img = result_img
            
            # [수정] 전체 후보군 데이터를 3번째 인자로 전달
            self.result_signal.emit(current_img, result_msg, all_candidates)
            
            return result_msg
           
        self.system.process_inspection_scenario = hooked_process

        try:
            self.loop.run_until_complete(self.server.start_server())
        except Exception as e:
            logging.error(f"Server Error: {e}")
        finally:
            self.system.feeder.disconnect()
            self.system.camera.stop()

    def stop(self):
        self.running = False

# =========================================================
# 3. 메인 윈도우 UI
# =========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Inspection System (Feeder Control)")
        self.resize(1280, 1024)

        # =========================================================
        # [수정] 로그 설정 (UI 출력 + 파일 저장)
        # =========================================================
        
        # 1. 경로 설정 (exe/script 구분)
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
        # 2. log 폴더 생성
        log_dir = os.path.join(base_dir, "log")
        os.makedirs(log_dir, exist_ok=True)
        
        # 3. 날짜별 로그 파일명 생성 (예: 2025-12-27.log)
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_path = os.path.join(log_dir, f"{date_str}.log")
        
        # 4. 로거 포맷 설정
        formatter = logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S')
        
        # 5. [기존] UI 로그 핸들러
        self.log_handler = QtLogHandler()
        self.log_handler.setFormatter(formatter)
        self.log_handler.log_signal.connect(self.update_log)
        
        # 6. [신규] 파일 로그 핸들러 추가
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # 7. 루트 로거에 핸들러 등록
        # (기존 핸들러가 쌓이는 것을 방지하기 위해 먼저 reset 할 수도 있지만, 
        #  여기서는 간단히 addHandler만 수행)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # 중복 방지: 이미 핸들러가 있는지 체크하거나 그냥 추가
        # (PyQt 특성상 리로드될 때 중복될 수 있으므로 깔끔하게 처리)
        if not any(isinstance(h, QtLogHandler) for h in root_logger.handlers):
            root_logger.addHandler(self.log_handler)
            
        # 파일 핸들러는 매 실행마다 추가하면 중복될 수 있으므로 체크 (선택사항이나 권장)
        # 간단하게는 그냥 추가해도 되지만, 깔끔하게 아래처럼 처리:
        has_file_handler = False
        for h in root_logger.handlers:
            if isinstance(h, logging.FileHandler):
                # 같은 파일이면 패스 (혹은 로직에 따라 다름)
                if h.baseFilename == os.path.abspath(log_path):
                    has_file_handler = True
                    break
        
        if not has_file_handler:
            root_logger.addHandler(file_handler)

        # =========================================================

        # 백그라운드 워커 시작
        self.worker = SystemWorker()
        self.worker.result_signal.connect(self.update_inspection_result)
        self.worker.stats_signal.connect(self.update_stats)
        self.worker.start()

        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_global_status)
        self.timer.start(1000)

        # 수동 테스트용 변수
        self.manual_test_image_path = ""

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 헤더
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #DCDCDC; border-bottom: 2px solid #AAAAAA;") 
        header_layout = QHBoxLayout(header_frame)
        
        title_label = QLabel("Vision Inspection System")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setStyleSheet("color: #333333; border: none;") 
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        logo_label = QLabel()
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(base_dir, "logo.png")
        
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path).scaled(300, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
            logo_label.setStyleSheet("border: none;") 
        else:
            logo_label.setText("[LOGO]")
            logo_label.setStyleSheet("font-weight: bold; color: gray; border: 1px solid gray; padding: 5px;")
            
        header_layout.addWidget(logo_label)
        layout.addWidget(header_frame)

        # 탭
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.create_main_tab()
        self.create_comm_tab()
        self.create_config_tab()
        self.create_log_tab() # 여기가 수정됨

    def create_main_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        '''
        self.image_label = QLabel("Waiting for Image...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #222; color: #EEE; border: 2px solid #555;")
        self.image_label.setMinimumSize(640, 480)
        layout.addWidget(self.image_label, stretch=2)

        right_panel = QVBoxLayout()
        '''

        self.image_viewer = ImageGraphicsView()
        layout.addWidget(self.image_viewer, stretch=2)

        right_panel = QVBoxLayout()
        
        status_group = QGroupBox("System Status & Globals")
        status_layout = QGridLayout()
        self.lbl_ready = QLabel("ReadyYN: 0")
        self.lbl_top = QLabel("TopProd: -")
        self.lbl_bot = QLabel("BotProd: -")
        self.lbl_retry = QLabel("RetryCnt: 0")
        for lbl in [self.lbl_ready, self.lbl_top, self.lbl_bot, self.lbl_retry]:
            lbl.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px; background: #E0E0E0; border-radius: 4px;")

        status_layout.addWidget(QLabel("Global Ready:"), 0, 0)
        status_layout.addWidget(self.lbl_ready, 0, 1)
        status_layout.addWidget(QLabel("Top Product:"), 1, 0)
        status_layout.addWidget(self.lbl_top, 1, 1)
        status_layout.addWidget(QLabel("Bot Product:"), 2, 0)
        status_layout.addWidget(self.lbl_bot, 2, 1)
        status_layout.addWidget(QLabel("Retry Count:"), 3, 0)
        status_layout.addWidget(self.lbl_retry, 3, 1)
        status_group.setLayout(status_layout)
        right_panel.addWidget(status_group)

        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout()
        self.lbl_total_cnt = QLabel("0")
        self.lbl_sent_cnt = QLabel("0")
        self.lbl_result_status = QLabel("WAITING") 
        self.lbl_result_status.setAlignment(Qt.AlignCenter)
        self.lbl_result_status.setStyleSheet("font-size: 20px; font-weight: bold; color: gray; border: 2px solid gray; padding: 10px;")

        stats_layout.addRow("Total Inspections:", self.lbl_total_cnt)
        stats_layout.addRow("Items Sent (OK):", self.lbl_sent_cnt)
        stats_group.setLayout(stats_layout)
        right_panel.addWidget(stats_group)
        right_panel.addWidget(self.lbl_result_status)

        table_group = QGroupBox("Inspection Results")
        table_layout = QVBoxLayout()
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(5)
       # [수정] 헤더 라벨 변경
        self.result_table.setHorizontalHeaderLabels(["ID", "X(mm)", "Y(mm)", "Angle", "Dir"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(self.result_table)
        table_group.setLayout(table_layout)
        right_panel.addWidget(table_group, stretch=1)
        
        layout.addLayout(right_panel, stretch=1)
        self.tabs.addTab(tab, "Main Monitor")

    def create_comm_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        feeder_group = QGroupBox("Feeder Manual Control")
        feeder_layout = QHBoxLayout()
        btn_vib = QPushButton("진동 시작 (Formula 1)")
        btn_vib.clicked.connect(lambda: self.run_feeder_cmd(1))
        btn_on = QPushButton("백라이트 ON (Formula 2)")
        btn_on.setStyleSheet("background-color: #AAFFAA;")
        btn_on.clicked.connect(lambda: self.run_feeder_cmd(2))
        btn_off = QPushButton("백라이트 OFF (Formula 3)")
        btn_off.setStyleSheet("background-color: #FFB6C1;")
        btn_off.clicked.connect(lambda: self.run_feeder_cmd(3))
        feeder_layout.addWidget(btn_vib)
        feeder_layout.addWidget(btn_on)
        feeder_layout.addWidget(btn_off)
        feeder_group.setLayout(feeder_layout)
        layout.addWidget(feeder_group)

        socket_group = QGroupBox("Socket Settings (Display Only)")
        socket_layout = QHBoxLayout()
        socket_layout.addWidget(QLabel("Port:"))
        socket_layout.addWidget(QLineEdit("8000"))
        socket_layout.addWidget(QLabel("Split Char:"))
        socket_layout.addWidget(QLineEdit(","))
        socket_layout.addWidget(QLabel("Start Char:"))
        socket_layout.addWidget(QLineEdit("#"))
        socket_layout.addWidget(QLabel("End Char:"))
        socket_layout.addWidget(QLineEdit(";"))
        socket_group.setLayout(socket_layout)
        layout.addWidget(socket_group)

        log_split = QSplitter(Qt.Vertical)
        feeder_log_group = QGroupBox("Feeder/System Log")
        fl_layout = QVBoxLayout()
        self.txt_feeder_log = QTextEdit()
        self.txt_feeder_log.setReadOnly(True)
        fl_layout.addWidget(self.txt_feeder_log)
        feeder_log_group.setLayout(fl_layout)
        
        socket_log_group = QGroupBox("Socket Communication Log")
        sl_layout = QVBoxLayout()
        self.txt_socket_log = QTextEdit()
        self.txt_socket_log.setReadOnly(True)
        sl_layout.addWidget(self.txt_socket_log)
        socket_log_group.setLayout(sl_layout)

        log_split.addWidget(feeder_log_group)
        log_split.addWidget(socket_log_group)
        layout.addWidget(log_split)
        self.tabs.addTab(tab, "Communication")

   # ---------------------------------------------------------
    # Tab 3: Config (수정됨)
    # ---------------------------------------------------------
    def create_config_tab(self):
        tab = QWidget()
        main_layout = QHBoxLayout(tab)

        # [좌측] Product Configuration (기존 유지)
        prod_group = QGroupBox("Product Parameters (1~25)")
        prod_layout = QVBoxLayout()
        
        sel_layout = QHBoxLayout()
        sel_layout.addWidget(QLabel("Select Product ID:"))
        self.combo_prod = QComboBox()
        for i in range(1, 26):
            self.combo_prod.addItem(f"Product {i}")
        self.combo_prod.currentIndexChanged.connect(self.load_product_config)
        sel_layout.addWidget(self.combo_prod)
        sel_layout.addStretch()
        
        save_btn = QPushButton("Save Product Config")
        save_btn.setStyleSheet("font-weight: bold; background-color: #DDD;")
        save_btn.clicked.connect(self.save_product_config)
        sel_layout.addWidget(save_btn)
        
        prod_layout.addLayout(sel_layout)

        self.config_table = QTableWidget()
        self.config_table.setColumnCount(2)
        self.config_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.config_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        prod_layout.addWidget(self.config_table)
        
        prod_group.setLayout(prod_layout)
        main_layout.addWidget(prod_group, stretch=2)

        # ==========================================
        # [우측] Global System Configuration (확장됨)
        # ==========================================
        global_group = QGroupBox("Global System & Algorithm Settings (Saved in global_config.json)")
        global_layout = QVBoxLayout()
        form_layout = QFormLayout()
        
        # 1) pixel_to_mm
        self.spin_pixel = QDoubleSpinBox()
        self.spin_pixel.setDecimals(6)
        self.spin_pixel.setRange(0.000001, 1.0)
        self.spin_pixel.setSingleStep(0.000001)
        form_layout.addRow("Pixel to MM:", self.spin_pixel)
        
        # 2) ReadyYN (수정 불가 - 모니터링용)
        self.spin_ready = QLineEdit() # QSpinBox 대신 QLineEdit 사용해도 됨
        self.spin_ready.setReadOnly(True) # 수정 불가
        self.spin_ready.setStyleSheet("background-color: #EEE; color: #555;")
        self.spin_ready.setToolTip("Current System Status (Read Only)")
        form_layout.addRow("Global ReadyYN:", self.spin_ready)
        
        # 3) Max Retry Count (설정값)
        self.spin_retry = QSpinBox()
        self.spin_retry.setRange(0, 100)
        self.spin_retry.setToolTip("Maximum number of retries upon failure")
        form_layout.addRow("Max Retry Count:", self.spin_retry)

        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        form_layout.addRow(line)
        
        # 4) Bush Area (Min/Max)
        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(0, 999999)
        self.spin_min_area.setSingleStep(1000)
        form_layout.addRow("Bush Min Area:", self.spin_min_area)
        
        self.spin_max_area = QSpinBox()
        self.spin_max_area.setRange(0, 999999)
        self.spin_max_area.setSingleStep(1000)
        form_layout.addRow("Bush Max Area:", self.spin_max_area)

        # [신규] 겹침 방지 최소 거리 (mm)
        self.spin_min_dist_mm = QDoubleSpinBox()
        self.spin_min_dist_mm.setRange(0.0, 100.0)
        self.spin_min_dist_mm.setSingleStep(0.1)
        self.spin_min_dist_mm.setToolTip("If distance < this value, both objects are removed.")
        form_layout.addRow("Overlap Min Dist (mm):", self.spin_min_dist_mm)
        
        # 5) Threshold Params
        self.spin_thresh_block = QSpinBox()
        self.spin_thresh_block.setRange(3, 99)
        self.spin_thresh_block.setSingleStep(2)
        form_layout.addRow("Thresh Block Size:", self.spin_thresh_block)
        
        self.spin_thresh_c = QDoubleSpinBox()
        self.spin_thresh_c.setRange(-100, 100)
        form_layout.addRow("Thresh C Value:", self.spin_thresh_c)
        
        global_layout.addLayout(form_layout)
        
        # 버튼 영역
        btn_layout = QHBoxLayout()
        btn_refresh = QPushButton("🔄 Load Current")
        btn_refresh.clicked.connect(self.load_global_config)
        
        btn_apply = QPushButton("💾 Save & Apply")
        btn_apply.setStyleSheet("background-color: #CCFFCC; font-weight: bold;")
        btn_apply.clicked.connect(self.apply_global_config)
        
        btn_layout.addWidget(btn_refresh)
        btn_layout.addWidget(btn_apply)
        
        global_layout.addStretch()
        global_layout.addLayout(btn_layout)
        
        global_group.setLayout(global_layout)
        main_layout.addWidget(global_group, stretch=1)

        self.tabs.addTab(tab, "Config")
        
        QTimer.singleShot(1000, self.load_initial_config)

    # ---------------------------------------------------------
    # Tab 4: Logs & Manual Test (수정된 부분)
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Tab 4: Logs & Manual Test (레이아웃 변경: 이미지 우측 크게)
    # ---------------------------------------------------------
    def create_log_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        left_panel = QWidget()
        left_panel.setFixedWidth(420)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 1. Manual Test Controls
        ctrl_group = QGroupBox("1. Manual Test Controls")
        ctrl_layout = QVBoxLayout()

        # 파일 업로드
        btn_upload = QPushButton("📂 이미지 불러오기 (Local)")
        btn_upload.setStyleSheet("padding: 8px; font-weight: bold;")
        btn_upload.clicked.connect(self.load_local_image)
        ctrl_layout.addWidget(btn_upload)

        self.lbl_file_path = QLabel("선택된 파일 없음")
        self.lbl_file_path.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_file_path.setWordWrap(True)
        ctrl_layout.addWidget(self.lbl_file_path)

        # 제품 ID
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Target Product ID:"))
        self.spin_test_id = QSpinBox()
        self.spin_test_id.setRange(1, 25)
        self.spin_test_id.setValue(1)
        self.spin_test_id.setMinimumHeight(30)
        input_layout.addWidget(self.spin_test_id)
        ctrl_layout.addLayout(input_layout)

        # 실행 버튼
        btn_start_test = QPushButton("▶ 수동 검사 수행")
        btn_start_test.setStyleSheet("background-color: #CCCCFF; padding: 10px; font-weight: bold; font-size: 14px;")
        btn_start_test.clicked.connect(self.run_manual_test)
        ctrl_layout.addWidget(btn_start_test)
        
        # [신규] Binary 이미지 보기 체크박스
        
        self.chk_show_binary = QCheckBox("Show Binary Image (Debug)")
        self.chk_show_binary.setStyleSheet("font-weight: bold; color: blue;")
        # 체크 상태 변경 시 이미지 다시 로드
        self.chk_show_binary.stateChanged.connect(self.toggle_test_image_view)
        ctrl_layout.addWidget(self.chk_show_binary)
        
        ctrl_group.setLayout(ctrl_layout)
        left_layout.addWidget(ctrl_group)

        # 2. Result Details (Text)
        res_group = QGroupBox("2. Inspection Details")
        res_layout = QVBoxLayout()
        self.txt_test_result = QTextEdit()
        self.txt_test_result.setReadOnly(True)
        self.txt_test_result.setPlaceholderText("검사 결과 데이터가 여기에 표시됩니다.")
        res_layout.addWidget(self.txt_test_result)
        res_group.setLayout(res_layout)
        left_layout.addWidget(res_group, stretch=1) # 텍스트 영역 비율 1

        # 3. System Logs (Small)
        log_group = QGroupBox("3. System Logs")
        log_layout = QVBoxLayout()
        self.txt_full_log = QTextEdit()
        self.txt_full_log.setReadOnly(True)
        self.txt_full_log.setStyleSheet("font-family: Consolas; font-size: 11px;")
        log_layout.addWidget(self.txt_full_log)
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group, stretch=1) # 로그 영역 비율 1

        # 좌측 패널 메인 레이아웃에 추가
        layout.addWidget(left_panel)

        # [우측 패널] 결과 이미지 (크게 표시)
        '''
        image_group = QGroupBox("Inspection Result Image")
        image_layout = QVBoxLayout()
        
        self.lbl_test_image = QLabel("No Image")
        self.lbl_test_image.setAlignment(Qt.AlignCenter)
        self.lbl_test_image.setStyleSheet("background-color: #202020; color: #777; border: 2px solid #444;")
        self.lbl_test_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # 확장 정책
        
        image_layout.addWidget(self.lbl_test_image)
       
        '''

        # [우측 패널] 결과 이미지 (크게 표시)
        image_group = QGroupBox("Inspection Result Image (Wheel to Zoom, Drag to Pan)") # 텍스트 변경
        image_layout = QVBoxLayout()
        
        # [수정] QLabel 대신 ImageGraphicsView 사용
        self.test_image_viewer = ImageGraphicsView()
        
        image_layout.addWidget(self.test_image_viewer)
        image_group.setLayout(image_layout)
        
        layout.addWidget(image_group, stretch=1)

        self.tabs.addTab(tab, "Logs & Test")

    # =========================================================
    # 수동 테스트 기능 구현
    # =========================================================
    def load_local_image(self):
        """로컬 이미지 파일 선택 다이얼로그"""
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        if file_dialog.exec():
            files = file_dialog.selectedFiles()
            if files:
                self.manual_test_image_path = files[0]
                self.lbl_file_path.setText(self.manual_test_image_path)
                
                # 미리보기 표시
                '''
                pixmap = QPixmap(self.manual_test_image_path).scaled(
                    self.lbl_test_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.lbl_test_image.setPixmap(pixmap)
                '''
                pixmap = QPixmap(self.manual_test_image_path)
                self.test_image_viewer.set_pixmap(pixmap) # ImageGraphicsView 사용
                self.txt_test_result.setText("이미지 로드됨. 검사 대기 중...")

    def toggle_test_image_view(self):
        # 현재 저장된 경로가 있으면 다시 로드
        if hasattr(self, 'current_test_result_dir') and hasattr(self, 'current_test_filename'):
            self.load_manual_result_images(self.current_test_result_dir, self.current_test_filename)

    def run_manual_test(self):
        """선택된 이미지와 ID로 수동 검사 수행 (저장 및 결과 표시 포함)"""
        # 0. 이미지 선택 여부 확인
        if not self.manual_test_image_path or not os.path.exists(self.manual_test_image_path):
            QMessageBox.warning(self, "Error", "이미지 파일을 먼저 선택해주세요.")
            return

        try:
            # 1. 이미지 읽기 (한글 경로 지원을 위해 numpy 사용)
            stream = open(self.manual_test_image_path.encode("utf-8"), "rb")
            bytes = bytearray(stream.read())
            numpy_array = np.asarray(bytes, dtype=np.uint8)
            image = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
            stream.close()
            
            if image is None:
                QMessageBox.warning(self, "Error", "이미지 로드 실패.")
                return

            # =========================================================
            # [저장] 날짜별 폴더 생성 및 원본 저장
            # =========================================================
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))

            now_dt = datetime.now()
            date_str = now_dt.strftime("%Y%m%d")
            time_str = now_dt.strftime("%H%M%S")

            # 폴더 경로 설정
            orig_dir = os.path.join(base_dir, "original", date_str)
            res_dir = os.path.join(base_dir, "result", date_str)
            
            os.makedirs(orig_dir, exist_ok=True)
            os.makedirs(res_dir, exist_ok=True)

            # 원본 이미지 저장
            orig_filename = f"{time_str}_manual_raw.jpg"
            cv2.imwrite(os.path.join(orig_dir, orig_filename), image)
            # =========================================================

            # 2. Inspector 설정
            inspector = self.worker.system.inspector
            target_id = self.spin_test_id.value()
            inspector.set_target_product(target_id)
            
            # 3. 검사 수행 
            # (변경된 bush_inspector에 따라 results와 vis_img를 반환받음)
            res_filename = f"{time_str}_manual_result" 
            results, vis_img = inspector.process_image(image, res_dir, image_name=res_filename)
            
            # [수정] 4. 결과 이미지 즉시 표시 (파일 로드 X -> 메모리 이미지 O)
            if vis_img is not None:
                self.test_image_viewer.set_image(vis_img)
                
                # [안전장치] UI에서 명시적으로 결과 파일 저장 (파일 누락 방지)
                # bush_inspector 내부 저장과 중복되더라도 덮어쓰므로 안전함
                save_path = os.path.join(res_dir, f"result_{res_filename}.jpg")
                cv2.imwrite(save_path, vis_img)

            # [상태 저장] 체크박스 토글 시 이미지를 다시 로드하기 위해 경로 저장
            self.current_test_result_dir = res_dir
            self.current_test_filename = res_filename
            
            # 4. 결과 텍스트 출력 (mm 단위 좌표 표기)
            result_text = f"검사 완료 (Product ID: {target_id})\n"
            result_text += f"검출된 객체 수: {len(results)}\n"
            
            for res in results:
                # 결과 리스트 텍스트 생성
                result_text += f"ID:{res.bush_id} Type:{res.text_type} Surf:{res.surface_type}\n"
                result_text += f"Pos(mm):({res.center_x_mm:.2f}, {res.center_y_mm:.2f}) Ang:{res.rotation_angle}\n"
            
            self.txt_test_result.setText(result_text)
            
            # 5. 결과 이미지 표시
            # (체크박스 상태에 따라 Binary 혹은 Result 이미지를 로드하는 함수 호출)
            self.load_manual_result_images(res_dir, res_filename)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"검사 중 오류 발생:\n{e}")
            logging.error(f"Manual Test Error: {e}")

    # [신규] 이미지 로드 로직 분리
    def load_manual_result_images(self, res_dir, res_filename):
        # 체크박스 상태 확인
        if self.chk_show_binary.isChecked():
            # Binary 이미지 로드
            img_path = os.path.join(res_dir, f"binary_{res_filename}.jpg")
        else:
            # 결과 이미지 로드
            img_path = os.path.join(res_dir, f"result_{res_filename}.jpg")
            
        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            self.test_image_viewer.set_pixmap(pixmap)
        else:
            # 파일이 없을 경우 (검출 실패 등)
            self.txt_test_result.append(f"\n[Warning] 이미지를 찾을 수 없습니다: {os.path.basename(img_path)}")

    def show_manual_result_image(self, path):
        if os.path.exists(path):
            '''
            pixmap = QPixmap(path).scaled(
                self.lbl_test_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.lbl_test_image.setPixmap(pixmap)
            '''
            pixmap = QPixmap(path)
            self.test_image_viewer.set_pixmap(pixmap)
        else:
            self.txt_test_result.append("\n[Warning] 결과 이미지를 찾을 수 없습니다.")

    # =========================================================
    # (기존 메서드들 그대로 유지)
    # =========================================================
    @Slot(str)
    def update_log(self, msg):
        self.txt_full_log.append(msg)
        if "RECV" in msg or "SEND" in msg or "Client" in msg:
            self.txt_socket_log.append(msg)
        else:
            self.txt_feeder_log.append(msg)

    @Slot(object, object, str)
    def update_inspection_result(self, image, msg, all_candidates):
        # 1. 이미지 업데이트 (동일)
        if image is not None:
            self.image_viewer.set_image(image)

        # 2. 결과 상태 텍스트 (동일)
        if "#OK" in msg:
            self.lbl_result_status.setText("SUCCESS")
            self.lbl_result_status.setStyleSheet("font-size: 24px; font-weight: bold; color: green; border: 3px solid green; background: #E0FFE0;")
        elif "#EMPTY" in msg:
            self.lbl_result_status.setText("FAIL (EMPTY)")
            self.lbl_result_status.setStyleSheet("font-size: 24px; font-weight: bold; color: red; border: 3px solid red; background: #FFE0E0;")
            
        # 3. 테이블 업데이트 (대폭 수정)
        # msg와 all_candidates를 모두 넘겨서 처리
        self.update_result_table(msg, all_candidates)

    def update_result_table(self, msg, all_candidates):
        """
        [정렬 순서 변경]
        1. 선정된 TOP/BOT (노란색)
        2. 나머지 TOP (흰색)
        3. 나머지 BOT (흰색)
        """
        # 1. 선정된 제품의 좌표 파싱 (Highlight용)
        selected_coords = []
        if "#OK" in msg:
            try:
                # #OK,x1,y1,a1,d1,x2,y2,a2,d2;
                data = msg.replace("#OK,", "").replace(";", "").split(",")
                if len(data) >= 8:
                    # (x1, y1), (x2, y2)
                    selected_coords.append((float(data[0]), float(data[1]))) # Top Selected
                    selected_coords.append((float(data[4]), float(data[5]))) # Bot Selected
            except:
                pass

        # 2. 데이터 분류 (Selected vs Others)
        if not all_candidates:
            self.result_table.setRowCount(0)
            return

        tops = all_candidates.get("tops", [])
        bots = all_candidates.get("bots", [])
        
        selected_list = []  # 최상단에 보여줄 목록 (노란색)
        other_tops = []     # 그 다음 보여줄 TOP 목록 (흰색)
        other_bots = []     # 마지막에 보여줄 BOT 목록 (흰색)

        # 헬퍼 함수: 이 객체가 선정된 좌표인지 확인
        def is_selected(res):
            x = res.center_x_mm
            y = res.center_y_mm
            for (sx, sy) in selected_coords:
                if abs(x - sx) < 0.1 and abs(y - sy) < 0.1: # 오차범위 0.1mm
                    return True
            return False

        # TOP 분류
        for res in tops:
            if is_selected(res):
                selected_list.append(("TOP", res))
            else:
                other_tops.append(("TOP", res))
        
        # BOT 분류
        for res in bots:
            if is_selected(res):
                selected_list.append(("BOT", res))
            else:
                other_bots.append(("BOT", res))

        # 3. 리스트 합치기 (순서: Selected -> Other Tops -> Other Bots)
        final_list = selected_list + other_tops + other_bots
        
        # 4. 테이블 렌더링
        self.result_table.setRowCount(len(final_list))
        dir_map = {1: "E", 2: "S", 3: "W", 4: "N", 0: "Err"}
        
        for row_idx, (type_str, res) in enumerate(final_list):
            self._add_row_to_table(row_idx, type_str, res, selected_coords, dir_map)

    def _add_row_to_table(self, row, type_str, res, selected_coords, dir_map):
        """테이블에 행을 추가하고 색상을 결정하는 내부 함수"""
        x_val = res.center_x_mm
        y_val = res.center_y_mm
        angle_val = res.final_angle
        dir_str = dir_map.get(res.direction_code, str(res.direction_code))
        
        # 아이템 생성
        items = [
            #QTableWidgetItem(f"{type_str} ({res.bush_id})"),
            QTableWidgetItem(f"{type_str}"),
            QTableWidgetItem(f"{x_val:.2f}"),
            QTableWidgetItem(f"{y_val:.2f}"),
            QTableWidgetItem(f"{angle_val:.2f}"),
            QTableWidgetItem(dir_str)
        ]
        
        # 색상 결정
        is_selected = False
        for (sel_x, sel_y) in selected_coords:
            if abs(x_val - sel_x) < 0.1 and abs(y_val - sel_y) < 0.1:
                is_selected = True
                break
        
        # 선정된 녀석은 노란색, 나머지는 흰색
        bg_color = QColor("yellow") if is_selected else QColor("white")
        
        for col, item in enumerate(items):
            item.setBackground(bg_color)
            item.setTextAlignment(Qt.AlignCenter)
            # 폰트 볼드 처리 (선정된 것만)
            if is_selected:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                
            self.result_table.setItem(row, col, item)

    @Slot(int, int)
    def update_stats(self, total, sent):
        self.lbl_total_cnt.setText(str(total))
        self.lbl_sent_cnt.setText(str(sent))

    def update_global_status(self):
        self.lbl_ready.setText(f"{GlobalState.readyYN}")
        if GlobalState.readyYN == 1:
            self.lbl_ready.setStyleSheet("background: #AAFFAA; padding: 5px;")
        else:
            self.lbl_ready.setStyleSheet("background: #FFAAAA; padding: 5px;")
        try:
            self.lbl_retry.setText(str(self.worker.system.retry_cnt))
        except:
            pass

    def run_feeder_cmd(self, cmd_type):
        feeder = self.worker.system.feeder
        if not feeder.is_connected:
            QMessageBox.warning(self, "Error", "Feeder is not connected!")
            return
        threading.Thread(target=self._feeder_thread, args=(feeder, cmd_type)).start()

    def _feeder_thread(self, feeder, cmd_type):
        if cmd_type == 1:
            feeder.run_vibration(1, 1000)
        elif cmd_type == 2:
            feeder.client.write_register(2, 2, slave=1)
            feeder.client.write_register(0, 1, slave=1)
        elif cmd_type == 3:
            feeder.client.write_register(2, 3, slave=1)
            feeder.client.write_register(0, 1, slave=1)

    def load_initial_config(self):
        """초기 실행 시 제품 및 글로벌 설정 로드"""
        self.load_product_config()
        self.load_global_config()

    def load_global_config(self):
        """시스템 변수 값을 UI로 불러오기"""
        try:
            inspector = self.worker.system.inspector
            
            # Config 값 로드
            self.spin_pixel.setValue(inspector.pixel_to_mm)
            self.spin_retry.setValue(inspector.max_retry_count) # Max Retry 값
            self.spin_min_area.setValue(inspector.bush_min_area)
            self.spin_max_area.setValue(inspector.bush_max_area)
            # [신규]
            self.spin_min_dist_mm.setValue(inspector.min_center_dist_mm)

            self.spin_thresh_block.setValue(inspector.thresh_block_size)
            self.spin_thresh_c.setValue(inspector.thresh_c)
            
            # ReadyYN은 GlobalState에서 가져와서 표기만 함
            self.spin_ready.setText(str(GlobalState.readyYN))
            
        except Exception as e:
            logging.error(f"Global config load failed: {e}")

    def apply_global_config(self):
        """UI 값을 시스템 변수 및 파일에 저장"""
        try:
            inspector = self.worker.system.inspector
            
            # 1. 딕셔너리 업데이트
            inspector.global_params["pixel_to_mm"] = self.spin_pixel.value()
            inspector.global_params["max_retry_count"] = self.spin_retry.value()
            inspector.global_params["bush_min_area"] = self.spin_min_area.value()
            inspector.global_params["bush_max_area"] = self.spin_max_area.value()
            # [신규]
            inspector.global_params["min_center_dist_mm"] = self.spin_min_dist_mm.value()
            
            # Block Size 홀수 보정
            block_val = self.spin_thresh_block.value()
            if block_val % 2 == 0: 
                block_val += 1
                self.spin_thresh_block.setValue(block_val)
            inspector.global_params["thresh_block_size"] = block_val
            inspector.global_params["thresh_c"] = self.spin_thresh_c.value()
            
            # 2. 메모리(멤버 변수) 적용
            inspector.apply_global_params()
            
            # 3. 파일 저장 (global_config.json)
            inspector.save_global_config()
            
            QMessageBox.information(self, "Success", "Global settings saved to file and applied!")
            logging.info("Global settings saved.")
            
            # ReadyYN은 여기서 수정하지 않음 (Read Only)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply settings: {e}")
            
    def load_product_config(self):
        idx = self.combo_prod.currentIndex() + 1
        idx_str = str(idx)
        configs = self.worker.system.inspector.product_configs
        if idx_str in configs:
            params = configs[idx_str]
            self.config_table.setRowCount(len(params))
            for row, (key, val) in enumerate(params.items()):
                self.config_table.setItem(row, 0, QTableWidgetItem(key))
                self.config_table.setItem(row, 1, QTableWidgetItem(str(val)))
                self.config_table.item(row, 0).setFlags(Qt.ItemIsEnabled)

    def save_product_config(self):
        idx = self.combo_prod.currentIndex() + 1
        idx_str = str(idx)
        new_params = {}
        for row in range(self.config_table.rowCount()):
            key = self.config_table.item(row, 0).text()
            val_str = self.config_table.item(row, 1).text()
            try:
                if "." in val_str: val = float(val_str)
                else: val = int(val_str)
            except: val = val_str
            new_params[key] = val
        self.worker.system.inspector.product_configs[idx_str] = new_params
        
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(base_dir, "product_config.json")
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.worker.system.inspector.product_configs, f, indent=4)
            QMessageBox.information(self, "Success", f"Product {idx} saved successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {e}")

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())