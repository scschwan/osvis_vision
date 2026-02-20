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
                               QTabWidget, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QCheckBox,
                               QGroupBox, QTextEdit, QLineEdit, QComboBox, QHeaderView, QSplitter, QDoubleSpinBox,
                               QFormLayout, QMessageBox, QFrame, QGridLayout, QFileDialog, QSpinBox, QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem) 
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThread, Slot
from PySide6.QtGui import QImage, QPixmap, QColor, QFont, QPainter

from main_app import InspectionSystem, IntegratedVisionServer
from vision_server import GlobalState

class ImageGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(30, 30, 30))
        self.setFrameShape(QFrame.NoFrame)
        self.zoom_factor = 1.15

    def set_image(self, cv_image):
        if cv_image is None: return
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(0, 0, w, h)
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def set_pixmap(self, pixmap):
        if pixmap is None: return
        self.pixmap_item.setPixmap(pixmap)
        w = pixmap.width()
        h = pixmap.height()
        self.scene.setSceneRect(0, 0, w, h)
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.scale(self.zoom_factor, self.zoom_factor)
        else:
            self.scale(1 / self.zoom_factor, 1 / self.zoom_factor)

class QtLogHandler(logging.Handler, QObject):
    log_signal = Signal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)

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
            if not self.running: 
                return "#STOP", None, {}

            self.total_inspect_count += 1
            result_msg, result_img, all_candidates = original_process(top_id, bot_id)
            
            if "#OK" in result_msg:
                self.total_sent_count += 1
            self.stats_signal.emit(self.total_inspect_count, self.total_sent_count)
            
            if result_img is None:
                current_img = self.system.camera.get_latest_image()
            else:
                current_img = result_img
            
            self.result_signal.emit(current_img, result_msg, all_candidates)
            return result_msg, result_img, all_candidates
            
        self.system.process_inspection_scenario = hooked_process

        try:
            if self.running:
                self.loop.run_until_complete(self.server.start_server())
        except Exception as e:
            logging.error(f"Server Error: {e}")
        finally:
            self.system.feeder.disconnect()
            self.system.camera.stop()
            if self.loop.is_running():
                self.loop.close()

    def stop(self):
        self.running = False
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.quit()
        self.wait(2000)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Inspection System (Feeder Control)")
        self.resize(1280, 1024)

        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
        log_dir = os.path.join(base_dir, "log")
        os.makedirs(log_dir, exist_ok=True)
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_path = os.path.join(log_dir, f"{date_str}.log")
        
        formatter = logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S')
        
        self.log_handler = QtLogHandler()
        self.log_handler.setFormatter(formatter)
        self.log_handler.log_signal.connect(self.update_log)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        if not any(isinstance(h, QtLogHandler) for h in root_logger.handlers):
            root_logger.addHandler(self.log_handler)
            
        has_file_handler = False
        for h in root_logger.handlers:
            if isinstance(h, logging.FileHandler):
                if h.baseFilename == os.path.abspath(log_path):
                    has_file_handler = True
                    break
        
        if not has_file_handler:
            root_logger.addHandler(file_handler)

        self.worker = SystemWorker()
        self.worker.result_signal.connect(self.update_inspection_result)
        self.worker.stats_signal.connect(self.update_stats)
        self.worker.start()

        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_global_status)
        self.timer.start(1000)

        self.manual_test_image_path = ""

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

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

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.create_main_tab()
        self.create_comm_tab()
        self.create_config_tab()
        self.create_log_tab()

    def create_main_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

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

        # [수정] 통계 그룹에 Est Qty, Valid TOP/BOT 추가
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout()
        self.lbl_total_cnt = QLabel("0")
        self.lbl_sent_cnt = QLabel("0")
        
        # 신규 라벨들
        self.lbl_est_qty = QLabel("0")
        self.lbl_valid_top = QLabel("0")
        self.lbl_valid_bot = QLabel("0")
        
        self.lbl_est_qty.setStyleSheet("font-weight: bold; color: blue;")
        self.lbl_valid_top.setStyleSheet("font-weight: bold; color: green;")
        self.lbl_valid_bot.setStyleSheet("font-weight: bold; color: green;")

        self.lbl_result_status = QLabel("WAITING") 
        self.lbl_result_status.setAlignment(Qt.AlignCenter)
        self.lbl_result_status.setStyleSheet("font-size: 20px; font-weight: bold; color: gray; border: 2px solid gray; padding: 10px;")

        stats_layout.addRow("Total Inspections:", self.lbl_total_cnt)
        stats_layout.addRow("Items Sent (OK):", self.lbl_sent_cnt)
        stats_layout.addRow("Estimated Qty:", self.lbl_est_qty) # 추가
        stats_layout.addRow("Valid TOP:", self.lbl_valid_top)   # 추가
        stats_layout.addRow("Valid BOT:", self.lbl_valid_bot)   # 추가
        
        stats_group.setLayout(stats_layout)
        right_panel.addWidget(stats_group)
        right_panel.addWidget(self.lbl_result_status)

        table_group = QGroupBox("Inspection Results")
        table_layout = QVBoxLayout()
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(5)
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

    def create_config_tab(self):
        tab = QWidget()
        main_layout = QHBoxLayout(tab)

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

        global_group = QGroupBox("Global System & Algorithm Settings (Saved in global_config.json)")
        global_layout = QVBoxLayout()
        form_layout = QFormLayout()
        
        self.spin_pixel = QDoubleSpinBox()
        self.spin_pixel.setDecimals(6)
        self.spin_pixel.setRange(0.000001, 1.0)
        self.spin_pixel.setSingleStep(0.000001)
        form_layout.addRow("Pixel to MM:", self.spin_pixel)
        
        self.spin_ready = QLineEdit()
        self.spin_ready.setReadOnly(True)
        self.spin_ready.setStyleSheet("background-color: #EEE; color: #555;")
        form_layout.addRow("Global ReadyYN:", self.spin_ready)
        
        self.spin_retry = QSpinBox()
        self.spin_retry.setRange(0, 100)
        form_layout.addRow("Max Retry Count:", self.spin_retry)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        form_layout.addRow(line)
        
        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(0, 999999)
        self.spin_min_area.setSingleStep(1000)
        form_layout.addRow("Bush Min Area:", self.spin_min_area)
        
        self.spin_max_area = QSpinBox()
        self.spin_max_area.setRange(0, 999999)
        self.spin_max_area.setSingleStep(1000)
        form_layout.addRow("Bush Max Area:", self.spin_max_area)

        self.spin_min_dist_mm = QDoubleSpinBox()
        self.spin_min_dist_mm.setRange(0.0, 100.0)
        self.spin_min_dist_mm.setSingleStep(0.1)
        form_layout.addRow("Overlap Min Dist (mm):", self.spin_min_dist_mm)
        
        self.spin_thresh_block = QSpinBox()
        self.spin_thresh_block.setRange(3, 99)
        self.spin_thresh_block.setSingleStep(2)
        form_layout.addRow("Thresh Block Size:", self.spin_thresh_block)
        
        self.spin_thresh_c = QDoubleSpinBox()
        self.spin_thresh_c.setRange(-100, 100)
        form_layout.addRow("Thresh C Value:", self.spin_thresh_c)

        # [추가] Simple Threshold 설정 UI
        self.spin_simple_thresh = QSpinBox()
        self.spin_simple_thresh.setRange(0, 255)
        form_layout.addRow("Simple Thresh Val:", self.spin_simple_thresh)
        
        self.spin_canny1 = QSpinBox()
        self.spin_canny1.setRange(0, 255)
        form_layout.addRow("Canny Low:", self.spin_canny1)
        
        self.spin_canny2 = QSpinBox()
        self.spin_canny2.setRange(0, 255)
        form_layout.addRow("Canny High:", self.spin_canny2)
        
        self.spin_blur_size = QSpinBox()
        self.spin_blur_size.setRange(0, 50)
        self.spin_blur_size.setSingleStep(2)
        form_layout.addRow("Canny Blur Size:", self.spin_blur_size)

        global_layout.addLayout(form_layout)
        
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

    def create_log_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        left_panel = QWidget()
        left_panel.setFixedWidth(420)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        ctrl_group = QGroupBox("1. Manual Test Controls")
        ctrl_layout = QVBoxLayout()

        btn_upload = QPushButton("📂 이미지 불러오기 (Local)")
        btn_upload.setStyleSheet("padding: 8px; font-weight: bold;")
        btn_upload.clicked.connect(self.load_local_image)
        ctrl_layout.addWidget(btn_upload)

        self.lbl_file_path = QLabel("선택된 파일 없음")
        self.lbl_file_path.setStyleSheet("color: gray; font-size: 11px;")
        self.lbl_file_path.setWordWrap(True)
        ctrl_layout.addWidget(self.lbl_file_path)

        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Target Product ID:"))
        self.spin_test_id = QSpinBox()
        self.spin_test_id.setRange(1, 25)
        self.spin_test_id.setValue(1)
        self.spin_test_id.setMinimumHeight(30)
        input_layout.addWidget(self.spin_test_id)
        ctrl_layout.addLayout(input_layout)

        btn_start_test = QPushButton("▶ 수동 검사 수행")
        btn_start_test.setStyleSheet("background-color: #CCCCFF; padding: 10px; font-weight: bold; font-size: 14px;")
        btn_start_test.clicked.connect(self.run_manual_test)
        ctrl_layout.addWidget(btn_start_test)
        
        self.chk_show_binary = QCheckBox("Show Binary Image (Debug)")
        self.chk_show_binary.setStyleSheet("font-weight: bold; color: blue;")
        self.chk_show_binary.stateChanged.connect(self.toggle_test_image_view)
        ctrl_layout.addWidget(self.chk_show_binary)
        
        ctrl_group.setLayout(ctrl_layout)
        left_layout.addWidget(ctrl_group)

        res_group = QGroupBox("2. Inspection Details")
        res_layout = QVBoxLayout()
        self.txt_test_result = QTextEdit()
        self.txt_test_result.setReadOnly(True)
        self.txt_test_result.setPlaceholderText("검사 결과 데이터가 여기에 표시됩니다.")
        res_layout.addWidget(self.txt_test_result)
        res_group.setLayout(res_layout)
        left_layout.addWidget(res_group, stretch=1)

        log_group = QGroupBox("3. System Logs")
        log_layout = QVBoxLayout()
        self.txt_full_log = QTextEdit()
        self.txt_full_log.setReadOnly(True)
        self.txt_full_log.setStyleSheet("font-family: Consolas; font-size: 11px;")
        log_layout.addWidget(self.txt_full_log)
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group, stretch=1)

        layout.addWidget(left_panel)

        image_group = QGroupBox("Inspection Result Image (Wheel to Zoom, Drag to Pan)")
        image_layout = QVBoxLayout()
        self.test_image_viewer = ImageGraphicsView()
        image_layout.addWidget(self.test_image_viewer)
        image_group.setLayout(image_layout)
        layout.addWidget(image_group, stretch=1)

        self.tabs.addTab(tab, "Logs & Test")

    def load_local_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        if file_dialog.exec():
            files = file_dialog.selectedFiles()
            if files:
                self.manual_test_image_path = files[0]
                self.lbl_file_path.setText(self.manual_test_image_path)
                pixmap = QPixmap(self.manual_test_image_path)
                self.test_image_viewer.set_pixmap(pixmap)
                self.txt_test_result.setText("이미지 로드됨. 검사 대기 중...")

    def toggle_test_image_view(self):
        if hasattr(self, 'current_test_result_dir') and hasattr(self, 'current_test_filename'):
            self.load_manual_result_images(self.current_test_result_dir, self.current_test_filename)

    def run_manual_test(self):
        if not self.manual_test_image_path or not os.path.exists(self.manual_test_image_path):
            QMessageBox.warning(self, "Error", "이미지 파일을 먼저 선택해주세요.")
            return

        try:
            stream = open(self.manual_test_image_path.encode("utf-8"), "rb")
            bytes = bytearray(stream.read())
            numpy_array = np.asarray(bytes, dtype=np.uint8)
            image = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
            stream.close()
            
            if image is None:
                QMessageBox.warning(self, "Error", "이미지 로드 실패.")
                return

            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))

            now_dt = datetime.now()
            date_str = now_dt.strftime("%Y%m%d")
            time_str = now_dt.strftime("%H%M%S")

            orig_dir = os.path.join(base_dir, "original", date_str)
            res_dir = os.path.join(base_dir, "result", date_str)
            os.makedirs(orig_dir, exist_ok=True)
            os.makedirs(res_dir, exist_ok=True)

            orig_filename = f"{time_str}_manual_raw.jpg"
            cv2.imwrite(os.path.join(orig_dir, orig_filename), image)

            inspector = self.worker.system.inspector
            target_id = self.spin_test_id.value()
            inspector.set_target_product(target_id)
            
            res_filename = f"{time_str}_manual_result" 
            
            results, vis_img = inspector.process_image_manual(image, res_dir, image_name=res_filename)
            
            if vis_img is not None:
                self.test_image_viewer.set_image(vis_img)
                save_path = os.path.join(res_dir, f"result_{res_filename}.jpg")
                cv2.imwrite(save_path, vis_img)

            self.current_test_result_dir = res_dir
            self.current_test_filename = res_filename
            
            result_text = f"검사 완료 (Product ID: {target_id})\n"
            result_text += f"검출된 객체 수: {len(results)}\n"
            
            for res in results:
                result_text += f"ID:{res.bush_id} Type:{res.text_type} Surf:{res.surface_type}\n"
                result_text += f"Pos(mm):({res.center_x_mm:.2f}, {res.center_y_mm:.2f}) Ang:{res.rotation_angle}\n"
            
            self.txt_test_result.setText(result_text)
            self.load_manual_result_images(res_dir, res_filename)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"검사 중 오류 발생:\n{e}")
            logging.error(f"Manual Test Error: {e}")

    def load_manual_result_images(self, res_dir, res_filename):
        if self.chk_show_binary.isChecked():
            img_path = os.path.join(res_dir, f"binary_{res_filename}.jpg")
        else:
            img_path = os.path.join(res_dir, f"result_{res_filename}.jpg")
            
        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            self.test_image_viewer.set_pixmap(pixmap)
        else:
            self.txt_test_result.append(f"\n[Warning] 이미지를 찾을 수 없습니다: {os.path.basename(img_path)}")

    @Slot(str)
    def update_log(self, msg):
        self.txt_full_log.append(msg)
        if "RECV" in msg or "SEND" in msg or "Client" in msg:
            self.txt_socket_log.append(msg)
        else:
            self.txt_feeder_log.append(msg)

    # [수정] Valid 개수와 Est Qty를 받아서 라벨 업데이트
    @Slot(object, object, str)
    def update_inspection_result(self, image, msg, all_candidates):
        if image is not None:
            self.image_viewer.set_image(image)

        if "#OK" in msg:
            self.lbl_result_status.setText("SUCCESS")
            self.lbl_result_status.setStyleSheet("font-size: 24px; font-weight: bold; color: green; border: 3px solid green; background: #E0FFE0;")
        elif "EMPTY" in msg:
            if "EMPTYTOP" in msg:
                self.lbl_result_status.setText("FAIL (TOP EMPTY)")
            elif "EMPTYBOT" in msg:
                self.lbl_result_status.setText("FAIL (BOT EMPTY)")
            else:
                self.lbl_result_status.setText("FAIL (EMPTY)")
            self.lbl_result_status.setStyleSheet("font-size: 24px; font-weight: bold; color: red; border: 3px solid red; background: #FFE0E0;")
            
        # all_candidates에서 추가 데이터 읽기
        if all_candidates:
            est = all_candidates.get("est_count", 0)
            vt = all_candidates.get("valid_top_cnt", 0)
            vb = all_candidates.get("valid_bot_cnt", 0)
            
            self.lbl_est_qty.setText(str(est))
            self.lbl_valid_top.setText(str(vt))
            self.lbl_valid_bot.setText(str(vb))
            
        self.update_result_table(msg, all_candidates)

    def update_result_table(self, msg, all_candidates):
        selected_coords = []
        if "#OK" in msg:
            try:
                data = msg.replace("#OK,", "").replace("#OK2,", "").replace(";", "").split(",")
                if len(data) >= 8:
                    selected_coords.append((float(data[0]), float(data[1])))
                    selected_coords.append((float(data[4]), float(data[5])))
            except: pass

        if not all_candidates:
            self.result_table.setRowCount(0)
            return

        tops = all_candidates.get("tops", [])
        bots = all_candidates.get("bots", [])
        
        selected_list = []
        other_tops = []
        other_bots = []

        def is_selected(res):
            x = res.center_x_mm
            y = res.center_y_mm
            for (sx, sy) in selected_coords:
                if abs(x - sx) < 0.1 and abs(y - sy) < 0.1:
                    return True
            return False

        for res in tops:
            if is_selected(res): selected_list.append(("TOP", res))
            else: other_tops.append(("TOP", res))
        
        for res in bots:
            if is_selected(res): selected_list.append(("BOT", res))
            else: other_bots.append(("BOT", res))

        final_list = selected_list + other_tops + other_bots
        
        self.result_table.setRowCount(len(final_list))
        dir_map = {1: "E", 2: "S", 3: "W", 4: "N", 0: "Err"}
        
        for row_idx, (type_str, res) in enumerate(final_list):
            self._add_row_to_table(row_idx, type_str, res, selected_coords, dir_map)

    def _add_row_to_table(self, row, type_str, res, selected_coords, dir_map):
        x_val = res.center_x_mm
        y_val = res.center_y_mm
        angle_val = res.final_angle
        dir_str = dir_map.get(res.direction_code, str(res.direction_code))
        
        items = [
            QTableWidgetItem(f"{type_str}"),
            QTableWidgetItem(f"{x_val:.2f}"),
            QTableWidgetItem(f"{y_val:.2f}"),
            QTableWidgetItem(f"{angle_val:.2f}"),
            QTableWidgetItem(dir_str)
        ]
        
        is_selected = False
        for (sel_x, sel_y) in selected_coords:
            if abs(x_val - sel_x) < 0.1 and abs(y_val - sel_y) < 0.1:
                is_selected = True
                break
        
        bg_color = QColor("yellow") if is_selected else QColor("white")
        
        for col, item in enumerate(items):
            item.setBackground(bg_color)
            item.setTextAlignment(Qt.AlignCenter)
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


            # 2. RetryCnt 및 Top/Bot ID 업데이트 (시스템 접근)
        if hasattr(self, 'worker') and hasattr(self.worker, 'system'):
            sys_obj = self.worker.system
            
            # [수정] Retry Count 업데이트
            self.lbl_retry.setText(str(sys_obj.retry_cnt))
            
            # [추가] Top/Bot Product ID 업데이트
            self.lbl_top.setText(str(sys_obj.current_top_id))
            self.lbl_bot.setText(str(sys_obj.current_bot_id))
            
        try:
            self.lbl_retry.setText(str(self.worker.system.retry_cnt))
        except: pass

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
        self.load_product_config()
        self.load_global_config()

    def load_global_config(self):
        try:
            inspector = self.worker.system.inspector
            self.spin_pixel.setValue(inspector.pixel_to_mm)
            self.spin_retry.setValue(inspector.max_retry_count)
            self.spin_min_area.setValue(inspector.bush_min_area)
            self.spin_max_area.setValue(inspector.bush_max_area)
            self.spin_min_dist_mm.setValue(inspector.min_center_dist_mm)
            self.spin_thresh_block.setValue(inspector.thresh_block_size)
            self.spin_thresh_c.setValue(inspector.thresh_c)

            # [추가] Simple Threshold 로드
            self.spin_simple_thresh.setValue(inspector.simple_thresh_val)

            self.spin_canny1.setValue(inspector.canny_thresh1)
            self.spin_canny2.setValue(inspector.canny_thresh2)
            self.spin_blur_size.setValue(inspector.canny_blur_size)
            
            self.spin_ready.setText(str(GlobalState.readyYN))
        except Exception as e:
            logging.error(f"Global config load failed: {e}")

    def apply_global_config(self):
        try:
            inspector = self.worker.system.inspector
            inspector.global_params["pixel_to_mm"] = self.spin_pixel.value()
            inspector.global_params["max_retry_count"] = self.spin_retry.value()
            inspector.global_params["bush_min_area"] = self.spin_min_area.value()
            inspector.global_params["bush_max_area"] = self.spin_max_area.value()
            inspector.global_params["min_center_dist_mm"] = self.spin_min_dist_mm.value()
            
            block_val = self.spin_thresh_block.value()
            if block_val % 2 == 0: block_val += 1
            self.spin_thresh_block.setValue(block_val)
            inspector.global_params["thresh_block_size"] = block_val
            inspector.global_params["thresh_c"] = self.spin_thresh_c.value()

            # [추가] Simple Threshold 저장
            inspector.global_params["simple_thresh_val"] = self.spin_simple_thresh.value()
            
            inspector.global_params["canny_thresh1"] = self.spin_canny1.value()
            inspector.global_params["canny_thresh2"] = self.spin_canny2.value()
            inspector.global_params["canny_blur_size"] = self.spin_blur_size.value()
            
            inspector.apply_global_params()
            inspector.save_global_config()
            QMessageBox.information(self, "Success", "Global settings saved!")
            logging.info("Global settings saved.")
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
        logging.info("Terminating Application...")
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        if hasattr(self, 'worker'):
            self.worker.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())