# Code Review: OSVIS Vision Inspection System

> **Review Date**: 2026-02-20
> **Scope**: `main_ui.py`, `main_app.py` 중심, 관련 모듈(`bush_inspector.py`, `camera_module.py`, `feeder_control.py`, `vision_server.py`) 포함
> **Project**: 산업용 비전 검사 시스템 (Bush 부품 자동 검사)

---

## 1. Architecture Overview

### 1.1 시스템 구성도

```
┌─────────────────────────────────────────────────────────────────┐
│                        main_ui.py                               │
│                   (PySide6 GUI Application)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │Main      │  │Communi-  │  │Config    │  │Logs &        │    │
│  │Monitor   │  │cation    │  │Tab       │  │Manual Test   │    │
│  │Tab       │  │Tab       │  │          │  │Tab           │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘    │
│       │                                          │              │
│       │         SystemWorker (QThread)            │              │
│       └──────────────┬───────────────────────────┘              │
└──────────────────────┼──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                       main_app.py                                │
│                  (Inspection Orchestrator)                        │
│                                                                  │
│  ┌─────────────────────┐    ┌──────────────────────────────┐     │
│  │  InspectionSystem   │    │  IntegratedVisionServer      │     │
│  │  - initialize()     │    │  (extends VisionServer)      │     │
│  │  - process_scenario │◄───│  - perform_inspection()      │     │
│  │  - retry logic      │    │  - handle_manual_command()   │     │
│  │  - disk management  │    └──────────────────────────────┘     │
│  └────────┬────────────┘                                         │
│           │                                                      │
└───────────┼──────────────────────────────────────────────────────┘
            │
    ┌───────┼───────────────────────┐
    │       │                       │
    ▼       ▼                       ▼
┌────────┐ ┌──────────────┐  ┌──────────────┐
│Camera  │ │ Bush         │  │ Feeder       │
│Module  │ │ Inspector    │  │ Control      │
│(Basler)│ │ (OpenCV)     │  │ (Modbus TCP) │
└────────┘ └──────────────┘  └──────────────┘
```

### 1.2 모듈 의존 관계

| 모듈 | 파일 | 역할 | 의존 |
|------|------|------|------|
| **GUI** | `main_ui.py` (908 lines) | PySide6 기반 4-탭 GUI | main_app, vision_server |
| **Orchestrator** | `main_app.py` (306 lines) | 검사 시나리오 조율, 재시도 로직 | bush_inspector, camera_module, feeder_control, vision_server |
| **Inspector** | `bush_inspector.py` (786 lines) | 이미지 처리 및 객체 검출 엔진 | OpenCV, numpy |
| **Camera** | `camera_module.py` (178 lines) | Basler 카메라 프레임 캡처 | pypylon |
| **Feeder** | `feeder_control.py` (180 lines) | Modbus TCP 피더 제어 | pymodbus |
| **Server** | `vision_server.py` (151 lines) | TCP 소켓 서버 (외부 통신) | asyncio |

### 1.3 데이터 흐름

```
외부 클라이언트               시스템 내부
    │
    │  #ALIGN,1,2;
    ▼
VisionServer ──► InspectionSystem.process_inspection_scenario()
                      │
                      ├─ 1. Feeder Light ON (Modbus)
                      ├─ 2. Camera Grab (Basler)
                      ├─ 3. Feeder Light OFF
                      ├─ 4. BushInspector.process_image()
                      │       ├─ preprocess_image()    → Binary 이미지 생성
                      │       ├─ detect_bushes()       → 후보 객체 검출
                      │       ├─ calculate_angles()    → 각도 계산
                      │       ├─ check_surface_type()  → TOP/BOT 판별
                      │       ├─ detect_text_region()  → 텍스트 영역 검출
                      │       ├─ determine_direction() → 방향(E/S/W/N) 결정
                      │       └─ visualize_result()    → 결과 이미지 생성
                      ├─ 5. _find_closest_to_center()  → 최적 후보 선택
                      └─ 6. 응답 생성
                              │
                              ▼
                  #OK,x1,y1,ang1,dir1,x2,y2,ang2,dir2;
```

---

## 2. Module Analysis: `main_app.py`

### 2.1 클래스 구조

#### `InspectionSystem`
시스템 전체를 조율하는 핵심 클래스. 하드웨어 초기화, 검사 시나리오 실행, 재시도 로직을 담당한다.

**멤버 변수**:

| 변수 | 타입 | 용도 |
|------|------|------|
| `feeder` | `FeederController` | Modbus TCP 피더 제어기 |
| `camera` | `CameraController` | Basler 카메라 제어기 |
| `inspector` | `BushInspector` | 이미지 처리 엔진 |
| `IMG_CENTER_X/Y` | `int` | 이미지 중심 좌표 (2736, 1577) - 하드코딩 |
| `retry_cnt` | `int` | 현재 재시도 횟수 |
| `last_sent_status` | `str` | 마지막 전송 상태 ("OK", "OK2", "EMPTY") |

### 2.2 핵심 함수 분석

#### `process_inspection_scenario(top_prod_id, bot_prod_id)` — Line 98~241

검사 시나리오의 메인 루프. 가장 중요한 함수이다.

**알고리즘 흐름**:

```
시작
  │
  ▼
retry_cnt = 0, max_retries = config에서 로드
  │
  ▼
┌─── while retry_cnt <= max_retries ──────────────────────┐
│                                                          │
│  1. Feeder Light ON (Register 2=2, Start)                │
│  2. 300ms 대기                                           │
│  3. Camera에서 최신 이미지 캡처                            │
│  4. Feeder Light OFF (Register 2=3, Start)               │
│  5. 디스크 용량 관리 (>90% 시 오래된 폴더 삭제)             │
│  6. 원본 이미지 저장 (original/YYYYMMDD/)                  │
│  7. BushInspector.process_image() 호출                    │
│     → results_all, vis_img, total_contour_area 반환       │
│  8. valid_tops 필터링:                                    │
│     surface_type=="TOP" AND text_type=="front"            │
│     AND edge_check_result=="edge_detect"                  │
│  9. valid_bots 필터링: 동일 조건 (BOT)                    │
│ 10. _find_closest_to_center()로 최적 TOP/BOT 선택         │
│ 11. 제품 수량 추정 (total_area / avg_area)                 │
│                                                          │
│  ┌── best_top AND best_bot 모두 존재? ──┐                 │
│  │ YES                                  │ NO              │
│  │ → 결과 이미지 저장                    │ → 실패 이미지 저장│
│  │ → total_sets < 2 ? "#OK2" : "#OK"    │ → retry_cnt++   │
│  │ → return msg, img, candidates        │ → 피더 진동 1초  │
│  └──────────────────────────────────────┘ → 1.3초 대기     │
│                                           → continue      │
└──────────────────────────────────────────────────────────┘
  │
  ▼ (최대 재시도 초과)
실패 유형 분기:
  - valid_tops < 3 AND valid_bots >= 3 → #EMPTYTOP
  - valid_bots < 3 AND valid_tops >= 3 → #EMPTYBOT
  - 그 외 → #EMPTY
```

**응답 프로토콜**:

| 응답 코드 | 의미 | 조건 |
|-----------|------|------|
| `#OK` | 정상 검출 | TOP+BOT 검출, 유효 세트 >= 2 |
| `#OK2` | 저밀도 검출 | TOP+BOT 검출, 유효 세트 < 2 |
| `#EMPTYTOP` | TOP 부족 | valid_tops < 3, valid_bots >= 3 |
| `#EMPTYBOT` | BOT 부족 | valid_bots < 3, valid_tops >= 3 |
| `#EMPTY` | 전체 부족 | 양쪽 모두 부족 |

#### `_find_closest_to_center(results)` — Line 243~252

유효 후보 중 이미지 중심에 가장 가까운 객체를 선택하는 함수.

```python
# 유클리드 거리의 제곱값으로 비교 (sqrt 생략하여 성능 최적화)
dist = (res.center_x_px - IMG_CENTER_X)^2 + (res.center_y_px - IMG_CENTER_Y)^2
# 최소 거리 객체 반환
```

이미지 중심에 가까운 객체가 로봇 픽업에 가장 적합하다는 전제에 기반한다.

#### `_manage_disk_space()` — Line 73~96

디스크 사용량이 90% 이상일 때 `original/`, `result/` 디렉토리의 오래된 폴더 10개를 삭제한다. 정렬 후 앞에서부터 삭제하므로 날짜순(YYYYMMDD)으로 가장 오래된 데이터가 제거된다.

#### `IntegratedVisionServer` — Line 264~286

`VisionServer`를 상속하여 실제 검사 로직을 연결하는 브릿지 클래스.

- `perform_inspection(p1, p2)`: `#ALIGN` 명령 수신 시 `process_inspection_scenario` 호출
- `handle_manual_command()`: `#MANUAL` 명령 수신 시 `handle_manual_trigger` 호출

---

## 3. Module Analysis: `main_ui.py`

### 3.1 클래스 구조

#### `ImageGraphicsView` (Line 22~60)
`QGraphicsView`를 상속한 이미지 뷰어. 줌(마우스 휠)과 패닝(드래그)을 지원한다.

| 메서드 | 기능 |
|--------|------|
| `set_image(cv_image)` | OpenCV BGR 이미지를 Qt 포맷으로 변환 후 표시 |
| `set_pixmap(pixmap)` | QPixmap 직접 표시 |
| `wheelEvent(event)` | 마우스 휠로 1.15배 줌 인/아웃 |

#### `QtLogHandler` (Line 62~71)
Python `logging.Handler`와 `QObject`를 다중 상속하여 로그 메시지를 Qt Signal로 전달하는 브릿지 클래스.

#### `SystemWorker(QThread)` (Line 73~137)
백그라운드 스레드에서 `InspectionSystem`과 `IntegratedVisionServer`를 실행한다.

**핵심 패턴 — Monkey Patching 훅**:

```python
# Line 98~119: 원본 함수를 감싸서 UI 업데이트 Signal을 주입
original_process = self.system.process_inspection_scenario

def hooked_process(top_id, bot_id):
    self.total_inspect_count += 1
    result_msg, result_img, all_candidates = original_process(top_id, bot_id)
    if "#OK" in result_msg:
        self.total_sent_count += 1
    self.stats_signal.emit(...)       # 통계 업데이트
    self.result_signal.emit(...)      # 결과 이미지/메시지 전달
    return result_msg, result_img, all_candidates

self.system.process_inspection_scenario = hooked_process
```

`process_inspection_scenario`를 직접 교체하여 검사 결과가 발생할 때마다 GUI에 Signal을 발생시킨다. 인터페이스 변경 없이 UI 연동을 달성했으나, 유지보수 시 추적이 어려울 수 있다.

#### `MainWindow(QMainWindow)` (Line 139~908)
메인 윈도우. 4개의 탭으로 구성된다.

### 3.2 탭 구조 및 핵심 함수

#### Tab 1: Main Monitor (`create_main_tab`, Line 235~304)

| 영역 | 위젯 | 용도 |
|------|------|------|
| 좌측 | `ImageGraphicsView` | 실시간 검사 이미지 (줌/패닝) |
| 우측 상단 | Status Group | ReadyYN, TopProd, BotProd, RetryCnt |
| 우측 중단 | Statistics Group | 총 검사수, OK수, 추정 수량, Valid TOP/BOT 수 |
| 우측 하단 | Result Table | 검출 객체 리스트 (선택된 항목 노란색 하이라이트) |

#### Tab 2: Communication (`create_comm_tab`, Line 306~357)

피더 수동 제어 버튼(진동, 백라이트 ON/OFF)과 소켓 통신 로그 뷰어.

#### Tab 3: Config (`create_config_tab`, Line 359~476)

| 영역 | 내용 |
|------|------|
| 좌측: Product Parameters | 제품별(1~25) 파라미터 편집 테이블 |
| 우측: Global Settings | 시스템 전역 파라미터 (pixel_to_mm, retry, threshold 등) |

#### Tab 4: Logs & Test (`create_log_tab`, Line 478~549)

수동 테스트 기능. 로컬 이미지를 불러와 검사를 실행하고 결과를 확인할 수 있다.

### 3.3 핵심 함수 분석

#### `update_inspection_result(image, msg, all_candidates)` — Line 652~678

검사 결과 수신 시 UI 업데이트를 수행하는 슬롯 함수.

```
1. 검사 이미지를 ImageGraphicsView에 표시
2. 결과 메시지에 따라 상태 라벨 업데이트:
   - "#OK" → 녹색 "SUCCESS"
   - "EMPTYTOP" → 빨간색 "FAIL (TOP EMPTY)"
   - "EMPTYBOT" → 빨간색 "FAIL (BOT EMPTY)"
   - "EMPTY" → 빨간색 "FAIL (EMPTY)"
3. all_candidates에서 추정 수량, Valid TOP/BOT 수 추출하여 라벨 업데이트
4. update_result_table() 호출하여 테이블 갱신
```

#### `update_result_table(msg, all_candidates)` — Line 680~754

검출된 모든 객체를 테이블에 표시한다. 선택된 객체(실제 응답에 포함된 좌표)는 노란색 배경 + 볼드로 하이라이트한다.

**정렬 우선순위**: 선택된 항목 → 나머지 TOP → 나머지 BOT

#### `run_manual_test()` — Line 567~628

수동 검사 실행 함수. 로컬 이미지를 불러와 `BushInspector.process_image_manual()`을 호출하고 결과를 표시한다.

**이미지 로드 방식** (한글 경로 대응):
```python
stream = open(path.encode("utf-8"), "rb")
bytes = bytearray(stream.read())
numpy_array = np.asarray(bytes, dtype=np.uint8)
image = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
```

`cv2.imread()`가 한글 경로를 처리하지 못하는 문제를 우회하기 위해 바이트 스트림으로 읽은 후 `imdecode`를 사용한다.

#### `update_global_status()` — Line 761~782

1초 타이머에 의해 호출. `GlobalState.readyYN`에 따라 상태 라벨 색상을 변경하고, `InspectionSystem`의 retry 및 product ID를 UI에 반영한다.

---

## 4. Module Analysis: `bush_inspector.py`

### 4.1 핵심 알고리즘: 이미지 전처리

#### `preprocess_image()` — Line 163~206

두 가지 바이너리 이미지를 생성하는 이중 전처리 파이프라인.

```
원본 이미지
    │
    ▼
Grayscale 변환
    │
    ▼
Gaussian Blur (5x5)
    │
    ├──────────────────────────┐
    ▼                          ▼
 Adaptive Threshold          Simple Threshold
 (GAUSSIAN_C, BINARY_INV)    (BINARY_INV, configurable val)
    │                          │
    ▼                          ▼
 Morphology CLOSE (iter=1)   Morphology CLOSE (iter=3)
 → binary_merged_weak        → binary_merged_strong
   (세밀한 엣지 보존)           (덩어리 합침, 겹침 확인용)
```

- **Weak**: Adaptive Threshold + 약한 모폴로지 → 개별 객체의 경계를 정밀하게 보존
- **Strong**: Simple Threshold + 강한 모폴로지 → 인접 객체를 하나로 합쳐 덩어리 단위 검출

### 4.2 핵심 알고리즘: 객체 검출

#### `detect_bushes()` — Line 208~312

```
binary_strong 이미지에서 contour 검출
    │
    ▼
┌── 각 contour에 대해 ──────────────────────────┐
│                                                │
│  1. 이미지 경계(5px 이내) 접촉 객체 → 제외     │
│     (잘린 객체는 중심/각도 계산이 부정확)        │
│                                                │
│  2. 유효 객체만 total_contour_area에 합산       │
│     (수량 추정에 사용)                          │
│                                                │
│  3. 면적 기반 필터링:                           │
│     - area < bush_min_area → Small로 분류      │
│     - area > bush_max_area → Large로 분류      │
│     - 범위 내 → candidates에 추가              │
│                                                │
│  4. find_corner_center_simple()로              │
│     중심점 및 코너 4개 계산                     │
└────────────────────────────────────────────────┘
    │
    ▼
겹침 제거 (Overlap Removal):
  - 모든 후보 쌍의 중심 거리 계산
  - min_center_dist_px 이내 → 뒤쪽 객체 제거
  - numpy vectorized 연산으로 O(n^2) 최적화
    │
    ▼
BushInfo 객체 리스트 반환
```

#### `find_corner_center_simple(contour)` — Line 314~344

부품의 4개 코너를 검출하는 알고리즘.

```
1. contour를 Douglas-Peucker 알고리즘으로 근사 (epsilon = 0.5% 둘레)
2. minAreaRect()로 중심점 계산
3. 각 꼭짓점에서 중심까지 거리 계산
4. target_dist_px (= 3.6√2 mm를 px로 환산)와의 차이가 작은 점 우선 선택
5. 선택된 점 간 최소 30px 이상 이격 조건 적용
6. 최종 4개 코너를 각도순 정렬 (atan2)
```

### 4.3 핵심 알고리즘: 표면/방향 판별

#### `check_surface_type(image, center)` — Line 382~390

부품 중심 영역(반경 10px)의 평균 밝기로 TOP/BOT을 구분한다.

```
mean_brightness < 200 → "TOP" (어두운 면)
mean_brightness >= 200 → "BOT" (밝은 면)
```

#### `determine_direction(center, blob_box, part_angle)` — Line 368~380

텍스트 blob의 상대적 위치로 부품 방향(E/S/W/N)을 결정한다.

```
1. 중심 → blob 중심 벡터의 각도 계산 (atan2)
2. 부품 자체 회전 각도를 보정 (raw_angle - part_angle)
3. 보정된 각도를 4방위로 분류:
   -45° ~ 45°   → East (1)
    45° ~ 135°  → South (2)
   -135° ~ -45° → North (4)
   그 외         → West (3)
```

### 4.4 핵심 알고리즘: 텍스트 영역 검출

#### `detect_text_blob_region()` — Line 425~562

ROI 내에서 텍스트가 있는 영역을 찾는 다단계 필터링 알고리즘.

```
1. ROI 추출 (contour의 bounding rect)
2. 마스크 생성:
   - contour 내부만 유효 영역
   - outer_margin만큼 침식 (외곽 노이즈 제거)
   - inner_margin + 내부 반경만큼 중심원 제거 (중심 구멍 영역)
3. Canny Edge Detection (configurable thresholds)
4. 마스크 적용 후 유효 엣지만 추출
5. Dilation (5x5 커널, configurable iterations) → 엣지 뭉침
6. Contour 검출 후 각 blob에 대해:
   ┌─ 면적 필터 (min_area ~ max_area)
   ├─ 종횡비 필터 (min_ratio ~ max_ratio)
   ├─ 크기 필터 (min/max short/long side)
   ├─ 거리 필터 (중심에서 min_dist ~ max_dist)
   └─ 코너 근접 필터 (코너에서 40px 이내 → 제외)
7. 통과한 blob 중 최대 면적 선택
8. check_text_like_edges()로 엣지 복잡도 최종 검증
```

#### `check_text_like_edges()` — Line 392~422

blob 영역의 Canny 엣지 총 길이로 텍스트 존재 여부를 판단한다.

```
1. blob의 minAreaRect 기준으로 이미지 회전 및 크롭
2. Gaussian Blur → Canny Edge Detection
3. 검출된 엣지 contour의 총 arcLength 합산
4. min_edge_length_sum 이상 → "edge_detect" (텍스트 있음)
   미만 → "none_edge" (텍스트 없음)
```

### 4.5 좌표 변환

```python
# 픽셀 → mm 변환 (X축 반전)
center_x_mm = (img_width - center_x_px) * pixel_to_mm
center_y_mm = center_y_px * pixel_to_mm
```

X축이 반전되는 이유는 카메라 좌표계와 로봇 좌표계의 방향이 반대이기 때문이다.

---

## 5. Module Analysis: Supporting Modules

### 5.1 `camera_module.py` — CameraController

Basler pypylon SDK를 래핑한 카메라 제어 클래스.

| 메서드 | 기능 |
|--------|------|
| `connect_camera()` | 첫 번째 Basler 디바이스 연결 |
| `start_capture()` | `GrabStrategy_LatestImageOnly`로 연속 캡처 시작 |
| `_capture_loop()` | 백그라운드 스레드에서 프레임 갱신 (5000ms 타임아웃) |
| `get_latest_image()` | thread-safe하게 최신 프레임 반환 (`threading.Lock`) |
| `get_jpeg_bytes()` | JPEG 인코딩된 바이트 반환 (스트리밍용) |
| `stop()` | 그랩 중지, 카메라 해제 |

**스레딩 모델**: producer-consumer 패턴. 캡처 스레드가 `current_frame`을 갱신하고, 검사 로직이 `get_latest_image()`로 읽는다. `threading.Lock`으로 동기화.

### 5.2 `feeder_control.py` — FeederController

Modbus TCP 프로토콜로 산업용 피더(진동기)를 제어한다.

| 레지스터 | 주소 | 용도 |
|----------|------|------|
| P0.00 | 0 | Start(1)/Stop(0) 신호 |
| P0.02 | 2 | Formula 번호 선택 |
| P0.03 | 3 | 제어 모드 (1=Formula) |
| P0.10 | 10 | 백라이트 스위치 (ON/OFF) |
| P0.11 | 11 | 밝기 (0~1000) |
| P0.21 | 21 | 신호 타입 (0=Switch) |
| P0.99 | 99 | 설정 저장 |

**진동 실행 흐름**: Formula 번호 설정 → Start → `time.sleep(duration)` → Stop

### 5.3 `vision_server.py` — VisionServer

asyncio 기반 TCP 소켓 서버.

**프로토콜 형식**: `#COMMAND,arg1,arg2;`

| 명령어 | 동작 | 응답 |
|--------|------|------|
| `#READY;` | 시스템 준비 상태 확인 | `#READY,,,,,,;` 또는 `#NG,,,,,,;` |
| `#MANUAL;` | 수동 모드 전환 | `#MANUAL,OK;` |
| `#AUTO;` | 자동 모드 전환 | `#AUTO,OK;` |
| `#ALIGN,p1,p2;` | 검사 실행 | `#OK,x,y,a,d,x,y,a,d;` 등 |

**Sticky Packet 처리**: `#`와 `;`를 기준으로 버퍼에서 완전한 메시지를 파싱한다. 네트워크 패킷 분할/병합에 대응.

---

## 6. Configuration System

### 6.1 Product Config (`product_config.json`)

25개 제품별로 개별 파라미터를 설정할 수 있다. 텍스트 blob 검출에 사용되는 파라미터들이다.

| 파라미터 | 용도 | 예시 |
|----------|------|------|
| `outer_margin` | 외곽 침식 마진 (px) | 10 |
| `inner_margin` | 내부 원 추가 마진 (px) | 10 |
| `min_area` / `max_area` | blob 면적 범위 | 800 ~ 2200 |
| `min_short_side` / `max_short_side` | blob 단변 범위 | 15 ~ 37 |
| `min_long_side` / `max_long_side` | blob 장변 범위 | 50 ~ 80 |
| `min_ratio` / `max_ratio` | blob 종횡비 범위 | 0.2 ~ 0.7 |
| `min_dist` / `max_dist` | 중심에서 blob 거리 범위 | 175 ~ 192 |
| `min_edge_length_sum` | 엣지 길이 합 최소값 | 200 |
| `dilation_iter` | Dilation 반복 횟수 | 3 |

### 6.2 Global Config (`global_config.json`)

시스템 전역 파라미터. 모든 제품에 공통 적용된다.

| 파라미터 | 용도 | 기본값 |
|----------|------|--------|
| `pixel_to_mm` | 카메라 캘리브레이션 계수 | 0.028761 |
| `max_retry_count` | 최대 재시도 횟수 | 3 |
| `bush_min_area` / `bush_max_area` | 부품 면적 범위 (px^2) | 143000 ~ 150000 |
| `thresh_block_size` | Adaptive Threshold 블록 크기 | 15 |
| `thresh_c` | Adaptive Threshold 상수 | 3.0 |
| `simple_thresh_val` | Simple Threshold 값 | 125 |
| `min_center_dist_mm` | 객체 간 최소 거리 (mm) | 13.0 |
| `canny_thresh1` / `canny_thresh2` | Canny 엣지 임계값 | 30 / 100 |
| `canny_blur_size` | Canny 전 블러 커널 크기 | 3 |

---

## 7. Threading Model

```
┌─────────────────────────────────────────────┐
│              Main Thread (GUI)               │
│  - PySide6 Event Loop                       │
│  - QTimer (1초 주기 상태 업데이트)             │
│  - Signal/Slot 기반 UI 업데이트               │
└──────────────────┬──────────────────────────┘
                   │ Signal
                   ▼
┌─────────────────────────────────────────────┐
│         SystemWorker (QThread)               │
│  - asyncio Event Loop                       │
│  - VisionServer (TCP 소켓)                   │
│  - InspectionSystem (검사 로직)               │
│    └─ process_inspection_scenario()          │
│       (동기적으로 실행, 완료 시 Signal emit)   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│       Camera Capture Thread (daemon)         │
│  - 연속 프레임 캡처                           │
│  - threading.Lock으로 current_frame 보호      │
└─────────────────────────────────────────────┘
```

**주의 사항**: `SystemWorker` 내부에서 `asyncio` 이벤트 루프를 실행하면서, TCP 클라이언트 요청 수신 시 `process_inspection_scenario()`를 동기적으로 호출한다. 이 함수는 카메라 캡처, 이미지 처리, 피더 제어를 모두 블로킹으로 수행하므로, 검사 중에는 다른 TCP 명령 처리가 지연된다.

---

## 8. Review Findings

### 8.1 설계 관련

| # | 항목 | 위치 | 내용 |
|---|------|------|------|
| D-1 | 하드코딩된 이미지 중심 | `main_app.py:41-42` | `IMG_CENTER_X=2736`, `IMG_CENTER_Y=1577`이 하드코딩되어 있다. 카메라 해상도 변경 시 수정 필요. 설정 파일로 이동하는 것이 바람직하다. |
| D-2 | Monkey Patching 패턴 | `main_ui.py:98-119` | `process_inspection_scenario`를 런타임에 교체하는 방식은 디버깅이 어렵다. Observer 패턴이나 callback 등록 방식이 더 적절하다. |
| D-3 | asyncio + 동기 블로킹 혼용 | `main_app.py:98~241` | `VisionServer`는 asyncio 기반이지만 `process_inspection_scenario()`는 `time.sleep()`을 사용하는 동기 함수이다. 검사 중 이벤트 루프가 블로킹된다. |
| D-4 | GlobalState 클래스 | `vision_server.py:12-14` | 클래스 변수로 전역 상태를 관리한다. 멀티 인스턴스 시 상태 공유 문제가 있을 수 있으나, 단일 프로세스 구조에서는 수용 가능하다. |

### 8.2 안정성 관련

| # | 항목 | 위치 | 내용 |
|---|------|------|------|
| S-1 | Feeder 직접 레지스터 접근 | `main_app.py:117-118, 124-125` | `self.feeder.client.write_register()`로 직접 레지스터에 접근한다. `FeederController`에 `light_on()`/`light_off()` 메서드가 있으나 사용하지 않고 있다. 에러 핸들링이 누락된다. |
| S-2 | 하드웨어 초기화 실패 무시 | `main_app.py:60, 67` | Feeder/Camera 연결 실패 시 `return False`가 주석 처리되어 있어, 하드웨어 없이도 시스템이 시작된다. 테스트 편의를 위한 것으로 보이나, 프로덕션에서는 위험하다. |
| S-3 | bare except 사용 | `main_ui.py:688, 878` | `except:` (bare except)가 여러 곳에서 사용된다. 예외 유형을 명시하지 않아 디버깅이 어려울 수 있다. |
| S-4 | 변수 미정의 가능성 | `main_app.py:223-224` | retry 루프 내에서 `image is None`인 경우 `valid_tops`/`valid_bots`가 정의되지 않은 채 루프 외부에서 참조될 수 있다. |

### 8.3 코드 품질 관련

| # | 항목 | 위치 | 내용 |
|---|------|------|------|
| Q-1 | 중복 코드 | `bush_inspector.py:653~723, 725~786` | `process_image()`와 `process_image_manual()`의 로직이 거의 동일하다. `debug_mode` 파라미터로 통합 가능하다. |
| Q-2 | 빌트인 이름 섀도잉 | `main_ui.py:574, bush_inspector.py:662` | `bytes = bytearray(stream.read())`에서 Python 빌트인 `bytes`를 지역변수로 덮어쓴다. |
| Q-3 | 미사용 import | `main_ui.py:8` | `math`, `numpy` 등 일부 import가 직접 사용되지 않는다. |
| Q-4 | 중복 retry 갱신 | `main_ui.py:774, 781` | `update_global_status()`에서 `lbl_retry`를 두 번 업데이트한다. |
| Q-5 | 한영 혼용 | 전체 | 로그 메시지, 주석, UI 텍스트에 한글과 영어가 혼재한다. 일관성을 위해 하나로 통일하는 것이 좋다. |

---

## 9. 파일 구조 요약

```
osvis_vision/
├── main_ui.py              # GUI (PySide6, 4탭 구조)
├── main_app.py             # 검사 오케스트레이터
├── bush_inspector.py       # 이미지 처리 엔진 (OpenCV)
├── camera_module.py        # Basler 카메라 제어 (pypylon)
├── feeder_control.py       # 피더 제어 (Modbus TCP)
├── vision_server.py        # TCP 소켓 서버 (asyncio)
├── create_config.py        # 설정 파일 생성 유틸리티
├── product_config.json     # 제품별 파라미터 (25개)
├── global_config.json      # 전역 시스템 파라미터
├── logo.png                # 애플리케이션 로고
├── requirements.txt        # 의존 패키지 목록
├── VisionSystem_Dir.spec   # PyInstaller 빌드 설정
├── 251231 BACKUP/          # 이전 버전 백업
├── regacy/                 # 레거시 코드 (1차/2차 버전)
└── manual_test_result/     # 수동 테스트 결과 이미지
```

---

## 10. 주요 설정값 & 상수 정리

| 상수 | 값 | 위치 | 용도 |
|------|----|------|------|
| TCP 서버 포트 | 8000 | `main_app.py:266` | 외부 클라이언트 통신 |
| Feeder IP/Port | 192.168.1.100:502 | `main_app.py:37` | Modbus TCP 연결 |
| 이미지 중심 X | 2736 | `main_app.py:41` | 최적 후보 선택 기준 |
| 이미지 중심 Y | 1577 | `main_app.py:42` | 최적 후보 선택 기준 |
| Zoom Factor | 1.15 | `main_ui.py:35` | 마우스 휠 줌 배율 |
| Light ON 대기 | 300ms | `main_app.py:118` | 조명 안정화 시간 |
| 진동 후 대기 | 1300ms | `main_app.py:216-217` | 부품 안정화 시간 |
| 디스크 임계치 | 90% | `main_app.py:78` | 자동 정리 트리거 |
| Edge Margin | 5px | `bush_inspector.py:216` | 경계 접촉 판정 여유 |
| Corner Proximity | 40px | `bush_inspector.py:515` | 코너 간섭 판정 거리 |
| 밝기 임계값 | 200 | `bush_inspector.py:390` | TOP/BOT 구분 기준 |

---

## 11. 추가 개선 사항 (Roadmap)

### 11.1 딥러닝 기반 Bush 제품 검출

#### As-Is (현재)

현재 시스템은 `product_config.json`에 정의된 **Rule 기반 파라미터**(면적, 종횡비, 변 길이, 엣지 길이 합 등)로 제품을 분류한다.

```
현재 검출 흐름:
  이미지 → Threshold → Contour 검출 → 면적/크기 필터 → 텍스트 Blob 검출 → 방향 결정
                                        ↑
                              product_config.json의 규칙 기반 파라미터
```

**한계점**:
- #1~#25 제품 중 **동일한 외형 형질**(면적, 종횡비, 크기 등)을 가진 제품 간 구분 불가
- 새로운 제품 추가 시 수동으로 파라미터를 튜닝해야 하며, 기존 제품과 겹치는 파라미터 범위가 있으면 오분류 발생
- 표면의 미세 패턴, 각인, 색상 차이 등 고차원 특징을 활용하지 못함
- `check_surface_type()`이 단순 밝기 평균(< 200)에 의존하여 조명 조건 변화에 취약

#### To-Be (개선안)

딥러닝 객체 감지(Object Detection) 모델을 도입하여 **#1~#25 전체 제품을 정확히 분류**한다.

```
개선 검출 흐름:
  이미지 → DL 모델 추론 → [class_id, bbox, confidence] → 후처리
                ↑
        학습된 모델 (25-class classification)
```

**권장 접근 방식**:

| 항목 | 내용 |
|------|------|
| **모델** | YOLOv8/v11 (프로젝트에 이미 `ultralytics` 의존성 존재) 또는 RT-DETR |
| **Task** | Object Detection (25-class) + Surface Classification (TOP/BOT) |
| **학습 데이터** | 기존 `original/`, `result/` 디렉토리에 축적된 검사 이미지 활용 |
| **추론 환경** | OpenVINO (이미 `openvino` 의존성 존재) 또는 TensorRT로 최적화 |
| **Fallback** | DL 모델 confidence가 낮을 경우 기존 Rule 기반 로직으로 대체 (Hybrid 방식) |

**예상 변경 범위**:

```
bush_inspector.py:
  - 신규 클래스: DLBushClassifier
    - load_model(): ONNX/OpenVINO 모델 로드
    - predict(image): 추론 실행 → (class_id, bbox, confidence) 리스트 반환
  - process_image() 수정:
    - DL 추론 → 검출 결과에서 class_id로 product 매핑
    - confidence threshold 이하 → 기존 Rule 기반 fallback

main_app.py:
  - InspectionSystem.__init__(): DLBushClassifier 초기화
  - process_inspection_scenario(): DL 결과 우선, Rule 기반 보조

main_ui.py:
  - Config Tab: 모델 경로, confidence threshold 설정 UI 추가
  - Result Table: class_id, confidence 컬럼 추가

product_config.json:
  - 각 제품에 dl_class_name, confidence_threshold 필드 추가
```

**데이터 수집/라벨링 전략**:
1. 기존 시스템에서 축적된 검사 이미지를 활용 (이미 제품별로 분류 가능한 메타데이터 존재)
2. `visualize_result()`에서 생성하는 결과 이미지의 bbox 정보를 YOLO 라벨 포맷으로 자동 변환하는 유틸리티 작성
3. 초기 학습 → 현장 배포 → 오분류 케이스 수집 → 재학습 (Active Learning 사이클)

---

### 11.2 Picking 정위치 검증 비전 모듈

#### 배경

현재 시스템은 부품의 **위치(x, y)와 각도(angle), 방향(direction)**을 로봇에 전달하고, 로봇이 해당 좌표로 이동하여 picking을 수행한다. 그러나 picking 후 부품이 picker 중앙에 정확히 위치했는지 검증하는 단계가 없다.

#### 개선안: 하단 검증 카메라 시스템

```
                    ┌──────────┐
                    │  Picker  │
                    │  (흡착)  │
                    └────┬─────┘
                         │ 부품
                         ▼
              ┌──────────────────────┐
              │  Ring Light (하단용)  │
              │  ┌────────────────┐  │
              │  │ 검증 카메라     │  │
              │  │ (하단 설치)     │  │
              │  └────────────────┘  │
              └──────────────────────┘
```

**시스템 구성**:

| 구성요소 | 사양 (권장) | 용도 |
|----------|------------|------|
| 카메라 | Basler ace2 (동일 라인업) | 하단에서 picker 촬영 |
| 조명 | Ring Light 또는 Backlight | 부품 윤곽 강조 |
| 마운트 | Picker 이동 경로 하단 고정 | Picking → 조립 사이 경유 |

**검증 알고리즘**:

```
1. Picker가 부품을 picking한 후 검증 포인트로 이동
2. 하단 카메라로 이미지 캡처
3. 검증 처리:
   a. Picker 중심 좌표 (기계적 고정값) 대비 부품 중심 오프셋 계산
   b. 부품 회전 각도 오차 계산
   c. 허용 오차 범위 판정:
      - 위치 오차: ±Δx, ±Δy (mm)
      - 각도 오차: ±Δθ (degree)
4. 판정 결과:
   - PASS: 조립 공정 진행
   - OFFSET: 오프셋 보정값 전달 → 로봇이 보정 후 조립
   - FAIL: Re-picking 또는 부품 폐기
```

**예상 아키텍처 변경**:

```
신규 모듈:
  picking_verifier.py
    - PickingVerifier 클래스
    - verify_picking(image) → (status, offset_x, offset_y, angle_error)

camera_module.py 수정:
  - CameraController를 멀티 카메라 지원으로 확장
  - 또는 별도의 VerificationCameraController 인스턴스 생성

vision_server.py 수정:
  - 신규 명령어 추가: #VERIFY; → 검증 실행
  - 응답: #VERIFY_OK,offset_x,offset_y,angle_err;
         #VERIFY_FAIL,reason;

main_ui.py 수정:
  - Main Monitor에 검증 카메라 이미지 영역 추가
  - 검증 결과 표시 (PASS/OFFSET/FAIL)
```

**통신 프로토콜 확장**:

| 명령어 | 방향 | 용도 |
|--------|------|------|
| `#VERIFY;` | Client → Server | Picking 검증 요청 |
| `#VERIFY_OK,dx,dy,da;` | Server → Client | 검증 통과 (보정값 포함) |
| `#VERIFY_NG,reason;` | Server → Client | 검증 실패 |

---

### 11.3 조립 완제품 Crack/불량 검사

#### 배경

현재 시스템은 **조립 전 개별 부품**(Bush)의 위치/방향 검출에 특화되어 있다. 상/하단 커버까지 조립이 완료된 **완제품의 표면 불량**(Crack, 스크래치, 이물질 등) 검사는 별도 시스템이 필요하다.

#### 검토 항목

```
┌─────────────────────────────────────────────────────────────┐
│                    완제품 불량 유형                           │
├─────────────────┬───────────────────────────────────────────┤
│ Crack (균열)    │ 조립 압력에 의한 미세 균열, 사출 불량       │
│ Scratch (긁힘)  │ 이송/조립 과정 중 표면 손상                │
│ Burr (버)       │ 사출 잔여물, 조립 면 돌출                  │
│ Gap (틈새)      │ 상/하단 커버 결합 불량, 들뜸                │
│ 이물질          │ 표면 오염, 먼지, 유분                      │
│ 변색/얼룩       │ 사출 온도 불균일에 의한 색상 차이            │
└─────────────────┴───────────────────────────────────────────┘
```

#### 권장 접근 방식: Anomaly Detection

완제품 불량은 **정상 샘플 대비 이상 패턴**을 검출하는 방식이 적합하다. 불량 유형이 다양하고 불량 샘플 수집이 어렵기 때문이다.

**후보 기술**:

| 기술 | 장점 | 단점 | 적합도 |
|------|------|------|--------|
| **AnomalyDetection (PatchCore, EfficientAD 등)** | 정상 샘플만으로 학습 가능, 불량 유형 사전 정의 불필요 | 미세 결함 민감도 튜닝 필요 | 높음 |
| **Segmentation (U-Net 등)** | 불량 영역 정밀 마킹 가능 | 충분한 불량 라벨 데이터 필요 | 중간 |
| **Rule 기반 (Canny + Hough)** | 구현 간단, 추가 학습 불필요 | Crack 유형별 규칙 작성 필요, 유연성 낮음 | 낮음 |

**Anomaly Detection 파이프라인**:

```
정상 완제품 이미지 수집 (N장)
         │
         ▼
  Feature Extraction (Pretrained CNN backbone)
         │
         ▼
  Memory Bank 구축 (PatchCore) 또는 Teacher-Student 학습 (EfficientAD)
         │
         ▼
  ┌─── 검사 시 ────────────────────────────┐
  │                                        │
  │  완제품 이미지 캡처                      │
  │       │                                │
  │       ▼                                │
  │  Feature 추출 → Memory Bank와 비교      │
  │       │                                │
  │       ▼                                │
  │  Anomaly Score Map 생성                 │
  │       │                                │
  │       ▼                                │
  │  Score > Threshold → 불량 검출 + 위치   │
  │  Score ≤ Threshold → 정상 판정          │
  └────────────────────────────────────────┘
```

**하드웨어 요구 사항**:

| 구성요소 | 권장 사양 | 비고 |
|----------|----------|------|
| 카메라 | 고해상도 (5MP+) Area Scan | 미세 Crack 검출을 위해 현재보다 높은 해상도 필요 |
| 조명 | 다방향 조명 (Dome Light 또는 Multi-angle) | 표면 결함은 조명 각도에 민감. 단일 조명으로는 검출률 한계 |
| 스테이지 | 회전 스테이지 또는 다면 촬영 지그 | 완제품 전면 검사를 위해 다각도 촬영 필요 |
| GPU | NVIDIA (추론용, 옵션) | OpenVINO CPU 추론도 가능하나 처리 속도 고려 |

**검토 필요 사항**:

| # | 항목 | 내용 |
|---|------|------|
| 1 | 검사 기준 정의 | 허용 Crack 크기(μm), 위치별 허용 기준 등 품질 스펙 확정 필요 |
| 2 | 조명 환경 설계 | Crack 유형(표면/관통/헤어라인)별 최적 조명 방식 PoC 필요 |
| 3 | Tact Time | 현재 검사 사이클(~2초) 대비 추가 검사 시간 허용 범위 확인 |
| 4 | 정상 샘플 확보 | Anomaly Detection 학습을 위한 정상 완제품 이미지 최소 200~500장 |
| 5 | 기존 시스템 통합 vs 독립 | 현 시스템에 모듈 추가 vs 별도 검사 스테이션 구축 결정 |
| 6 | 판정 기준 검증 | 과검출(False Positive) 비율 vs 미검출(False Negative) 비율의 트레이드오프 합의 |
