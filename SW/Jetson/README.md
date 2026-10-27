# Jetson 통합 의류 정리 시스템

**옷개스트라 · 접신 | NVIDIA Jetson Orin Nano Super**

이 문서는 `SW/Jetson/`의 **유일한 README**입니다. 최종 통합 진입점인 `main/main.py`를 기준으로 의류 투입, 상·하의 분류, 전처리, 학습 기반 후처리(VLA), 접이판 동작까지 설명합니다. 기존 `preprocessing/`은 개발·제출 당시의 독립 실행 코드 보관 영역이며, 현재 통합 실행 진입점이 아닙니다.

> **검증 범위:** 팀의 Jetson/Docker 및 실제 하드웨어 구성에서 상의 1사이클과 하의 1사이클의 물리 실행을 각각 완료했습니다. 독립 Docker와 Fresh Clone 기준 통합 Self-test도 확인했습니다. 다른 Jetson 보드 또는 Docker 이미지에서 모든 TensorRT 엔진이 호환된다고 보장하지는 않습니다.

## 1. 시스템 구성과 전체 흐름

```text
V22 통합 UI · ELP OV2710 카메라 단일 소유
    │
    ├─ Mode A: 바구니 의류 파지 → 보드 위 배치 → 보드 준비 판정
    └─ Mode B: 보드 위 수동 배치
    │
    ▼
YOLO Segmentation + TOP/BOTTOM Classification
    │
    ├─ TOP
    │   ├─ upper_process.py: 상의 전처리 (91 mm 재중앙 정렬, 양팔 펼치기 등)
    │   └─ upper_vla_process.py: V51 상의 상태 판단·반복 조작·FINISH
    │
    └─ BOTTOM
        └─ bottom_process.py: V114 하의 전처리 및 하의 VLA 반복 조작·FINISH
    │
    ▼
양팔 대기 위치 확인 → Arduino 접이판 동작 → 사이클 종료
```

실제 통합 실행에서 상의·하의 선택은 분류 모델의 결과를 사용합니다. 분류 클래스는 `0=BOTTOM`, `1=TOP`이며 통합 컨트롤러가 이를 물리적 상·하의 경로로 매핑합니다. UI와 각 모듈은 공용 카메라 프레임 공급자를 이용하며, 하위 모듈이 `/dev/video0`을 중복 개방하지 않는 구조입니다.

## 2. 디렉터리 구조

```text
SW/Jetson/
├── README.md                        # 본 통합 안내서
├── main/
│   ├── main.py                      # 최종 실행 진입점
│   ├── V22_main/
│   │   ├── integration_runtime.py  # UI·분류·라우팅·접이판 통합 제어
│   │   ├── app_folding_board_test.py
│   │   ├── arduino_plate_controller.py
│   │   └── folding_machine.ui / *.qss / 이미지
│   ├── garment_transfer.py          # 바구니 → 보드 초기 이송
│   ├── upper_process.py             # 상의 전처리
│   ├── upper_vla_process.py         # 상의 VLA 공개 실행 래퍼
│   ├── upper_runtime_v51.py
│   ├── micro_unfold_e156_core_v51.py
│   ├── bottom_process.py            # 하의 전처리·VLA 라우팅
│   ├── bottom/                       # 하의 V114 실행에 필요한 소스
│   ├── shared_empty_board_state.py
│   ├── upper_basket_runtime.py
│   └── upper_board_runtime.py
├── models/
│   ├── classification/top_bottom/
│   ├── segmentation/
│   ├── pose/upper/ 및 pose/lower/
│   └── decision/upper/ 및 decision/lower/
├── common/
│   ├── camera/                       # ELP 카메라 보정
│   └── calibration/                  # 보드·바구니 좌표계 및 Homography
└── preprocessing/                    # 기존 독립 실행/제출 코드 (통합 메인과 구분)
```

`main.py`와 하위 모듈은 서로 의존하므로 특정 파일만 별도로 복사해 실행하는 방식은 권장하지 않습니다. 저장소의 `SW/Jetson/main`, `models`, `common` 전체를 함께 유지하세요.

## 3. 주요 모델 및 필수 데이터

| 역할 | 저장소 상대경로 |
|---|---|
| 상·하의 분류 | `models/classification/top_bottom/top_bottom_vla_finetune_mix5_e8_v1_best.pt` |
| 분류 체크포인트 사용자 정의 모듈 | `models/classification/top_bottom/train_top_bottom_cls_letterbox.py` |
| 의류 세그멘테이션 | `models/segmentation/kfashion_yolo26s_seg3_e100_best.engine` |
| 상의 포즈 | `models/pose/upper/tshirt_pose_yolo26m_synth_artf_board_v1_best.engine` |
| 하의 포즈 | `models/pose/lower/bottom_pose8_beige_finetune_v2_best.engine` |
| 상의 V51 판단 | `models/decision/upper/top_board_state_micro_v2/top_board_state_micro_v2_fp32.engine` |
| 상의 V51 상태 정규화 | `models/decision/upper/top_board_state_micro_v2/state_normalization.npz` |
| 하의 활성 판단(Robust30) | `models/decision/lower/robust30/robust30_aug_fp16.engine` |
| 하의 활성 판단 정규화 | `models/decision/lower/robust30/robust30_aug_normalization.npz` |

카메라 보정 `common/camera/elp_ov2710_1280x720_calibration.npz`, 바구니 보정 `common/calibration/basket_arm2_5point_affine.json`, 보드 설정 `common/calibration/dual_roarm_folding_board_config.json`, 통합 Homography `common/calibration/integrated_runtime_homography.json`도 필요합니다.

**외부 경로 주의:** 일부 레거시 소스와 하의 SHADOW 모델(V2/V6) 설정은 `/workspace/project_train/...` 경로를 기본값 또는 선택적 참조로 포함합니다. 실제 활성 하의 판단 모델은 저장소 내부 Robust30입니다. 통합 Self-test 성공만으로 모든 외부 경로가 모든 실행 경로에서 사용되지 않는다고 단정할 수 없으므로, 새로운 운영 환경으로 이전할 때 경로와 선택적 모델 사용 여부를 확인하세요. Homography의 `H`, `raw_H`, `camera_geometry` 및 보정 리소스를 임의로 합치거나 덮어쓰지 마세요.

## 4. 실행 환경

- 대상: NVIDIA Jetson Orin Nano Super, NVIDIA CUDA/TensorRT를 사용할 수 있는 검증된 Docker 환경
- 로봇: Waveshare RoArm-M2-S 2대 (`/dev/roarm_1` ARM1, `/dev/roarm_2` ARM2)
- 카메라: ELP OV2710, `/dev/video0`, 운용 해상도 1280 × 720
- 접이판: Arduino 기반 접이판 제어기
- 통신: USB 시리얼 및 공유 카메라 프레임

아래 명령은 팀에서 검증한 Docker 컨테이너 `a1`과 마운트 경로의 예시입니다. 다른 PC나 Jetson에서 실행할 때는 컨테이너 이름, 실제 소스 마운트 위치, 장치 접근 권한을 알맞게 설정해야 합니다.

### 비물리 통합 Self-test

```bash
sudo docker exec \
  -w /workspace/project_train/esw_final_validation/SW/Jetson/main \
  a1 python3 main.py --integration-self-test
```

성공 판정: `[MAIN-INTEGRATION-SELF-TEST] ALL_PASS=True` 및 `camera=NOT_OPENED robot=NOT_STARTED`.

### 실제 물리 자동 실행

```bash
sudo docker exec -it \
  -w /workspace/project_train/esw_final_validation/SW/Jetson/main \
  a1 python3 main.py \
    --routing-mode integrated \
    --physical \
    --roarm-send \
    --arduino-send \
    --plate-standby-safety
```

**안전:** 위 명령은 실제 로봇팔과 접이판을 구동합니다. 전원, 로봇팔 위치, 케이블, 작업 영역, 카메라 프레임, 비상 정지 수단을 확인한 상태에서만 실행하세요. 물리 실행 이전에는 반드시 비물리 Self-test를 통과시켜야 합니다.

### GitHub Fresh Clone 검사

```bash
git clone --branch final-jetson-integration --single-branch \
  https://github.com/ChoBeomGyu25/2026ESWContest_free_Otgaestra.git
```

새로 Clone한 저장소의 `SW/Jetson/`을 검증된 Jetson Docker 이미지에 읽기 전용으로 마운트하고 `main.py --integration-self-test`를 실행하여 외부 개발 소스 없이 필수 파일이 확인되는지 검사할 수 있습니다. 물리 장치가 없는 환경에서 실제 자동 실행 옵션을 사용하지 마세요.

## 5. 안전·상태 경계

- 바구니 이송은 보드 위 의류를 검증한 뒤 `[GARMENT-TRANSFER-BOARD-READY]` 상태를 반환합니다.
- 분류 결과에 따라 통합 컨트롤러가 `upper_process.py` 또는 `bottom_process.py`를 실행합니다.
- 상의 전처리는 FIX103 두 번째 중력 펼치기와 양팔 Standby까지 성공해야 V51 단계로 제어권을 넘깁니다.
- FIX136은 실패 후 그리퍼 개방·Standby 복구 및 제한된 재시도를 담당합니다. 안전 위치 검증 실패를 무시하고 다음 단계로 강제 진행하지 않습니다.
- 학습 기반 VLA에서 FINISH가 확인되고 필요한 로봇 대기 상태가 확보되면 접이판 단계로 진행합니다.
- 카메라는 V22 UI가 소유하며, 전처리·분류·VLA는 공유 프레임을 받습니다.

## 6. 검증 이력 및 제출 기준

- **상의 전체 사이클:** 바구니 이송 → TOP 분류 → 상의 전처리 → V51 → 접이판 → 종료까지 물리 실행 성공(1회 확인).
- **하의 전체 사이클:** 바구니 이송 → BOTTOM 분류 → 하의 처리·VLA → 접이판 → 종료까지 물리 실행 성공(1회 확인).
- **분류 모델:** 사용자 정의 학습 모듈 복원 후 Docker 모델 로딩·CPU 추론 성공.
- **통합 Self-test:** 기존 검증 Docker 및 독립 Docker에서 통과. GitHub Fresh Clone 기반 검증도 완료.
- **제출 브랜치:** `final-jetson-integration`.
- **작업 원칙:** 물리 검증한 로봇 동작 알고리즘은 임의 변경하지 않고, 제출 자료와 의존성의 일관성을 우선합니다.

`preprocessing/`의 과거 독립 실행 스크립트는 보존하되, **통합 운용의 공식 진입점은 `SW/Jetson/main/main.py` 하나**입니다.
