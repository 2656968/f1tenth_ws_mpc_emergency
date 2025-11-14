# F1TENTH 우승자들의 워크스페이스 (F1TENTH Winners' Workspaces)

이 문서는 F1TENTH 자율주행 경주 대회에서 우수한 성적을 거둔 팀들의 워크스페이스와 기술 스택을 정리한 자료입니다.

## 주요 대회 우승팀 및 상위권 팀 워크스페이스

### 1. ICRA 2022 F1TENTH Competition - 상위권 팀

#### 🏆 zzjun725/f1tenth-racing-stack-ICRA22
- **GitHub**: https://github.com/zzjun725/f1tenth-racing-stack-ICRA22
- **Stars**: 24 ⭐
- **설명**: ICRA'22 F1TENTH 경기에서 사용된 레이싱 스택 (ETH Zurich 팀과 경쟁)
- **주요 기술**:
  - Pure Pursuit Controller
  - Lane Switcher (추월 및 장애물 회피용)
  - Opponent Predictor (상대방 예측)
  - TUM Global Raceline Optimization 사용
  - Trajectory Generator

**기술 스택 구조**:
```
├── config/              # 설정 파일
├── csv/                 # 궤적 데이터
├── dummy_car/           # 더미 차량 노드
├── lane_follow/         # 레인 팔로우 제어
├── maps/                # 맵 파일
├── opponent_predictor/  # 상대 예측 시스템
├── pure_pursuit/        # Pure Pursuit 컨트롤러
├── scripts/             # 유틸리티 스크립트
└── trajectory_generator/ # 궤적 생성기
```

**핵심 알고리즘**:
- TUM의 Global Raceline Optimization을 활용한 최적 궤적 생성
- Pure Pursuit 기반의 경로 추종
- 상대방 위치 예측을 통한 동적 장애물 회피
- 레인 전환 로직을 통한 추월 전략

---

### 2. ICRA 2024 - Stein Variational Guided MPPI

#### 🏆 kohonda/proj-svg_mppi
- **GitHub**: https://github.com/kohonda/proj-svg_mppi
- **Stars**: 141 ⭐⭐⭐
- **논문**: [ICRA 2024] Stein Variational Guided Model Predictive Path Integral Control
- **비디오**: https://www.youtube.com/watch?v=ML_aOYQIDL0
- **설명**: 고속 기동이 가능한 차량을 위한 고급 MPPI 제어 기법

**주요 기술**:
- **SVG-MPPI**: Stein Variational Guided Model Predictive Path Integral Control
- C++ 기반 고성능 구현
- ROS Noetic/ROS2 지원
- Docker 환경 제공

**기술적 특징**:
- 기존 MPPI보다 빠른 수렴 속도
- 고속 기동 시나리오에 최적화
- 샘플링 기반 계획으로 비선형 시스템 제어
- 실시간 경로 계획 및 제어

---

### 3. 기타 주요 경쟁력 있는 워크스페이스

#### UC Irvine Racing Stack
- **GitHub**: https://github.com/uci-f1tenth/UCI_race_stack
- **Stars**: 7 ⭐
- **설명**: UC Irvine F1TENTH 팀의 자율주행 레이싱 스택
- **특징**: Python 기반, 현대적인 ROS2 아키텍처

#### Vanderbilt F1TENTH
- **GitHub**: https://github.com/verivital/F1TenthVanderbilt
- **Stars**: 28 ⭐
- **설명**: Vanderbilt 대학의 F1TENTH 경기 코드
- **특징**: C++ 구현, 검증된 안정성

---

## 주요 제어 알고리즘 분류

### 1. MPC (Model Predictive Control) 기반

#### 추천 저장소:
1. **mlab-upenn/ISP2021-mpc_stack**
   - https://github.com/mlab-upenn/ISP2021-mpc_stack
   - UPenn의 MPC 스택
   - Stars: 9 ⭐

2. **smitdumore/f110-mpc**
   - https://github.com/smitdumore/f110-mpc
   - F1TENTH 시뮬레이터용 MPC
   - Stars: 15 ⭐

3. **JayCeeON/f1tenth_controllers**
   - https://github.com/JayCeeON/f1tenth_controllers
   - Pure Pursuit와 MPC 모두 포함
   - Stars: 4 ⭐

### 2. MPPI (Model Predictive Path Integral) 기반

#### 추천 저장소:
1. **kohonda/proj-svg_mppi** (위에서 상세 설명)
   - 가장 고급 기법
   - ICRA 2024 논문 기반
   - Stars: 141 ⭐⭐⭐

2. **bosky2001/f1tenth_mppi**
   - https://github.com/bosky2001/f1tenth_mppi
   - 기본적인 MPPI 구현
   - Stars: 11 ⭐

3. **vaithak/f1tenth_shield_mppi**
   - https://github.com/vaithak/f1tenth_shield_mppi
   - Control Barrier Functions와 결합된 MPPI
   - Stars: 4 ⭐

### 3. Pure Pursuit 기반

#### 추천 저장소:
- **zzjun725/f1tenth-racing-stack-ICRA22** (위에서 상세 설명)
  - 실전에서 검증된 Pure Pursuit 구현
  - 레인 전환 로직 포함

---

## IROS 대회 참가팀 워크스페이스

### IROS 2020
- **BhooshanDeshpande/IROS2020-TeamiCart**
  - https://github.com/BhooshanDeshpande/IROS2020-TeamiCart
  - Stars: 5 ⭐
  - 온라인 장애물 회피 및 헤드투헤드 레이싱

### IROS 2021
- **zygn/IROS2021_ICE**
  - https://github.com/zygn/IROS2021_ICE
  - IROS 2021 F1TENTH 경기 저장소

---

## 기술별 워크스페이스 선택 가이드

### 🎯 초보자에게 추천
1. **Pure Pursuit 시작**: zzjun725/f1tenth-racing-stack-ICRA22
   - 이해하기 쉬운 알고리즘
   - 실전 검증된 코드
   - 상세한 문서

### 🚀 중급자에게 추천
1. **MPC 구현**: smitdumore/f110-mpc 또는 mlab-upenn/ISP2021-mpc_stack
   - 예측 제어의 기초 학습
   - 최적화 기법 이해

### 🏆 고급자/경쟁력 극대화
1. **SVG-MPPI**: kohonda/proj-svg_mppi
   - 최신 연구 기반
   - 최고 성능
   - ICRA 2024 논문 구현

---

## 공통 기술 스택 요소

대부분의 우승팀들이 공통적으로 사용하는 기술:

### 1. 궤적 최적화
- **TUM Global Raceline Optimization**
  - https://github.com/TUMFTM/global_racetrajectory_optimization
  - 최적 레이스라인 계산
  - 많은 우승팀이 사용

### 2. 로컬라이제이션
- **Particle Filter**
  - AMCL (Adaptive Monte Carlo Localization)
  - 대부분의 팀이 사용

### 3. 시뮬레이터
- **F1TENTH Gym**
  - 표준 시뮬레이터
  - 대회 공식 환경

### 4. ROS 버전
- **ROS Noetic** (대부분)
- **ROS2 Humble** (최신 팀들)

---

## 우리 팀을 위한 권장사항

현재 저장소(`f1tenth_ws_mpc_emergency`)는 MPC/Emergency 시스템을 개발 중인 것으로 보입니다.

### 추천 학습 경로:

1. **단기 (1-2주)**:
   - zzjun725의 Pure Pursuit 코드 분석
   - 레인 전환 로직 이해
   - 궤적 생성 기법 학습

2. **중기 (1-2개월)**:
   - MPC 기본 구현 학습 (mlab-upenn 또는 smitdumore)
   - Emergency 브레이킹 시스템 통합
   - TUM Raceline Optimization 적용

3. **장기 (3개월+)**:
   - MPPI 또는 SVG-MPPI 학습
   - 고급 최적화 기법 적용
   - 실차 테스트 및 튜닝

### 우선 구현 추천:

1. **Safety Node** ✅ (이미 있음)
2. **Particle Filter** ✅ (이미 있음)
3. **Pure Pursuit Controller** (zzjun725 참고)
4. **Trajectory Generator** (TUM 기반)
5. **MPC Controller** (점진적으로 고도화)

---

## 추가 리소스

### 공식 F1TENTH 리소스
- **공식 웹사이트**: https://f1tenth.org/
- **공식 GitHub**: https://github.com/f1tenth
- **시뮬레이터**: https://github.com/f1tenth/f1tenth_gym
- **Docker Agent Template**: https://github.com/f1tenth/f1tenth_docker_agent

### 대회 결과 및 로그
- **ICRA 2022 Competition Logs**: https://github.com/f1tenth/icra_2022_logs

### 학습 자료
- F1TENTH Course: http://f1tenth.org/learn.html
- F1TENTH 온라인 강좌 및 Lab 자료

---

## 결론

F1TENTH 경쟁에서 성공하기 위해서는:

1. **견고한 기초**: Pure Pursuit와 같은 검증된 알고리즘으로 시작
2. **점진적 개선**: MPC → MPPI → SVG-MPPI 순으로 고도화
3. **최적 궤적**: TUM Raceline Optimization 활용
4. **안전 우선**: Safety Node와 Emergency 시스템 필수
5. **실전 테스트**: 시뮬레이터와 실차 모두에서 충분한 테스트

가장 중요한 것은 **안정적으로 동작하는 시스템**을 먼저 구축하고, 그 위에 고급 기능을 추가하는 것입니다.

---

*최종 업데이트: 2025-11-11*
*작성자: F1TENTH Team Echo (전남대학교)*
