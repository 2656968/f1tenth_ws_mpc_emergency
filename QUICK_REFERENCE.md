# F1TENTH 우승팀 워크스페이스 빠른 참조 (Quick Reference)

## 🏆 Top 3 추천 워크스페이스

### 1위: kohonda/proj-svg_mppi ⭐⭐⭐⭐⭐
- **GitHub**: https://github.com/kohonda/proj-svg_mppi
- **Stars**: 141 ⭐
- **난이도**: 고급
- **알고리즘**: SVG-MPPI (Stein Variational Guided MPPI)
- **언어**: C++
- **ROS**: Noetic/ROS2
- **추천 대상**: 최고 성능을 원하는 경험 많은 팀
- **특징**: ICRA 2024 논문, 고속 기동 최적화, Docker 제공

### 2위: zzjun725/f1tenth-racing-stack-ICRA22 ⭐⭐⭐⭐
- **GitHub**: https://github.com/zzjun725/f1tenth-racing-stack-ICRA22
- **Stars**: 24 ⭐
- **난이도**: 초급~중급
- **알고리즘**: Pure Pursuit + Lane Switching
- **언어**: Python
- **ROS**: ROS1/ROS2
- **추천 대상**: 시작하는 팀, 안정적인 구현 필요
- **특징**: ICRA 2022 실전 검증, TUM Raceline 사용, 명확한 문서

### 3위: smitdumore/f110-mpc ⭐⭐⭐
- **GitHub**: https://github.com/smitdumore/f110-mpc
- **Stars**: 15 ⭐
- **난이도**: 중급
- **알고리즘**: MPC (Model Predictive Control)
- **언어**: C++
- **ROS**: ROS1
- **추천 대상**: MPC를 배우고 싶은 팀
- **특징**: F1TENTH 시뮬레이터 통합, 명확한 구현

---

## 📊 알고리즘별 최고 구현

| 알고리즘 | 저장소 | Stars | 난이도 |
|---------|-------|-------|--------|
| SVG-MPPI | kohonda/proj-svg_mppi | 141 | ★★★★★ |
| Pure Pursuit | zzjun725/f1tenth-racing-stack-ICRA22 | 24 | ★★☆☆☆ |
| MPC | smitdumore/f110-mpc | 15 | ★★★☆☆ |
| MPPI | bosky2001/f1tenth_mppi | 11 | ★★★★☆ |
| MPCC | 22902716/MPCC | 3 | ★★★★☆ |

---

## 🎯 당신의 팀에 맞는 선택

### 처음 시작하는 팀
→ **zzjun725/f1tenth-racing-stack-ICRA22**
- 이해하기 쉬움
- 실전 검증됨
- 좋은 문서

### MPC 경험이 있는 팀
→ **smitdumore/f110-mpc**
- MPC 기초 학습
- 시뮬레이터 통합
- C++ 구현

### 최고 성능을 추구하는 팀
→ **kohonda/proj-svg_mppi**
- 최신 연구 기반
- 최고 성능
- 프로덕션 레벨

---

## 🚀 빠른 시작 가이드

### 1단계: Pure Pursuit로 시작 (1-2주)
```bash
git clone https://github.com/zzjun725/f1tenth-racing-stack-ICRA22
cd f1tenth-racing-stack-ICRA22
# README.md 따라하기
```

### 2단계: MPC 학습 (2-4주)
```bash
git clone https://github.com/smitdumore/f110-mpc
cd f110-mpc
# README.md 따라하기
```

### 3단계: MPPI로 고도화 (4-8주)
```bash
git clone https://github.com/kohonda/proj-svg_mppi
cd proj-svg_mppi
make docker_build
# Quick Start 가이드 따라하기
```

---

## 🔧 필수 도구

### TUM Global Raceline Optimization
- **GitHub**: https://github.com/TUMFTM/global_racetrajectory_optimization
- **용도**: 최적 레이스라인 생성
- **사용팀**: 대부분의 우승팀

### F1TENTH Gym Simulator
- **GitHub**: https://github.com/f1tenth/f1tenth_gym
- **용도**: 시뮬레이션 및 테스트
- **필수**: 모든 팀

---

## 📞 더 많은 정보

- **상세 문서 (한국어)**: [F1TENTH_WINNERS_WORKSPACES.md](F1TENTH_WINNERS_WORKSPACES.md)
- **Detailed Document (English)**: [F1TENTH_WINNERS_WORKSPACES_EN.md](F1TENTH_WINNERS_WORKSPACES_EN.md)

---

## 🎓 학습 순서 권장

```
1. Pure Pursuit 마스터 (2주)
   ↓
2. TUM Raceline 적용 (1주)
   ↓
3. MPC 기초 학습 (3주)
   ↓
4. Safety & Emergency 시스템 (2주)
   ↓
5. MPPI 고급 기법 (4주)
   ↓
6. SVG-MPPI 연구 (선택사항, 8주+)
```

**총 소요 시간**: 약 3개월 (기초부터 고급까지)

---

## ⚡ 핵심 메시지

1. **시작은 간단하게**: Pure Pursuit으로 시작
2. **안정성 우선**: Safety Node는 필수
3. **점진적 개선**: 한 번에 하나씩 추가
4. **실전 테스트**: 시뮬레이터 → 실차 순서로
5. **커뮤니티 활용**: 우승팀 코드를 적극 참고

---

*최종 업데이트: 2025-11-11*
*Team Echo, Chonnam National University*
