# F1TENTH Winners' Workspaces Research

This document provides a comprehensive overview of workspaces and technical stacks from top-performing teams in the F1TENTH autonomous racing competition.

## Major Competition Winners and Top Teams

### 1. ICRA 2022 F1TENTH Competition - Top Teams

#### 🏆 zzjun725/f1tenth-racing-stack-ICRA22
- **GitHub**: https://github.com/zzjun725/f1tenth-racing-stack-ICRA22
- **Stars**: 24 ⭐
- **Description**: Racing stack used in ICRA'22 F1TENTH competition (competed with ETH Zurich team)
- **Key Technologies**:
  - Pure Pursuit Controller
  - Lane Switcher (for overtaking and obstacle avoidance)
  - Opponent Predictor
  - TUM Global Raceline Optimization
  - Trajectory Generator

**Stack Structure**:
```
├── config/              # Configuration files
├── csv/                 # Trajectory data
├── dummy_car/           # Dummy car node
├── lane_follow/         # Lane following control
├── maps/                # Map files
├── opponent_predictor/  # Opponent prediction system
├── pure_pursuit/        # Pure Pursuit controller
├── scripts/             # Utility scripts
└── trajectory_generator/ # Trajectory generator
```

**Core Algorithms**:
- Optimal trajectory generation using TUM's Global Raceline Optimization
- Pure Pursuit-based path following
- Dynamic obstacle avoidance through opponent prediction
- Overtaking strategy via lane switching logic

---

### 2. ICRA 2024 - Stein Variational Guided MPPI

#### 🏆 kohonda/proj-svg_mppi
- **GitHub**: https://github.com/kohonda/proj-svg_mppi
- **Stars**: 141 ⭐⭐⭐ (Highly recommended!)
- **Paper**: [ICRA 2024] Stein Variational Guided Model Predictive Path Integral Control
- **Video**: https://www.youtube.com/watch?v=ML_aOYQIDL0
- **Description**: Advanced MPPI control technique for fast maneuvering vehicles

**Key Technologies**:
- **SVG-MPPI**: Stein Variational Guided Model Predictive Path Integral Control
- High-performance C++ implementation
- ROS Noetic/ROS2 support
- Docker environment provided

**Technical Features**:
- Faster convergence than traditional MPPI
- Optimized for high-speed maneuvering scenarios
- Sampling-based planning for nonlinear system control
- Real-time path planning and control

---

### 3. Other Competitive Workspaces

#### UC Irvine Racing Stack
- **GitHub**: https://github.com/uci-f1tenth/UCI_race_stack
- **Stars**: 7 ⭐
- **Description**: Autonomous racing stack for UC Irvine's F1TENTH team
- **Features**: Python-based, modern ROS2 architecture

#### Vanderbilt F1TENTH
- **GitHub**: https://github.com/verivital/F1TenthVanderbilt
- **Stars**: 28 ⭐
- **Description**: Vanderbilt University's F1TENTH competition code
- **Features**: C++ implementation, proven stability

---

## Control Algorithm Classification

### 1. MPC (Model Predictive Control) Based

#### Recommended Repositories:
1. **mlab-upenn/ISP2021-mpc_stack**
   - https://github.com/mlab-upenn/ISP2021-mpc_stack
   - UPenn's MPC stack
   - Stars: 9 ⭐

2. **smitdumore/f110-mpc**
   - https://github.com/smitdumore/f110-mpc
   - MPC for F1TENTH simulator
   - Stars: 15 ⭐

3. **JayCeeON/f1tenth_controllers**
   - https://github.com/JayCeeON/f1tenth_controllers
   - Includes both Pure Pursuit and MPC
   - Stars: 4 ⭐

### 2. MPPI (Model Predictive Path Integral) Based

#### Recommended Repositories:
1. **kohonda/proj-svg_mppi** (detailed above)
   - Most advanced technique
   - ICRA 2024 paper-based
   - Stars: 141 ⭐⭐⭐

2. **bosky2001/f1tenth_mppi**
   - https://github.com/bosky2001/f1tenth_mppi
   - Basic MPPI implementation
   - Stars: 11 ⭐

3. **vaithak/f1tenth_shield_mppi**
   - https://github.com/vaithak/f1tenth_shield_mppi
   - MPPI combined with Control Barrier Functions
   - Stars: 4 ⭐

### 3. Pure Pursuit Based

#### Recommended Repositories:
- **zzjun725/f1tenth-racing-stack-ICRA22** (detailed above)
  - Field-tested Pure Pursuit implementation
  - Includes lane switching logic

---

## IROS Competition Teams

### IROS 2020
- **BhooshanDeshpande/IROS2020-TeamiCart**
  - https://github.com/BhooshanDeshpande/IROS2020-TeamiCart
  - Stars: 5 ⭐
  - Online obstacle avoidance and head-to-head racing

### IROS 2021
- **zygn/IROS2021_ICE**
  - https://github.com/zygn/IROS2021_ICE
  - IROS 2021 F1TENTH competition repository

---

## Technology-Based Selection Guide

### 🎯 Recommended for Beginners
1. **Start with Pure Pursuit**: zzjun725/f1tenth-racing-stack-ICRA22
   - Easy-to-understand algorithm
   - Field-tested code
   - Detailed documentation

### 🚀 Recommended for Intermediate
1. **MPC Implementation**: smitdumore/f110-mpc or mlab-upenn/ISP2021-mpc_stack
   - Learn basics of predictive control
   - Understand optimization techniques

### 🏆 Advanced/Competitive Edge
1. **SVG-MPPI**: kohonda/proj-svg_mppi
   - Latest research-based
   - Best performance
   - ICRA 2024 paper implementation

---

## Common Technical Stack Components

Technologies commonly used by winning teams:

### 1. Trajectory Optimization
- **TUM Global Raceline Optimization**
  - https://github.com/TUMFTM/global_racetrajectory_optimization
  - Optimal raceline calculation
  - Used by many winning teams

### 2. Localization
- **Particle Filter**
  - AMCL (Adaptive Monte Carlo Localization)
  - Used by most teams

### 3. Simulator
- **F1TENTH Gym**
  - Standard simulator
  - Official competition environment

### 4. ROS Version
- **ROS Noetic** (most teams)
- **ROS2 Humble** (newer teams)

---

## Recommendations for Our Team

Current repository (`f1tenth_ws_mpc_emergency`) appears to be developing MPC/Emergency systems.

### Recommended Learning Path:

1. **Short-term (1-2 weeks)**:
   - Analyze zzjun725's Pure Pursuit code
   - Understand lane switching logic
   - Learn trajectory generation techniques

2. **Mid-term (1-2 months)**:
   - Learn basic MPC implementation (mlab-upenn or smitdumore)
   - Integrate emergency braking system
   - Apply TUM Raceline Optimization

3. **Long-term (3+ months)**:
   - Learn MPPI or SVG-MPPI
   - Apply advanced optimization techniques
   - Real vehicle testing and tuning

### Priority Implementation Recommendations:

1. **Safety Node** ✅ (already exists)
2. **Particle Filter** ✅ (already exists)
3. **Pure Pursuit Controller** (reference zzjun725)
4. **Trajectory Generator** (TUM-based)
5. **MPC Controller** (progressive enhancement)

---

## Additional Resources

### Official F1TENTH Resources
- **Official Website**: https://f1tenth.org/
- **Official GitHub**: https://github.com/f1tenth
- **Simulator**: https://github.com/f1tenth/f1tenth_gym
- **Docker Agent Template**: https://github.com/f1tenth/f1tenth_docker_agent

### Competition Results and Logs
- **ICRA 2022 Competition Logs**: https://github.com/f1tenth/icra_2022_logs

### Learning Materials
- F1TENTH Course: http://f1tenth.org/learn.html
- F1TENTH online courses and lab materials

---

## Quick Reference Table

| Repository | Stars | Algorithm | Language | Best For |
|-----------|-------|-----------|----------|----------|
| kohonda/proj-svg_mppi | 141 | SVG-MPPI | C++ | Advanced racing, ICRA 2024 |
| zzjun725/f1tenth-racing-stack-ICRA22 | 24 | Pure Pursuit | Python | Beginners, proven results |
| verivital/F1TenthVanderbilt | 28 | Various | C++ | General competition |
| smitdumore/f110-mpc | 15 | MPC | C++ | Learning MPC basics |
| bosky2001/f1tenth_mppi | 11 | MPPI | Python | MPPI introduction |
| mlab-upenn/ISP2021-mpc_stack | 9 | MPC | Python | MPC development |
| uci-f1tenth/UCI_race_stack | 7 | Various | Python | Modern ROS2 stack |

---

## Conclusion

To succeed in F1TENTH competition:

1. **Solid Foundation**: Start with proven algorithms like Pure Pursuit
2. **Progressive Improvement**: MPC → MPPI → SVG-MPPI progression
3. **Optimal Trajectory**: Utilize TUM Raceline Optimization
4. **Safety First**: Safety Node and Emergency system are essential
5. **Real Testing**: Sufficient testing in both simulator and real vehicle

The most important thing is to **build a stable working system first**, then add advanced features on top.

---

*Last Updated: 2025-11-11*
*Team: Echo, Chonnam National University*
