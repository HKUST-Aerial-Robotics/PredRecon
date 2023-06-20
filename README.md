# PredRecon

## News

* 20/06/2023: The code of Hierarchical Planner is available.
* 06/06/2023: The simulator (AirSim) is available.
* 10/02/2023: The code of Surface Prediction Module (SPM) is available.

## Introduction

**[ICRA'23]** This repository maintains the implementation of "PredRecon: A Prediction-boosted Planning Framework for Fast and High-quality Autonomous Aerial Reconstruction".

<div align=center><img src="imgs/sys_big.png" height=70% width=70% ></div>

**Paper**: [PrePrint_arXiv](https://arxiv.org/pdf/2302.04488.pdf)

**Complete video**: [Video](https://www.youtube.com/watch?v=ek7yY_FZYAc)

**Authors**: [Chen Feng](https://chen-albert-feng.github.io/AlbertFeng.github.io/), Haojia Li, [Fei Gao](http://zju-fast.com/fei-gao/), [Boyu Zhou](https://boyuzhou.net/), and [Shaojie Shen](https://uav.hkust.edu.hk/group/).

**Institutions**: [HKUST Aerial Robotics Group](https://uav.hkust.edu.hk/), [SYSU STAR Group](https://boyuzhou.net/), and [ZJU FASTLab](http://zju-fast.com/fei-gao/).

**PredRecon** is a prediction-boosted planning framework that can efficiently reconstruct high-quality 3D models for the target areas in unknown environments with a single flight. We obtain inspiration from humans can roughly infer the complete construction structure from partial observation. Hence, we devise a surface prediction module (SPM) to predict the coarse complete surfaces of the target from current partial reconstruction. Then, the uncovered surfaces are produced by online volumetric mapping waiting for the observation by UAV. Lastly, a hierarchical planner plans motions for 3D reconstruction, which sequentially find efficient global coverage paths, plans local paths for maximizing the performance of Multi-View Stereo (MVS) and generate smooth trajectories for image-pose pairs acquisition. We conduct benchmark in the realistic simulator, which validates the performance of PredRecon compared with classical and state-of-the-art methods.

<p align="center">
  <img src="imgs/palace_fly.gif" width = "400" height = "240"/>
  <img src="imgs/house_fly.gif" width = "400" height = "240"/>
  <img src="imgs/palace_recon.gif" width = "400" height = "240"/>
  <img src="imgs/house_recon.gif" width = "400" height = "240"/>
</p>

<div align=center><img src="imgs/bmk.png" height=50% width=50% ></div>

Please kindly star ⭐️ this project if it helps you. We take great efforts to develop and maintain it 😁.

## Installation

The project has been tested on Ubuntu 20.04 LTS (ROS Noetic). Directly clone our package (using ssh here):

```
  cd ${YOUR_WORKSPACE_PATH}/src
  git clone git@github.com:HKUST-Aerial-Robotics/PredRecon.git
```

Then install individual components of **PredRecon**:

* To install Surface Prediction Module, please follow the steps in [SPM](./SPM/README.md).
* To install Hierarchical Planner, please follow the steps in [Planner](./Planner/README.md).

## Quick Start

Open your Unreal Engine platform, then run AirSim simulator in the terminal: 
```
  source devel/setup.zsh && roslaunch airsim_ros_pkgs airsim_node.launch
```
Firstly, run ```Rviz``` for trajectory visualization:
```
  source devel/setup.zsh && roslaunch exploration_manager rviz.launch
```
Then, run the simulation:
```
  source devel/setup.zsh && roslaunch exploration_manager recon.launch
```
All images-pose pairs captured is stored in your given folder, and we recommend [RealityCapture](https://www.capturingreality.com/) as the 3D reconstruction platform.

## Acknowledgements

We use **NLopt** for non-linear optimization, use **LKH** for travelling salesman problem, and thank the source code of **mmdetection3d** and **FUEL**.
