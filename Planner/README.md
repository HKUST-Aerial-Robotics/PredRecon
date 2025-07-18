# Hierarchical Planner

<p align="center">
  <img src="imgs/global.png" width = "450" height = "300"/>
  <img src="imgs/local.png" width = "350" height = "300"/>
</p>
<p align="center">
    <strong>Left</strong>: Global Coverage Path Planning, <strong>Right</strong>: Quality-driven Local Path Planning
</p>

## Setup

* ROS Noetic (Ubuntu 20.04)
* NVIDIA RTX 3070Ti (Single GPU)
* CUDA 11.6
* cuDNN 8.9.0
* LibTorch 1.12.1-cu116
* PCL 1.7
* Eigen3
* gcc9

## Installation

* **[Pre-requisites]** Make sure 50GB space in your disk.

1. Install Unreal Engine

```
  git clone -b 4.25 git@github.com:EpicGames/UnrealEngine.git
  cd UnrealEngine
  ./Setup.sh
  ./GenerateProjectFiles.sh
  make
```

2. Install AirSim

```
  git clone https://github.com/Microsoft/AirSim.git
  cd AirSim
  ./setup.sh
  ./build.sh
```

  Download the [setting.json](https://github.com/HKUST-Aerial-Robotics/PredRecon/releases/tag/v1.0) and move it to ``~/Documents/AirSim/settings.json``.

3. Install cuDNN and LibTorch

```
  https://developer.nvidia.com/rdp/cudnn-download
  https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.12.0%2Bcu116.zip
```
You should change ```Torch_DIR``` in all ```CMakeLists.txt``` as your own LibTorch path, e.g. /home/albert/3rdparty/libtorch/share/cmake/Torch.

4. Download environments in AirSim and pre-trained model of SPM in C++ version

  [Palace Env](https://github.com/HKUST-Aerial-Robotics/PredRecon/releases/tag/v1.0)
  [Village Env](https://github.com/HKUST-Aerial-Robotics/PredRecon/releases/tag/v1.0) (Note: merge village_house_1.zip and village_house_2.zip into village house environment.)
  [spm.pt](https://github.com/HKUST-Aerial-Robotics/PredRecon/releases/tag/v1.0)

  **Note**: Put trained model into [address](https://github.com/HKUST-Aerial-Robotics/PredRecon/blob/master/Planner/Code/src/fuel_planner/exploration_manager/launch/algorithm.xml#L147) using your own file address.

5. Complie the planner

```
  cd ${YOUR_WORKSPACE_PATH}
  catkin_make -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

6. Change prediction model path

You should change ```surf_pred/model_``` in ```src/predrecon/exploration_manager/launch/algorithm.xml``` as your own spm checkpoint path.