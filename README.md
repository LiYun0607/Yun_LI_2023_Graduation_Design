# Cognitive Radio for Automated Driving Systems

This project focuses on cognitive radio (CR) for automated driving systems that require stable and reliable wireless communications. It enhances the single policy multi-agent RL algorithm called SPMA-PPO and Transformers into Transformer-SPMA-PPO, demonstrating significant improvements over conventional methods.

## Introduction

Almost all relevant studies in vehicle-to-everything (V2X) communications have adopted overly simplistic models. More realistic models for V2X are complex and dynamically variant, making it challenging for off-policy-based reinforcement learning (RL) algorithms to find stable solutions.

This paper addresses these challenges by:
- Combining the CARLA simulator with the SUMO simulator to simulate both macroscopic and microscopic traffic scenarios.
- Using parallel computing techniques to speed up training.
- Applying on-policy RL algorithms like proximal policy optimization (PPO) to find stable solutions to complex optimization problems.

## Results

Transformer-SPMA-PPO demonstrates:
- 53.8%, 11.7%, and 24.5% improvement in lost data over MA-DQN, SPMA-PPO, and CNN-SPMA-PPO, respectively.
- 45.5%, 49.2%, and 53.9% improvement in policy variance over MA-DQN, SPMA-PPO, and CNN-SPMA-PPO, respectively.

## Hardware Specifications

| **Component**                                   | **Specification**          |
|-------------------------------------------------|----------------------------|
| Operating System                                | Ubuntu 20.04 LTS           |
| GPU                                             | 2x NVIDIA RTX A5000        |
| CUDA Cores per GPU                              | 8192                       |
| Memory per GPU                                  | 24GB GDDR6                 |
| Peak Single-Precision Performance per GPU       | 19.5 teraflops             |

## Software and Simulator Specifications

| **Software/Simulator**                          | **Version**                |
|-------------------------------------------------|----------------------------|
| PyTorch                                         | 2.0.1+cu117                |
| CARLA Simulator                                 | 0.9.12                     |
| SUMO Simulator                                  | 1.17.0                     |
| Number of parallel environments \(N_e\)         | 3                          |

## System Models and Experimental Results

(System model diagrams and experimental results can be found in the `images/` folder.)

## Usage

Follow the instructions in the respective Python files to execute the experiments:
- `carla_Transformer_mul.py`: Transformer-SPMA-PPO model
- `carla_CNN.py`: CNN-SPMA-PPO model
- `carla_PPO.py`: PPO model

## Contact

For any questions or suggestions, please contact [liyun0607@outlook.com](mailto:liyun0607@outlook.com).
