<div align="center">


### Deliberate Before You Fly: Vision-Guided Spatial Deliberation for UAV See-and-Reach Navigation

**Fanfu Xue**, **En Yu**, **Bohang Liu**, **Hongjun Wang**, **Yang Yang**, **Xindi Wang**, **Jiande Sun**

[![Project Page](https://img.shields.io/badge/Project-Page-2f80ed?style=for-the-badge)](https://xuefanfu.github.io/DBFly-Page/)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/xuefanfu/DBFly)
[![Dataset](https://img.shields.io/badge/Dataset-UAV--VLN--FOV-00a67d?style=for-the-badge)](https://pan.baidu.com/s/1slWa79ZdNIHid_fwqyhdxA?pwd=ymav)
![Model](https://img.shields.io/badge/Model-Coming_Soon-lightgrey?style=for-the-badge)
[![Paper](https://img.shields.io/badge/Paper-DBFly-00a67d?style=for-the-badge)](https://arxiv.org/abs/2608.04825)

</div>

## Overview

**DBFly** (*Deliberate Before You Fly*) is a vision-language waypoint prediction framework for UAV **see-and-reach navigation**. Instead of directly mapping visual observations and language instructions to low-level waypoints, DBFly explicitly performs structured spatial deliberation before flight execution.

<p align="center">
  <img src="https://raw.githubusercontent.com/xuefanfu/DBFly-Page/main/assets/DBFly.png" width="96%" alt="DBFly framework">
</p>

## Repository Structure

After downloading the external resources, the recommended repository layout is:

```text
DBFly/
├── airsim_plugin/
│   ├── settings/                  # AirSim settings
│   ├── AirVLNSimulatorClientTool.py
│   └── AirVLNSimulatorServerTool.py
├── meta/
│   ├── instruction.json           # Navigation instructions
│   └── map_spawnarea_info.json    # Scene and target-position metadata
├── scripts/
│   ├── eval_DBFly.sh              # Closed-loop evaluation launcher
│   └── metric.sh                  # Metric computation launcher
├── src/
│   ├── common/
│   │   └── param.py               # Runtime arguments and shared configuration
│   └── vlnce_src/
│       ├── closeloop_util_nohelp.py
│       ├── env_uav.py
│       └── eval_IFC-VLN.py        # Main DBFly evaluation program
├── utils/
│   ├── env_utils_uav.py
│   ├── env_vector_uav.py
│   ├── logger.py
│   ├── metric.py                  # SR, OSR, NE, and SPL computation
│   └── utils.py
├── dataset/                       # Downloaded UAV-VLN-FOV data
│   ├── train/
│   ├── test/
│   ├── unobject/
│   └── unscene/
├── env_unzip/                     # Downloaded simulation environments
│   ├── carla_town_envs/
│   ├── closeloop_envs/
│   └── extra_envs/
├── model/
│   └── DBFly/                     # Merged Hugging Face checkpoint
├── result/                        # Evaluation trajectories and logs
│   ├── test/
│   ├── unobject/
│   └── unscene/
├── log_files/                     # Runtime console logs
├── requirements.txt
└── README.md
```

The `dataset/`, `env_unzip/`, `model/`, `result/`, and `log_files/` directories are local runtime resources and are not included in this repository.

## Pretrained Model

The pretrained DBFly model will be released after the paper-release process is completed.

> **Status:** DBFly (Coming Soon)

After downloading the checkpoint, place it under:

```text
DBFly/
└── model/
    └── DBFly/
        ├── config.json
        ├── generation_config.json
        ├── preprocessor_config.json
        ├── tokenizer_config.json
        ├── model-*.safetensors
        └── ...
```


## UAV-VLN-FOV Dataset

DBFly is trained and evaluated on **UAV-VLN-FOV**, a high-resolution UAV see-and-reach navigation benchmark containing:

- 2,717 trajectories;
- concise language instructions;
- front-view and downward-view egocentric observations;
- continuous 3D waypoint annotations; and
- seen, unseen-object, and unseen-scene evaluation splits.

### Download

- **Baidu Cloud:** [UAV-VLN-FOV](https://pan.baidu.com/s/1slWa79ZdNIHid_fwqyhdxA?pwd=ymav)
- **Source repository:** [3DG-VLN repository](https://github.com/xuefanfu/3DG-VLN)

Organize the extracted dataset as follows:

```text
DBFly/
└── dataset/
    ├── train/
    ├── test/
    ├── unobject/
    └── unscene/
```

Only `test`, `unobject`, and `unscene` are required for evaluation. The `train` split is required only for training.


## Simulator Environments

DBFly uses the Unreal Engine and AirSim simulation environments released by **TravelUAV**.

### Download

- **Hugging Face:** [TravelUAV Simulation Environments](https://huggingface.co/datasets/wangxiangyu0814/TravelUAV_env)
- **Source repository:** [TravelUAV](https://github.com/prince687028/TravelUAV)

Extract the environments into:

```text
DBFly/
└── env_unzip/
    ├── carla_town_envs/
    ├── closeloop_envs/
    └── extra_envs/
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/xuefanfu/DBFly.git
cd DBFly
```

### 2. Create a Conda environment

```bash
conda create -n dbfly python=3.10 -y
conda activate dbfly
```

### 3. Install dependencies

The released configuration uses PyTorch 2.4.0 with CUDA 11.8.

```bash
pip install -r requirements.txt 
```

## Evaluation

###  Run closed-loop evaluation

```bash
bash scripts/eval_DBFly.sh
```

## Metric Computation

The evaluation reports:

- **SR:** Success Rate;
- **OSR:** Oracle Success Rate;
- **NE:** Navigation Error in meters;
- **SPL:** Success weighted by Path Length.

```bash
bash scripts/metric.sh
```

## Acknowledgements

This repository builds upon or benefits from the following projects and resources:

- [UAV-VLN-FOV / 3DG-VLN](https://github.com/xuefanfu/3DG-VLN)
- [TravelUAV](https://github.com/prince687028/TravelUAV)

We sincerely thank the authors and maintainers of these projects.

## Citation

If DBFly is useful for your research, please cite the DBFly paper. The following entry is provisional and will be updated when the official publication information becomes available:

```bibtex
@misc{xue2026dbfly,
      title={Deliberate Before You Fly: Vision-Guided Spatial Deliberation for UAV See-and-Reach Navigation}, 
      author={Fanfu Xue and En Yu and Bohang Liu and Hongjun Wang and Yang Yang and Xindi Wang and Jiande Sun},
      year={2026},
      eprint={2608.04825},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2608.04825}, 
}
```

Please also cite the UAV-VLN-FOV benchmark:

```bibtex
@misc{xue2026seeandreach,
  title         = {See-and-Reach: Precise Vision-Language Navigation for UAVs within the Field of View},
  author        = {Fanfu Xue and En Yu and Yantian Shen and Zhikun Hu and Hongjun Wang and Yang Yang and Xindi Wang and Jiande Sun},
  year          = {2026},
  eprint        = {2606.20045},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2606.20045}
}
```

