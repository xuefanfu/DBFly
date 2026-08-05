<div align="center">


### Deliberate Before You Fly: Vision-Guided Spatial Deliberation for UAV See-and-Reach Navigation

**Fanfu Xue**, **En Yu**, **Bohang Liu**, **Hongjun Wang**, **Yang Yang**, **Xindi Wang**, **Jiande Sun**

[![Project Page](https://img.shields.io/badge/Project-Page-2f80ed?style=for-the-badge)](https://xuefanfu.github.io/DBFly-Page/)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/xuefanfu/DBFly)
[![Dataset](https://img.shields.io/badge/Dataset-UAV--VLN--FOV-00a67d?style=for-the-badge)](https://pan.baidu.com/s/1slWa79ZdNIHid_fwqyhdxA?pwd=ymav)
[![Simulator](https://img.shields.io/badge/Simulator-TravelUAV-f59e0b?style=for-the-badge)](https://huggingface.co/datasets/wangxiangyu0814/TravelUAV_env)
![Model Weights](https://img.shields.io/badge/Model_Weights-Coming_Soon-lightgrey?style=for-the-badge)
![Paper](https://img.shields.io/badge/Paper-Coming_Soon-lightgrey?style=for-the-badge)

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
├── weights/
│   └── DBFly/                     # Merged Hugging Face checkpoint
├── result/                        # Evaluation trajectories and logs
│   ├── test/
│   ├── unobject/
│   └── unscene/
├── log_files/                     # Runtime console logs
├── requirements.txt
└── README.md
```

The `dataset/`, `env_unzip/`, `weights/`, `result/`, and `log_files/` directories are local runtime resources and are not included in this repository.

## Pretrained Weights

The pretrained DBFly weights will be released after the paper-release process is completed.

> **Status:** Coming soon.

After downloading the checkpoint, place it under:

```text
DBFly/
└── weights/
    └── DBFly/
        ├── config.json
        ├── generation_config.json
        ├── preprocessor_config.json
        ├── tokenizer_config.json
        ├── model-*.safetensors
        └── ...
```

```
## UAV-VLN-FOV Dataset

DBFly is trained and evaluated on **UAV-VLN-FOV**, a high-resolution UAV see-and-reach navigation benchmark containing:

- 2,717 trajectories;
- concise language instructions;
- front-view and downward-view egocentric observations;
- continuous 3D waypoint annotations; and
- seen, unseen-object, and unseen-scene evaluation splits.

### Download

- **Baidu Cloud:** [Download UAV-VLN-FOV](https://pan.baidu.com/s/1slWa79ZdNIHid_fwqyhdxA?pwd=ymav)
- **Extraction code:** `ymav`
- **Dataset source:** [3DG-VLN repository](https://github.com/xuefanfu/3DG-VLN)

Organize the extracted dataset as follows:

```text
DBFly/
└── dataset/
    ├── train/
    ├── test/
    ├── unobject/
    └── unscene/
```

Only `test`, `unobject`, and `unscene` are required for evaluation. The `train` split is required only for training or additional analysis.

Large datasets may also be stored outside the repository and linked into the expected location:

```bash
ln -s /absolute/path/to/UAV-VLN-FOV/dataset ./dataset
```

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

Keep the downloaded directory structure unchanged. Ensure that the environment launch scripts are executable:

```bash
find env_unzip -type f -name "*.sh" -exec chmod +x {} \;
```

The default evaluation launcher starts `AirVLNSimulatorServerTool.py` automatically with:

```text
Simulator root: ./env_unzip
Simulator port: 30000
Auxiliary ports: 30001 and 30002
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
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu118

pip install numpy scipy pillow
```

Recommended platform:

- Linux;
- NVIDIA GPU with CUDA support;
- at least 24 GB GPU memory for FP16 inference;
- sufficient storage for the UAV-VLN-FOV dataset and simulation environments.

## Evaluation

### 1. Configure the evaluation script

Open `scripts/eval_DBFly.sh` and verify the following variables:

```bash
SIM_ROOT="$ROOT_DIR/env_unzip"
SIM_PORT=30000
MASTER_PORT=60001
GPU_ID=0
```

The repository currently contains a local absolute path in `--model_path`. Replace it with the downloaded checkpoint path. For the standard Test split, the core arguments should be:

```bash
--dataset_path "$ROOT_DIR/dataset/test" \
--eval_save_path "$ROOT_DIR/result/test" \
--model_path "$ROOT_DIR/weights/DBFly" \
--map_spawn_area_json_path "$ROOT_DIR/meta/map_spawnarea_info.json" \
--obj_desc_json_path "$ROOT_DIR/meta/instruction.json"
```

### 2. Run closed-loop evaluation

```bash
bash scripts/eval_DBFly.sh
```

The launcher:

1. terminates stale processes on ports `30000`, `30001`, and `30002`;
2. starts the AirSim simulator server;
3. launches DBFly closed-loop evaluation; and
4. restarts the simulator and evaluator after an abnormal exit.

Set `MAX_RETRY` in `scripts/eval_DBFly.sh` to control the number of automatic restarts. The default value `-1` enables unlimited retries. Press `Ctrl+C` to stop evaluation manually.

### 3. Evaluate different splits

Run the evaluator once for each split by changing `--dataset_path` and `--eval_save_path`:

| Evaluation split | Dataset path | Result path |
|:--|:--|:--|
| Test | `dataset/test` | `result/test` |
| Test UO | `dataset/unobject` | `result/unobject` |
| Test US | `dataset/unscene` | `result/unscene` |

The evaluator predicts a structured JSON decision and exactly five body-frame waypoints at each step. Predicted waypoints are transformed into the world frame and executed in closed loop.

### 4. Runtime logs

The evaluation program redirects console output to:

```text
log_files/<result-directory-name>.txt
```

For example, evaluation with `--eval_save_path result/test` writes runtime messages to:

```text
log_files/test.txt
```

## Metric Computation

The evaluation reports:

- **SR:** Success Rate;
- **OSR:** Oracle Success Rate;
- **NE:** Navigation Error in meters;
- **SPL:** Success weighted by Path Length.

To evaluate the Test split directly, run:

```bash
python utils/metric.py \
  --eval_save_path result/test \
  --eval_test_path dataset/test \
  --eval_unscene_path dataset/unscene \
  --eval_unobject_path dataset/unobject \
  --object_info_path meta/map_spawnarea_info.json
```

For Test UO or Test US, change only `--eval_save_path`:

```bash
# Test UO
--eval_save_path result/unobject

# Test US
--eval_save_path result/unscene
```

Alternatively, update `EVAL_SAVE_PATH` in `scripts/metric.sh` to a split-specific result directory and run:

```bash
bash scripts/metric.sh
```

## Acknowledgements

This repository builds upon or benefits from the following projects and resources:

- [UAV-VLN-FOV / 3DG-VLN](https://github.com/xuefanfu/3DG-VLN)
- [TravelUAV](https://github.com/prince687028/TravelUAV)
- [Microsoft AirSim](https://github.com/microsoft/AirSim)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL)

We sincerely thank the authors and maintainers of these projects.

## Citation

If DBFly is useful for your research, please cite the DBFly paper. The following entry is provisional and will be updated when the official publication information becomes available:

```bibtex
@misc{xue2026dbfly,
  title        = {Deliberate Before You Fly: Vision-Guided Spatial Deliberation for UAV See-and-Reach Navigation},
  author       = {Fanfu Xue and En Yu and Bohang Liu and Hongjun Wang and Yang Yang and Xindi Wang and Jiande Sun},
  year         = {2026},
  note         = {Manuscript under review}
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

## License

No license file is currently included in this repository. Please contact the authors before reusing, redistributing, or commercially deploying the code.

## Contact

For questions, bug reports, or reproducibility issues, please open a GitHub issue in this repository.

<div align="center">

**DBFly — Deliberate before execution, navigate with structured spatial reasoning.**

</div>
