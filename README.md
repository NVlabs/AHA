# 🤖 AHA: A Vision-Language-Model for Detecting and Reasoning over Failures in Robotic Manipulation

*Precise failure reasoning and detection for robotic manipulation*

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://aha-vlm.github.io/) 
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://aha-vlm.github.io/Aha_paper.pdf)

**AHA: A Vision-Language-Model for Detecting and Reasoning over Failures in Robotic Manipulation [ICLR 2025]** 
[[Paper](https://arxiv.org/abs/2410.00371)]

[Jiafei Duan](https://duanjiafei.com), [Wilbert Pumacay](https://wpumacay.github.io), [Nishanth Kumar](https://nishanthjkumar.com/), [Yi Ru Wang](https://helen9975.github.io/), [Shulin Tian](https://shulin16.github.io/), [Wentao Yuan](https://wentaoyuan.github.io), [Ranjay Krishna](https://ranjaykrishna.com), [Dieter Fox](https://homes.cs.washington.edu/~fox/), [Ajay Mandlekar*](https://ai.stanford.edu/~amandlek/), [Yijie Guo*](https://research.nvidia.com/person/yijie-guo)

![Overview](aha-teaser.gif)

## 📖 Introduction
AHA is an open-source VLM specifically designed to detect and reason about failures in robotic manipulation through natural language. Through failure reasoning, AHA can improve performance for robotic manipulation systems that rely on VLMs (such as Trust the PRoC3S, Manipulate-Anything, and Eureka).

## 📑 Contents
- [Data Generation](#data-generation)
- [Visual Instruction Finetuning](#visual-instruction-finetuning)

## 🛠️ Data Generation

### 1. Environment Setup

```bash
git clone https://github.com/NVlabs/AHA.git
conda create -n aha python=3.10 -y
conda activate aha

pip install --upgrade pip  # enable PEP 660 support

# this is optional if you prefer to system built-in nvcc.
conda install -c nvidia cuda=12.1 -y
```

### 2. PyRep and Coppelia Simulator

Download CoppeliaSim v4.1:
- [Ubuntu 16.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://www.coppeliarobotics.com/previousVersions#)

Install PyRep:

```bash
git clone https://github.com/stepjam/PyRep.git
cd PyRep
pip install -r requirements.txt
pip install .
```

Add to your `~/.bashrc`:

```bash
export COPPELIASIM_ROOT=/path/to/coppeliasim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Remember to run `source ~/.bashrc` after adding these lines.

> ⚠️ **Warning**: CoppeliaSim might cause conflicts with ROS workspaces.

### 3. RLBench

Install the fork:

```bash
git clone -b peract https://github.com/MohitShridhar/RLBench.git
python update.py
cd RLBench
pip install -r requirements.txt
pip install .
```

### 4. Generating Failure Demos via FailGen

```bash
cd aha/Data_Generation/rlbench-failgen
pip install -r requirements.txt
pip install -e .
```

Generate failure trajectories with keyframes only:

```bash
python /aha/Data_Generation/rlbench-failgen/examples/ex_custom_data_generator.py \
  --num-episodes 1 \
  --max-tries 3 \
  --output-folder <OUTPUT_DIR>
```

For headless servers:
```bash
xvfb-run -a -s "-screen 0 1400x900x24" \
  python /aha/Data_Generation/rlbench-failgen/examples/ex_custom_data_generator.py \
  --num-episodes 1 \
  --max-tries 3 \
  --output-folder <OUTPUT_DIR>
```

Generate failure trajectories with all frames:
```bash
python /aha/Data_Generation/rlbench-failgen/examples/ex_failgen_data_collection.py
```

## 🧠 Visual Instruction Finetuning

> Training takes ~40 hours on 8 A100 GPUs (80GB).
> 
> AHA is instruction finetuned with RoboPoint codebase, so setup finetuning code with RoboPoint.

Setup:

```bash
git clone https://github.com/wentaoyuan/RoboPoint.git
conda create -n robopoint python=3.10 -y
conda activate robopoint
pip install --upgrade pip  # Enable PEP 660 support

# Optional: if you prefer to use conda's CUDA
conda install -c nvidia cuda=12.1 -y

pip install -e .
pip install -e ".[train]"  # Only needed for training
pip install flash-attn --no-build-isolation
```

Merge the AHA failure dataset with the [Co-training data](https://huggingface.co/datasets/wentao-yuan/robopoint-data) and run the training scripts found under `scripts` folder.

## 🙏 Acknowledgments
We thank the following projects that parts of our code are derived from:
- [REFLECT](https://github.com/real-stanford/reflect)
- [RLBench](https://github.com/stepjam/RLBench)
- [RoboPoint](https://github.com/wentaoyuan/RoboPoint)

## 📝 Citation

```bibtex
@article{duan2024aha,
  title={AHA: A vision-language-model for detecting and reasoning over failures in robotic manipulation},
  author={Duan, Jiafei and Pumacay, Wilbert and Kumar, Nishanth and Wang, Yi Ru and Tian, Shulin and Yuan, Wentao and Krishna, Ranjay and Fox, Dieter and Mandlekar, Ajay and Guo, Yijie},
  journal={arXiv preprint arXiv:2410.00371},
  year={2024}
}
```
