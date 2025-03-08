# AHA: A Vision-Language-Model for Detecting and Reasoning over Failures in Robotic Manipulation

*Precise failure reasoning and detection for robotic manipulation

[[Project Page](https://aha-vlm.github.io/)] [[Paper](https://aha-vlm.github.io/Aha_paper.pdf)] 

**AHA: A Vision-Language-Model for Detecting and Reasoning over Failures in Robotic Manipulation [ICLR 2025]** [[Paper](https://arxiv.org/abs/2410.00371)] <br>
[Jiafei Duan](https://duanjiafei.com), [Wilbert Pumacay](https://wpumacay.github.io), [Nishanth Kumar](https://nishanthjkumar.com/), [Yi Ru Wang](https://helen9975.github.io/), [Shulin Tian](https://shulin16.github.io/), [Wentao Yuan](https://wentaoyuan.github.io), [Ranjay Krishna](https://ranjaykrishna.com), [Dieter Fox](https://homes.cs.washington.edu/~fox/), [Ajay Mandlekar*](https://ai.stanford.edu/~amandlek/), [Yijie Guo*](https://research.nvidia.com/person/yijie-guo)

![Overview](aha-teaser.gif)

## Introduction
 AHA, an open-source VLM specifically designed to detect and reason about failures in robotic manipulation through natural language. Through failure reasoning, AHA can aid to improve performance for robotic manipulation systems that relys on VLM (such as Trust the PRoC3S, Manipulate-Anything, and Eureka). 

## Contents
- [Data Generation](#data-generation)
- [Visual Instruction Finetuning](#visual-instruction-finetuning)

## Data Generation

This contains the steps to setup and generate failure data via FailGen. 

#### 1. Environment

Open your terminal and clone the repository:

```bash
git clone https://github.com/yourusername/aha](https://github.com/NVlabs/AHA.git
cd AHA
conda env create -f environment.yml
conda create -n aha python=3.9
conda activate aha
pip install -r requirements.txt
```

#### 2. PyRep and Coppelia Simulator

Follow instructions from the official [PyRep](https://github.com/stepjam/PyRep) repo; reproduced here for convenience:

PyRep requires version **4.1** of CoppeliaSim. Download: 
- [Ubuntu 16.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://www.coppeliarobotics.com/previousVersions#)

Once you have downloaded CoppeliaSim, you can pull PyRep from git:

```bash
cd <install_dir>
git clone https://github.com/stepjam/PyRep.git
cd PyRep
```

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

```bash
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

**Warning**: CoppeliaSim might cause conflicts with ROS workspaces. 

Finally install the python library:

```bash
pip install -r requirements.txt
pip install .
```

You should be good to go!
You could try running one of the examples in the *examples/* folder.

If you encounter errors, please use the [PyRep issue tracker](https://github.com/stepjam/PyRep/issues).

#### 3. RLBench

AHA uses my [RLBench fork](https://github.com/MohitShridhar/RLBench/tree/peract). 

```bash
cd <install_dir>
git clone -b peract https://github.com/MohitShridhar/RLBench.git # From Mohit's branch
python update.py
cd RLBench
pip install -r requirements.txt
python setup.py develop
```

#### 4. Generating Failure Demos via FailGen

```bash
cd aha/Data_Generation/rlbench-failgen
pip install -r requirements.txt
pip install -e .
```

Generate failure trajectories with only keyframes:

```bash
python /aha/Data_Generation/rlbench-failgen/examples/ex_custom_data_generator.py --num-episodes 1 --max-tries 3 --output-folder <OUTPUT DIR>
```
or 
```bash
xvfb-run -a -s "-screen 0 1400x900x24" python /aha/Data_Generation/rlbench-failgen/examples/ex_custom_data_generator.py --num-episodes 1 --max-tries 3 --output-folder <OUTPUT DIR> #For running on cluster or server
```

Generate failure trajectories with all frames:
```bash
python /aha/Data_Generation/rlbench-failgen/examples/ex_failgen_data_collection.py
```
or 
```bash
xvfb-run -a -s "-screen 0 1400x900x24" python /aha/Data_Generation/rlbench-failgen/examples/ex_failgen_data_collection.py
```


### Visual Instruction Finetuning

Visual instruction tuning takes around 40 hours for on 8 A100 GPUs with 80GB memory. We trained AHA in similar ways as RoboPoint (Even with the same data mix excluding the Pointing data). 

Here is the instruction for you perform visual instruction tuning on AHA generated failure dataset + Co-training data mix.

```bash
git clone https://github.com/wentaoyuan/RoboPoint.git
conda create -n robopoint python=3.10 -y
conda activate robopoint

pip install --upgrade pip  # enable PEP 660 support

# this is optional if you prefer to system built-in nvcc.
conda install -c nvidia cuda=12.1 -y

pip install -e .

# this is optional if you don't need to train the model
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

Merge the AHA failure dataset with the [Co-training data](https://huggingface.co/datasets/wentao-yuan/robopoint-data) and run the training scripts that can be found under `scripts` to perform full visual instruction finetuning.


## Acknowledgments
We would like to acknowledge the following projects where parts of the codes in this repo is derived from:
- [REFLECT][(https://github.com/wayveai/mile](https://github.com/real-stanford/reflect))
- [RLBench]([https://github.com/real-stanford/diffusion_policy](https://github.com/stepjam/RLBench))
- [RoboPoint](https://github.com/wentaoyuan/RoboPoint)

## Citation

If you find RoboPoint useful for your research and applications, please consider citing our paper:
```bibtex
@article{duan2024aha,
  title={AHA: A vision-language-model for detecting and reasoning over failures in robotic manipulation},
  author={Duan, Jiafei and Pumacay, Wilbert and Kumar, Nishanth and Wang, Yi Ru and Tian, Shulin and Yuan, Wentao and Krishna, Ranjay and Fox, Dieter and Mandlekar, Ajay and Guo, Yijie},
  journal={arXiv preprint arXiv:2410.00371},
  year={2024}
}
```
