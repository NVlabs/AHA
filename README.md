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
- [Training](#train)
- [Evaluation](#evaluation)

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


### Training

Visual instruction tuning takes around 40 hours for on 8 A100 GPUs with 80GB memory. We trained AHA in similar ways as RoboPoint (Even with the same data mix excluding the Pointing data). 

Here is the instruction for you perform visual instruction tuning on AHA generated failure dataset

Training scripts can be found under `scripts`.

If you are do not have enough GPU memory, you can reduce `BATCH_PER_GPU` and increase the `GRAD_ACC_STEPS` accordingly. Always keep the global batch size the same: `NUM_NODES` x `NUM_GPUS` x `BATCH_PER_GPU` x `GRAD_ACC_STEPS`.

Hyperparameters used in instruction tuning are provided below.

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| ---: | ---: | ---: | ---: | ---: | ---: |
| RoboPoint-v1-13B | 128 | 2e-5 | 1 | 2048 | 0 |

## Evaluation

Where2Place, a benchmark for spatial free-space reference on challenging real world images, can be found on HuggingFace at [wentao-yuan/where2place](https://huggingface.co/datasets/wentao-yuan/where2place).

To evaluate on Where2Place, first run the following command to generate results
```
python robopoint/eval/model_vqa.py \
    --model-path wentao-yuan/robopoint-v1-vicuna-v1.5-13b \
    --image-folder datasets/where2place/images \
    --question-file datasets/where2place/point_questions.jsonl \
    --answer-file output/robopoint-v1-vicuna-v1.5-13b.jsonl
```
Then, run the following command to compute the accuracy
```
python robopoint/eval/summarize_vqa.py --answer output/robopoint-v1-vicuna-v1.5-13b.jsonl
```
If needed, the following command visualizes the outputs of different models together with the ground truth
```
python robopoint/eval/visualize_vqa.py \
    --label gpt-4o robopoint \
    --answer output/gpt-4o.jsonl output/robopoint-v1-vicuna-v1.5-13b.jsonl \
    --output output/gpt-4o-vs-robopoint \
    --num 10
```

## Citation

If you find RoboPoint useful for your research and applications, please consider citing our paper:
```bibtex
@article{yuan2024robopoint,
  title={RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics},
  author={Yuan, Wentao and Duan, Jiafei and Blukis, Valts and Pumacay, Wilbert and Krishna, Ranjay and Murali, Adithyavairavan and Mousavian, Arsalan and Fox, Dieter},
  journal={arXiv preprint arXiv:2406.10721},
  year={2024}
}
```
