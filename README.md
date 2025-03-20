# 🤖 AHA: A Vision-Language-Model for Detecting and Reasoning over Failures in Robotic Manipulation

*Precise failure reasoning and detection for robotic manipulation*

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://aha-vlm.github.io/) 
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://aha-vlm.github.io/Aha_paper.pdf)

**[ICLR 2025]** 
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
- [Ubuntu 16.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

Extract it somewhere in your system, and set the following environemnt variables
(add it to `.bashrc` to make changes last):

```bash
export COPPELIASIM_ROOT=/path/to/coppeliasim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Remember to run `source ~/.bashrc` after adding these lines.

> ⚠️ **Warning**: CoppeliaSim might cause conflicts with ROS workspaces.

Install PyRep:

```bash
git clone https://github.com/stepjam/PyRep.git
cd PyRep
pip install -r requirements.txt
pip install .
```

### 3. RLBench

Install the fork:

```bash
git clone -b peract https://github.com/MohitShridhar/RLBench.git
python update.py
cd RLBench
pip install -r requirements.txt
pip install -e .
```

### 4. FailGen

```bash
cd aha/Data_Generation/rlbench-failgen
pip install -r requirements.txt
pip install -e .
```

After installing the packages, the structure is now:
- **
  - **aha/**
    - ...
  - **PyRep/**
    - ...
  - **RLBench/**
    - ...
  - ....

### 5. Generate failure trajectories with keyframes only:
For specific tasks:

```bash
python ./aha/Data_Generation/rlbench-failgen/examples/ex_custom_data_generator.py \
  --task basketball_in_hoop \
  --episodes 1 \
  --max_tries 1 \
  --savepath <Output Dir>
```

For headless servers:
```bash
xvfb-run -a -s "-screen 0 1400x900x24" \
  python ./aha/Data_Generation/rlbench-failgen/examples/ex_custom_data_generator.py \
  --task basketball_in_hoop \
  --episodes 1 \
  --max_tries 1 \
  --savepath <Output Dir>
```

Generate all 80 tasks as in the paper:
```bash
bash ./aha/Data_Generation/rlbench-failgen/examples/ex_custom_data_generator.sh
```

After generating all of the tasks, you would need to run these to generate the json file for instruction fine-tuning. 

```bash
#Process the data generated into right format
python ./aha/Data_Generation/rlbench-failgen/process_data.py /path/to/input_folder /path/to/output_folder

#Format the processed data into json for finetuning.
python ./aha/Data_Generation/rlbench-failgen/make_json.py /path/to/processed_data --output ./aha_training.json
```

## 🧠 Visual Instruction Finetuning

> Training takes ~40 hours on 8 A100 GPUs (80GB).
> 
> AHA is instruction finetuned with RoboPoint codebase, so setup finetuning code with RoboPoint.

Setup:

```bash
git clone https://github.com/wentaoyuan/RoboPoint.git
pip install -e .
pip install -e ".[train]"  # Only needed for training
pip install flash-attn --no-build-isolation
```

Merge the AHA failure dataset generated previously with the [Co-training data](https://huggingface.co/datasets/wentao-yuan/robopoint-data).

### Download pretrained projector weights

We use pretrained projector weights from [LLaVA](https://github.com/haotian-liu/LLaVA). The projector is trained on image-text pairs from the 558K subset of the LAION-CC-SBU dataset with BLIP captions (see [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)). When using these projector weights, please make sure that the vision encoder and the projector type are set correctly.

For CLIP-L-336px vision encoder,
```
--vision_tower openai/clip-vit-large-patch14-336
```

For MLP-2x projector,
```
--mm_projector_type mlp2x_gelu
```

For Linear projector,
```
--mm_projector_type linear
```

| Base LLM | Vision Encoder | Projection | Pretrain Data | Download |
|----------|----------------|------------|---------------|----------|
| Vicuna-13B-v1.5 | CLIP-L-336px | MLP-2x | LCS-558K | [projector](https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5) |
| Vicuna-7B-v1.5 | CLIP-L-336px | MLP-2x | LCS-558K | [projector](https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5) |
| LLaMA-2-13B-Chat | CLIP-L-336px | Linear | LCS-558K | [projector](https://huggingface.co/liuhaotian/llava-336px-pretrain-llama-2-13b-chat) |
| LLaMA-2-7B-Chat | CLIP-L-336px | Linear | LCS-558K | [projector](https://huggingface.co/liuhaotian/llava-336px-pretrain-llama-2-7b-chat) |

### Training

If you are do not have enough GPU memory, you can reduce `BATCH_PER_GPU` and increase the `GRAD_ACC_STEPS` accordingly. Always keep the global batch size the same: `NUM_NODES` x `NUM_GPUS` x `BATCH_PER_GPU` x `GRAD_ACC_STEPS`.

Hyperparameters used in instruction tuning are provided below.

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| ---: | ---: | ---: | ---: | ---: | ---: |
| RoboPoint-v1-13B | 128 | 2e-5 | 1 | 2048 | 0 |

```
#For full finetuning of RoboPoint with AHA dataset via Vicuna 1.5
bash ./RoboPoint/scripts/finetune_vicuna.sh 
```

## Evaluation:

We evaluated **AHA** on three test datasets:  
- **AHA (test)**  
- **Maniskill FailGen data**  
- **REFLECT**

Below are the instructions to generate or obtain each dataset:

- ⚙️ **AHA (test):** Generate this dataset using the same dataset generation script, but with different tasks.
  ```bash
  bash ./aha/Data_Generation/rlbench-failgen/examples/ex_data_generator_eval.sh
  ```
- 📖 **Maniskill FailGen:** Follow the instructions [here](https://github.com/wpumacay/maniskill-failgen) to generate the dataset.
- 🔍 **REFLECT:** Sub-sample the REFLECT dataset from [this source](https://www.cs.columbia.edu/~liuzeyi/reflect_data/) and use our annotated JSON file for evaluation.

After evaluated your trained model with the respective datasets you can measure the ROGUE-L, LLM Fuzzy, or Binary Success results via these:

### LLM Fuzzy

```bash
python aha/evaluation/eval_metrics/LLM_fuzzy.py --gt_path /path/to/real_qa.json --res_path /path/to/your_results.json
```
### ROGUE-L

```bash
python aha/evaluation/eval_metrics/check_answer_ROGUE.py --data_path /path/to/out_qa.json --answers_path /path/to/aha_arnold_out_final_qa_failgen_answers.json --indx_num 11291
```

### Binary Success

```bash
python aha/evaluation/eval_metrics/check_answer_Yes_No.py --data_path /path/to/out_qa.json --answers_path /path/to/aha_fr_out_final_qa_failgen_answers.json --indx_num 11291
```

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
