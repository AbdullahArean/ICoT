# [CVPR'25] Interleaved-Modal Chain-of-Thought

This repository contains the official implementation of **Interleaved-Modal Chain-of-Thought**, accepted at **CVPR 2025**.


## 🖥️ Introduction

Interleaved-modal Chain-of-Thought (ICoT) is a novel reasoning concept for VLMs that integrates both visual and textual information in a structured chain-of-thought manner.
Our approach enhances multi-modal understanding by interleaving visual and textual cues, leading to improved performance on various benchmarks.

## 🚀 Features

- **Multi-Modal Chain-of-Thought:** Interleaves textual and visual reasoning steps for better multi-modal understanding.

![示例图片](figs/icot.png)


- **Generalizable Architecture:** Plug-and-play and applicable to different VLMs.
  
![示例图片](figs/ads.png)

## 📦 Installation

### Setup

 ```bash
# Clone the repository
git clone https://github.com/jungao1106/ICoT.git
cd ICoT
mv icot/processing_chameleon.py path/to/your/environments/transformers/models/chameleon/
mv icot/modeling_chameleon.py path/to/your/environments/transformers/models/chameleon/


## 🔥 Usage
### Data Preparation

Download and preprocess datasets in `data/`:
1. [M^3CoT](https://huggingface.co/datasets/LightChen2333/M3CoT)
2. [ScienceQA](http://scienceqa.github.io)
3. [LLaVA-W](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild)

After all datasets are downloaded, update the path in run.py

### Inference
```
bash run.sh
```



## 📜 Citation

If you find our work useful, please consider citing:

```bibtex
@article{gao2024interleaved,
  title={Interleaved-modal chain-of-thought},
  author={Gao, Jun and Li, Yongqi and Cao, Ziqiang and Li, Wenjie},
  journal={arXiv preprint arXiv:2411.19488},
  year={2024}
}
```

