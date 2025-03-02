# [CVPR'25] Interleaved-Modal Chain-of-Thought

This repository contains the official implementation of **Interleaved-Modal Chain-of-Thought**, accepted at **CVPR 2025**.

## 📝 Paper

**Title:** Interleaved-Modal Chain-of-Thought\
**Conference:** CVPR 2025\
**Authors:** [Jun Gao], [Yongqi Li], ...\
[📄 Paper Link](#) (To be updated)

## 🖥️ Introduction

Interleaved-Modal Chain-of-Thought (IM-CoT) is a novel reasoning framework that integrates multiple modalities in a structured chain-of-thought manner. Our approach enhances multi-modal understanding by interleaving visual and textual cues, leading to improved performance on various benchmarks.

## 🚀 Features

- **Multi-Modal Chain-of-Thought:** Interleaves textual and visual reasoning steps for better multi-modal understanding.
- **Generalizable Architecture:** Applicable to different multi-modal tasks including VQA, image captioning, and vision-language reasoning.
- **State-of-the-Art Performance:** Achieves competitive results on multiple benchmarks.

## 📦 Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.10
- CUDA (if using GPU)
- Other dependencies in `requirements.txt`

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/IM-CoT.git
cd IM-CoT

# Install dependencies
pip install -r requirements.txt
```

## 🔥 Usage

### Data Preparation

Download and preprocess datasets following the instructions in `data/README.md`.

### Training

```bash
python train.py --config configs/config.yaml
```

### Evaluation

```bash
python evaluate.py --checkpoint path/to/checkpoint.pth
```

## 📊 Results

Our method achieves the following results on key benchmarks:

| Dataset | Accuracy | Improvement |
| ------- | -------- | ----------- |
| VQA v2  | XX%      | +X.X%       |
| NLVR2   | XX%      | +X.X%       |

## 📜 Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{yourpaper2025,
  title={Interleaved-Modal Chain-of-Thought},
  author={Your Name and Co-author Name and Others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

## 🤝 Acknowledgements

This research was supported by [Funding Source]. We also thank [Collaborators] for valuable discussions.

---

For any questions or issues, please open an issue or contact us via email.

📌 **GitHub Repository:** [https://github.com/yourusername/IM-CoT](https://github.com/yourusername/IM-CoT)

