# WaterSAM: Adapting SAM for Underwater Object Segmentation

This repository provides the official implementation of **WaterSAM**, a specialized adaptation of the Segment Anything Model (SAM) for underwater object segmentation.

## Introduction

Underwater object segmentation is challenging due to unique environmental complexities such as turbulence diffusion, light absorption, noise, and low contrast. To address these issues with limited labeled data, we propose **WaterSAM**. 

Inspired by Low-Rank Adaptation (LoRA), WaterSAM incorporates trainable rank decomposition matrices into the image encoder of the original SAM. This approach enhances feature extraction for underwater images while significantly reducing the number of trainable parameters to just 6.7% of the original SAM's parameters, thereby lowering computational costs.

## Quantitative Results

WaterSAM has been validated on three underwater datasets: COD10K, SUIM, and UIIS. The model significantly outperforms the pre-trained SAM (ViT-B) baseline across all tested metrics.

| Dataset | Metric | WaterSAM | SAM | Improve (%) |
| :--- | :--- | :--- | :--- | :--- |
| **COD10K** | Mean IoU Score | 84.96 | 16.98 | 400 |
| | Overall Accuracy | 96.09 | 29.47 | 226 |
| **SUIM** | Mean IoU Score | 90.47 | 38.41 | 136 |
| | Overall Accuracy | 98.08 | 61.98 | 58 |
| **UIIS** | Mean IoU Score | 84.24 | 46.64 | 81 |
| | Overall Accuracy | 93.16 | 64.58 | 44 |
| **UIIS*** | Mean IoU Score | 94.38 | 50.35 | 87 |
| | Overall Accuracy | 99.31 | 79.92 | 24 |
| **UIIS\*\*** | Mean IoU Score | 94.49 | - | - |
| | Overall Accuracy | 99.34 | - | - |

*Note: UIIS\*\* results were achieved using a LoRA rank of 128.*

## Platform Configuration

The experiments were conducted on the AutoDL platform with the following environment:
* Python: 3.8.19
* CUDA: 10.0
* GPU: NVIDIA L20 with 48 GB GPU memory
* OS: Ubuntu 18.04.6 LTS

## Directory Structure

The repository is structured as follows:
```text
WaterSAM/
├── SAM-Adapter-PyTorch-main/
│   ├── SAM-adapter/              # Main execution directory
│   │   ├── configs/              # SAM configuration files
│   │   │   └── adapter_sam.py    # Primary configuration script
│   │   ├── load/                 # Dataset storage
│   │   ├── models/               
│   │   │   ├── sam_lora.py       # LoRA implementation
│   │   │   └── adapter.py        # Main execution file
│   ├── weights/                  
│   │   └── sam_vit_b_sam_LoRA_best_score.pth # WaterSAM weights
│   ├── requirements.txt          # Dependency list
│   └── README_original.md        # Documentation for the original SAM-Adapter
├── .gitignore
└── README.md                     
```

## Installation

Install the required dependencies using the following command:
```bash
pip install -r SAM-Adapter-PyTorch-main/requirements.txt
```
*Note: Additional packages may need to be installed manually based on system prompts.*

## Usage

Navigate to the `SAM-adapter` directory:
```bash
cd SAM-Adapter-PyTorch-main/SAM-adapter/
```

### Training
1. In `configs/adapter_sam.py`, set `is_eval` to `False`.
2. In `adapters_weights_paths_dict`, set `vit_b` to `None`.
3. Configure the `Train dataset` path in `dataset_cod`.
4. Run the script:
```bash
python adapter.py
```

### Testing
1. In `configs/adapter_sam.py`, set `is_eval` to `True`.
2. Set `vit_b` in `adapters_weights_paths_dict` to the path of your trained weights.
3. Configure the `Test dataset` path in `dataset_cod_val`.
4. Run the script:
```bash
python adapter.py
```

**Implementation Notes:**
* Hyper-parameters such as learning rate, epoch size, and log paths can be adjusted in `configs/adapter_sam.py`.
* The default LoRA rank is set to `64` in `adapter.py`.

## Citation

If you find this work useful for your research, please cite:

```bibtex
@inproceedings{hong2024watersam,
  title={WaterSAM: Adapting SAM for Underwater Object Segmentation},
  author={Hong, Yang and Zhou, Xiaowei and Hua, Ruzhuang and Lyv, Qingxuan and Dong, Junyu},
  booktitle={Proceedings of Conference XX},
  year={2024},
  publisher={ACM}
}
```

## Acknowledgments and License

This project is released under the **MIT License**.

The core architecture is adapted from the `SAM-Adapter-PyTorch` project. We thank the original authors for their open-source contributions.

```text
Copyright (c) 2023 tianrun-chen (Original Code)
Copyright (c) 2024 Yang Hong (Adapter modifications)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
