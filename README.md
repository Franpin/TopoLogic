<div align="center">

# [NeurIPS 2024] TopoLogic: An Interpretable Pipeline for Lane Topology Reasoning on Driving Scenes
## 🔥New Topology Reasoning work [TopoPoint](https://github.com/Franpin/TopoPoint) is released.
[![NIPS](https://img.shields.io/badge/NeurIPS-2405.14747-479ee2.svg)](https://papers.nips.cc/paper_files/paper/2024/file/7116cda41d75d580bae15d9e484a8466-Paper-Conference.pdf)
[![TopoLogic](https://img.shields.io/badge/GitHub-TopoLogic-blueviolet.svg)](https://github.com/Franpin/TopoLogic)

![method](figs/pipeline.png "Model Architecture")


</div>

> - Production from [Institute of Computing Technology, Chinese Academy of Sciences](http://www.ict.ac.cn/). 
> - Primary contact: **Yanping Fu** ( fuyanping23s@ict.ac.cn ) or/and [Xinyuan Liu](https://scholar.google.cz/citations?user=eXwizz8AAAAJ&hl=zh-CN&oi=sra).


TL;DR
---
This repository contains the source code of **TopoLogic**, [An Interpretable Pipeline for Lane Topology Reasoning on Driving Scenes](https://papers.nips.cc/paper_files/paper/2024/file/7116cda41d75d580bae15d9e484a8466-Paper-Conference.pdf).

TopoLogic is the first to employ an interpretable approach for lane topology reasoning. TopoLogic fuses **the geometric distance of lane line endpoints** mapped through a designed function and **the similarity of lane query in a high-dimensional semantic space** to reason lane topology. Experiments on the large-scale autonomous driving dataset OpenLane-V2 benchmark demonstrate that TopoLogic significantly outperforms existing methods in topology reasoning in complex scenarios.


Updates
--- 
- [2025.5.26] 🔥New work [TopoPoint](https://github.com/Franpin/TopoPoint) is released.
- [2024.10.6] Code and Model are released.
- [2024.9.26] TopoLogic is accepted by NeurIPS 2024.
- [2024.5.23] TopoLogic paper is released at [arXiv](https://arxiv.org/abs/2405.14747).
## Table of Contents
- [TopoLogic: An Interpretable Pipeline for Lane Topology Reasoning on Driving Scenes](#topologic-an-interpretable-pipeline-for-lane-topology-reasoning-on-driving-scenes)
  - [Table of Contents](#table-of-contents)
  - [Model Zoo](#model-zoo)
  - [Main Results](#main-results)
    - [Results on OpenLane-V2 subset-A val](#results-on-openlane-v2-subset-a-val)
    - [Results on OpenLane-V2 subset-B val](#results-on-openlane-v2-subset-b-val)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Prepare Dataset](#prepare-dataset)
  - [Train and Evaluate](#train-and-evaluate)
    - [Train](#train)
    - [Evaluate](#evaluate)
  - [Citation](#citation)
  - [Related resources](#related-resources)

## Model Zoo


|    Method    | Backbone  | Epoch | Dataset | OLS |Version | Config | Download |  
| :----------: | :-------: | :---: | :-------------: | :--------------: | :-------------: | :--------------: | :------: |
| **TopoLogic**  | ResNet-50 |  24   |   subset-A | 44.1 | OpenLane-V2-v2.1.0 | [config](/projects/configs/topologic_r50_8x1_24e_olv2_subset_A.py) | [ckpt](https://huggingface.co/Franpin/topologic/resolve/main/topologic_r50_8x1_24e_olv2_subset_A.pth?download=true) / [log](https://huggingface.co/Franpin/topologic/resolve/main/topologic_r50_8x1_24e_olv2_subset_A.json?download=true) |

## Main Results
> The result is based on the `v1.0.0` OpenLane-V2 devkit and metrics. 
### Results on OpenLane-V2 subset-A val

We provide results on **[Openlane-V2](https://github.com/OpenDriveLab/OpenLane-V2) subset-A val** set.

|    Method    | Backbone | Epoch |SDMap | DET<sub>l</sub> | TOP<sub>ll</sub> | DET<sub>t</sub> | TOP<sub>lt</sub> |   OLS    |
| :----------: |----| :-------: | :---: | :-------------: | :--------------: | :-------------: | :--------------: | :------: |
|     STSU     | ResNet-50 |  24   |  × |     12.7       |       0.5        |      43.0       |       15.1       |   25.4   |
| VectorMapNet | ResNet-50 |  24   |  × |    11.1       |       0.4        |      41.7       |       6.2        |   20.8   |
|    MapTR     | ResNet-50 |  24   |  × |     8.3       |       0.2        |      43.5       |       5.8        |   20.0   |
|    MapTR*    | ResNet-50 |  24   | × |     17.7       |       1.1        |      43.5       |       10.4       |   26.0   |
| TopoNet  | ResNet-50 |  24   | × |   28.6     |     4.1      |    **48.6**     |    20.3     | 35.6 |
|**TopoLogic** | ResNet-50 | 24 | × |**29.9**| **18.6**  |47.2|**21.5** |**41.6**|
|SMERF     |ResNet-50 | 24  |√ |33.4 | 7.5 |**48.6**|23.4 |39.4| 15.4 |
|**TopoLogic** | ResNet-50 | 24 | √ |**34.4** |**23.4** |48.3|**24.4**| **45.1**|


> The result of TopoLogic is from this repo.


### Results on OpenLane-V2 subset-B val

|    Method    | Backbone  | Epoch | DET<sub>l</sub> | TOP<sub>ll</sub> | DET<sub>t</sub> | TOP<sub>lt</sub> |   OLS    |
| :----------: | :-------: | :---: | :-------------: | :--------------: | :-------------: | :--------------: | :------: |
| **TopoLogic**  | ResNet-50 |  24   |  **25.9** |**15.1**|**54.7** | **15.1**| **39.6**| **21.6** |

> The result is based on the updated `v2.1.0` OpenLane-V2 devkit and metrics.  
> The result of TopoLogic is from this repo.

|    Method    | Backbone  | Epoch | DET<sub>l</sub> | TOP<sub>ll</sub> | DET<sub>t</sub> | TOP<sub>lt</sub> |   OLS    |
| :----------: | :-------: | :---: | :-------------: | :--------------: | :-------------: | :--------------: | :------: |
| **TopoLogic**  | ResNet-50 |  24   |   **29.9** |**23.9** |**47.2** |**25.4** |**44.1**|


## Prerequisites

- Linux
- Python 3.8.x
- NVIDIA GPU + CUDA 11.1
- PyTorch 1.9.1

## Installation

We recommend using [conda](https://docs.conda.io/en/latest/miniconda.html) to run the code.
```bash
conda create -n topologic python=3.8 -y
conda activate topologic

# (optional) If you have CUDA installed on your computer, skip this step.
conda install cudatoolkit=11.1.1 -c conda-forge

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Install other required packages.
```bash
pip install -r requirements.txt
```

## Prepare Dataset

Following [OpenLane-V2 repo](https://github.com/OpenDriveLab/OpenLane-V2/blob/v1.0.0/data) to download the data and run the [preprocessing](https://github.com/OpenDriveLab/OpenLane-V2/tree/v1.0.0/data#preprocess) code.


## Train and Evaluate

### Train

We recommend using 8 GPUs for training. If a different number of GPUs is utilized, you can enhance performance by configuring the `--autoscale-lr` option. The training logs will be saved to `work_dirs/[work_dir_name]`.

```bash
cd TopoLogic
mkdir work_dirs

./tools/dist_train.sh 8 [work_dir_name] [--autoscale-lr]
```

### Evaluate
You can set `--show` to visualize the results.

```bash
./tools/dist_test.sh 8 [work_dir_name] [--show]
```



## Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.

``` bibtex
@inproceedings{fu2024topologic,
 author = {Fu, Yanping and Liao, Wenbin and Liu, Xinyuan and Xu, Hang and Ma, Yike and Zhang, Yucheng and Dai, Feng},
 booktitle = {Advances in Neural Information Processing Systems},
 pages = {61658--61676},
 title = {TopoLogic: An Interpretable  Pipeline for Lane Topology Reasoning on Driving Scenes},
 volume = {37},
 year = {2024}
}

@misc{fu2025topopoint,
      title={TopoPoint: Enhance Topology Reasoning via Endpoint Detection in Autonomous Driving}, 
      author={Yanping Fu and Xinyuan Liu and Tianyu Li and Yike Ma and Yucheng Zhang and Feng Dai},
      year={2025},
      eprint={2505.17771},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.17771}, 
}


```



## Related resources

We acknowledge all the open-source contributors for the following projects to make this work possible:
- [TopoPoint](https://github.com/Franpin/TopoPoint)
- [TopoNet](https://github.com/OpenDriveLab/TopoNet)
- [Openlane-V2](https://github.com/OpenDriveLab/OpenLane-V2)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
