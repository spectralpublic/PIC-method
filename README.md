
# Visible-Infrared Person Re-Identification via Partially Interactive Collaboration


### 1. Introduction

This is the reserch code of the IEEE Transactions on Image Processing (TIP) paper “Visible-Infrared Person Re-Identification via Partially Interactive Collaboration”.

[X. Zheng, X. Chen and X. Lu, "Visible-Infrared Person Re-Identification via Partially Interactive Collaboration," in IEEE Transactions on Image Processing, vol. 31, pp. 6951-6963, 2022.](https://ieeexplore.ieee.org/document/9935815)

Visible-infrared person re-identification (VI-ReID) task aims to retrieve the same person between visible and infrared images. VI-ReID is challenging as the images captured by different spectra present large cross-modality discrepancy. Many methods adopt a two-stream network and design additional constraint conditions to extract shared features for different modalities. However, the interaction between the feature extraction processes of different modalities is rarely considered. In this paper, a partially interactive collaboration method is proposed to exploit the complementary information of different modalities to reduce the modality gap for VI-ReID. Specifically, the proposed method is achieved in a partially interactive-shared architecture: collaborative shallow layers and shared deep layers. The collaborative shallow layers consider the interaction between modality-specific features of different modalities, encouraging the feature extraction processes of different modalities constrain each other to enhance feature representations. The shared deep layers further embed the modality-specific features to a common space to endow them the same identity discriminability. To ensure the interactive collaborative learning implement effectively, the conventional loss and collaborative loss are utilized jointly to train the whole network. Extensive experiments on two publicly available VI-ReID datasets verify the superiority of the proposed PIC method. Specifically, the proposed method achieves a rank-1 accuracy of 83.6% and 57.5% on RegDB and SYSU-MM01 datasets, respectively.

### 2. Start

  Train a model by
  ```bash
python train.py --dataset sysu
```

  - `--dataset`: select dataset between "sysu" or "regdb".

### 3. Referneces

[1] Ye M , Shen J , Lin G , et al. Deep Learning for Person Re-identification: A Survey and Outlook[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021, PP(99):1-1.

[2] https://github.com/mangye16/Cross-Modal-Re-ID-baseline


### 4. Related work 

If you find the code and dataset useful in your research, please consider citing:
 
    @article{9935815,
    author={Zheng, Xiangtao and Chen, Xiumei and Lu, Xiaoqiang},
    journal={IEEE Transactions on Image Processing}, 
    title={Visible-Infrared Person Re-Identification via Partially Interactive Collaboration}, 
    year={2022},
    volume={31},
    pages={6951-6963},
    doi={10.1109/TIP.2022.3217697}
    }


