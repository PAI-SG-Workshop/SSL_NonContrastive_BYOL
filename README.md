# SSL_NonContrastive_BYOL

**BYOL self-supervised pre-training of ResNet-18 on FashionLetters36 - PAI-SG SSL**

Part of the *Physical AI* reference series of the **Statera Working Group**. One algorithm (non-contrastive, EMA teacher) of the
SSL comparison in [SSL_Wave1_FL36](https://github.com/Statera-Workshop/SSL_Wave1_FL36); semi-supervised follow-up in [SemiSL_Wave2_FL36](https://github.com/Statera-Workshop/SemiSL_Wave2_FL36).

## Method

Projector 512-4096(BN)-256 and predictor 256-4096(BN)-256, EMA target tau 0.99 -> 1 (cosine), LARS lr 1.0, wd 1e-5, asymmetric augmentation (solarize 0.2 on the second view).

**Common setup** (identical for the six SSL repositories; full protocol and the cross-method comparison:
[SSL_Wave1_FL36](https://github.com/Statera-Workshop/SSL_Wave1_FL36)). ResNet-18 with the ImageNet stem (= the CenterNet ResNet-18; `torchvision` keys = `CenterNet.net.*`),
pre-trained on the 4,096 FashionLetters36 train images without labels or boxes. Two 96 x 96 views per sample, each
a RandomResizedCrop of one 128-256 px region of a 608 x 608 image; no horizontal flip (letters are not
mirror-invariant); grayscale input x / 255 as 3 channels, no mean / std; batch 256, warmup 10 epochs then cosine,
fp32, seed 777; 1 epoch = 32,768 samples. Recipes from the solo-learn CIFAR configs.

**Evaluation.** Representation: Crop36 (26,716 / 6,699 object crops, 36 classes), kNN (k 20) and linear probe
after re-estimating BN statistics on the train crops (labels unused). Detection: CenterNet on FL36 (36 classes),
the series protocol of [OD_Image_1-Stage_CenterNet](https://github.com/Statera-Workshop/OD_Image_1-Stage_CenterNet) v1.1 (Adam 1e-4, effective
batch 8, PreciseBN, last epoch, test mAP50); only the backbone initialisation differs.
From-scratch CenterNet reference: 30 epochs 0.611 / 0.609 (public baseline 0.644), 90 epochs 0.862 / 0.859.

## Results (FashionLetters36)

| Backbone | Crop36 kNN (BN re-est.) | Linear probe top-1 |
| --- | --- | --- |
| random init (reference) | 0.642 | 0.605 |
| 50 epochs, RTX 4070 | 0.561 | 0.769 |
| 50 epochs, GTX 1060 | 0.604 | 0.789 |
| 200 epochs | 0.702 | 0.835 |

**CenterNet, 100 % labels, 30 epochs (test mAP50):** 50 epochs: 0.566; 200 epochs: 0.590 (from scratch 0.611).

Weak on this data. 200 epochs improve the representation strongly (kNN +0.14) but the detector stays below from-scratch at the 30-epoch budget. BYOL is known to need long schedules (the solo-learn CIFAR recipe uses 1,000 epochs on 50k images); here the budget was 50 / 200 epochs on 4,096 images.

## Weights

Release `v1.0` (torchvision ResNet-18 keys, `fc` removed, SHA-256 in `SHA256SUMS.txt`):

- `ssl_byol_e50_backbone.pth`
- `ssl_byol_e200_backbone.pth`

```python
import torch, torchvision
m = torchvision.models.resnet18(weights=None); m.fc = torch.nn.Identity()
m.load_state_dict(torch.load("ssl_byol_e200_backbone.pth", weights_only=True))   # strict
# input: grayscale x / 255 repeated to 3 channels, no mean / std
```

## Run

```bash
export PHYSICAL_AI_ROOT=~/SSL_Wave1        # FL36 generated with OD_Image_1-Stage_CenterNet (data_processing_fl36.py)
python code/prep_ssl_data.py               # unlabeled image array + Crop36 (RESULT OK)
python code/ssl_train.py --method byol --epochs 200 --workers 8 --gate-margin -1 --knn-every 25
python code/final_eval.py                  # kNN / probe with re-estimated BN
# CenterNet: copy code/train_image_centernet_ssl.py and code/train_image_centernet_lab.py into OD_Image_1-Stage_CenterNet
python train_image_centernet_ssl.py --pretrained $PHYSICAL_AI_ROOT/runs/ssl_byol/backbone_e200.pth
```

`code/` holds the shared Wave 1 code (all SSL methods live in `ssl_train.py` / `ssl_train_ext.py`; this repository's
runs use `--method byol`). `results/` holds this method's training logs, its CenterNet runs and the
from-scratch baselines. Third-party code: [THIRD_PARTY.md](THIRD_PARTY.md) (solo-learn MIT notice in `licenses/`).
