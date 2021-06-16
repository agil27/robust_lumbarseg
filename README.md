# Lumbar segmentation with contrastive learning pretrain and localization transform

Author: Yuanbiao Wang, Han Huang, Xiang Li, Ruiyang Liu

This is the final project for Tsinghua University Course Pattern Recognition 2021 Fall.

This repository is written with the support of the high-performance auto differentiation library [Jittor](https://github.com/Jittor/jittor)

## Prequisities
- Data directory `data` (can be downloaded from Tsinghua cloud disk [coming soon])
- Environment
  - Linux
  - Anaconda
  - jittor (latest version)


## Usage
- Main Script
```bash
> python3.7 run.py [--model MODEL] [--mode MODE] [--gpu GPU] [-o {Adam, SGD}] [-e EPOCHS] [-b BATCH_SIZE] [-l LR] [--loss LOSS]
```

- If using hrnet, set the batch size to be 2

- Models:

  - unet, hrnet, setr

  - aug-*:  use random color space augmentation

  - stn-*:  add spatial transformer network(STN) module

  - ssl-*:  add contrastive learning pretrain

  - models available: `{stn_unet, ssl_unet, ssl_stn_unet, aug_unet, aug_ssl_stn_unet}`

- Modes:
  - `train`: train only
  - `test`: test only
  - `train-test`: train, validation, model selection & test on Xiehe dataset
  - `predcit`: inference and contour
  - `test_zs_big`, `test_zs_small`: test on Zhongshan dataset
  - `test_hard`: test and contour difficult samples selected from Xiehe dataset

- GPU
  - designate the GPU number, e.g. 0

- Loss
  - Choose from `{'ce', 'iou', 'dice', 'focal'}`
  - or combine with '_' and ratio, e.g. `'ce_0.8_dice_0.2'`

- Designate the GPU to be used, e.g. use #2 GPU：
  
  ```bash
  > CUDA_VISIBLE_DEVICES="2" python3.7 run.py --model hrnet --mode train-test -b 2 -e 8
  ```

## Pretrain
```bash
> log_silent=1 python run_ssl.py
```

## Demos
- Demo API: refer to [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/d6bbfa925b87400eb707/)
- Demo usage

  ```bash
  > export FLASK_APP='app.py'
  > python -m flask run -p [PORT]
  ```

## Pretrained weights
- Refer to the Tsinghua Cloud link given above
