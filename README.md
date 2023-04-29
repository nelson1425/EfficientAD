# EfficientAD
Unofficial implementation of paper https://arxiv.org/abs/2303.14535

## Results

| Model         | Dataset    | Official Paper | efficientad.py |
|---------------|------------|----------------|----------------|
| EfficientAD-M | VisA       | 98.1           | pending        |
| EfficientAD-M | Mvtec LOCO | 90.7           | pending        |
| EfficientAD-M | Mvtec AD   | 99.1           | 99.1           |
| EfficientAD-S | VisA       | 97.5           | pending        |
| EfficientAD-S | Mvtec LOCO | 90.0           | pending        |
| EfficientAD-S | Mvtec AD   | 98.8           | 98.8           |


## Benchmarks

| Model         | GPU   | Official Paper | benchmark.py |
|---------------|-------|----------------|--------------|
| EfficientAD-M | A6000 | 4.5 ms         | 4.4 ms       |
| EfficientAD-M | A100  | -              | 4.6 ms       |
| EfficientAD-M | A5000 | 5.3 ms         | 5.3 ms       |


## Setup

### Packages

```
Python==3.10
torch==1.13.0
torchvision==0.14.0
tifffile==2021.7.30
tqdm==4.56.0
```

### Mvtec AD Dataset

For Mvtec evaluation code install:

```
numpy==1.18.5
Pillow==7.0.0
scipy==1.7.1
tabulate==0.8.7
tifffile==2021.7.30
tqdm==4.56.0
```

Download dataset:

```
mkdir mvtec_anomaly_detection
cd mvtec_anomaly_detection
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
tar -xvf mvtec_anomaly_detection.tar.xz
cd ..
```

Download evaluation code:

```
wget https://www.mydrive.ch/shares/60736/698155e0e6d0467c4ff6203b16a31dc9/download/439517473-1665667812/mvtec_ad_evaluation.tar.xz
tar -xvf mvtec_ad_evaluation.tar.xz
rm mvtec_ad_evaluation.tar.xz
```

### efficientad.py

Training requires ImageNet stored somewhere. Download ImageNet training images from https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data or set `--imagenet_train_path` of `efficientad.py` to other folder with general images in children folders.

Training and inference for Mvtec object:

```
python efficientad.py --dataset mvtec_ad --subdataset bottle
```

Evaluation with Mvtec evaluation code:

```
python mvtec_ad_evaluation/evaluate_experiment.py --dataset_base_dir './mvtec_anomaly_detection/' --anomaly_maps_dir './output/1/anomaly_maps/mvtec_ad/' --output_dir './output/1/metrics/mvtec_ad/' --evaluated_objects bottle
```
