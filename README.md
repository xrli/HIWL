# Galaxy Image Classification

This repo contains the code for our paper *Galaxy Image Classification using Hierarchical Data Learning with Weighted Sampling and Label Smoothing*.

### Requirements

- numpy==1.19.5
- matplotlib==3.2.1
- tqdm==4.56.0
- torch>=1.7.1
- torchvision>=0.8.2

### Select cleandata

- Selecting clean data from the original data downloaded from https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge/data.
```
select cleandata.py
```

### Noscheme
- Training the models without HIWL
```
train_dieleman.py
train_vgg.py
train_googlenet.py
train_resnet26.py
train_resnet.py
train_efficientnet.py
train_vit.py

```


### Scheme
- Training the models with HIWL
```
train_dieleman.py
train_vgg.py
train_googlenet.py
train_resnet26.py
train_resnet.py
train_efficientnet.py
train_vit.py

```



### Usage

- Training a new model:
1. Selecting cleandata
2. training new model from scheme or noscheme
  ```shell
  python select cleandata.py
  python train_efficientnet.py
  ```

### Citation

- If you found this code useful please cite our paper: 
- 

