# LASSO-MLP for stellar atmospheric parameters estimation

This repo contains the code for our paper *Galaxy Image Classification using Hierarchical Data Learning with Weighted Sampling and Label Smoothing*.

### Requirements

- numpy==1.19.5
- matplotlib==3.2.1
- tqdm==4.56.0
- torch>=1.7.1
- torchvision>=0.8.2




### Select cleandata

-Select clean data from the original data downloaded from /url:https://www.kaggle.com/competitions/1273 galaxy-zoo-the-galaxy-challenge/data.
```
select cleandata.py
```

-training label and test label：
```
LAMOST_APOGEE.csv
```


-estimation catalog
```
LASSO-MLP.csv
```



### Usage

- Training a new model:

  ```shell
  Jupyter Notebook LASSO_MLP.ipynb
  ```

### Citation

- If you found this code useful please cite our paper: 
- 

