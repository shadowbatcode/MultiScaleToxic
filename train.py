import numpy as np
import torch
import os
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
#/root/autodl-tmp/ToxMol/dataset/train_data/unimol_data.npy
train_data = np.load('/root/autodl-tmp/ToxMol/data_raw.npy',allow_pickle=True).item()


from unimol_tool import MolTrain
size = '570m'          ##['310m','84m','164m','570m','1.1B']
clf = MolTrain(
    task='regression', 
    early_stopping=5,
    learning_rate=3e-5 ,
    max_norm=1.0,
    split='hybrid',
    kfold=5,
    target_anomaly_check=False,
    model_name= 'toxmol',
    model_size=size,
    save_path=f'/root/autodl-tmp/ToxMol/exp/Test/{size}/',
    batch_size=6,
    comparative_learning = False,
    #fusion_location = 'after',
    freeze_layers= [],#'feature_fusion','classification_head'],
    freeze_layers_reversed = True
    )
clf.fit(train_data)

