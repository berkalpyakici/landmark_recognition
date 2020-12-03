import sys
import os
import cv2
import glob
import math
import pickle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import albumentations
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

import geffnet

from models import Swish_module, ArcFaceLossAdaptiveMargin, ArcMarginProduct_subcenter

device = torch.device('cuda')
batch_size = 4
num_workers = 4
out_dim = 302

data_dir = './Data/landmark-recognition-2020'
model_dir = './weights'
pickle_path = 'C:/Users/Mason/Downloads/Kaggle/idx2landmark_id.pkl'

transforms_64 = albumentations.Compose([
    albumentations.Resize(64, 64),
    albumentations.Normalize()
])

class LandmarkDataset(Dataset):
    def __init__(self, csv, split, mode, transform=transforms_64):

        self.csv = csv.reset_index()
        self.split = split
        self.mode = mode
        self.transform64 = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        
        image = cv2.imread(row.filepath)
        image = image[:, :, ::-1]
        
        res0 = self.transform64(image=image)
        image0 = res0['image'].astype(np.float32)
        image0 = image0.transpose(2, 0, 1)        
        
        return torch.tensor(image0)

def load_data(data_dir, model_dir):
    df = pd.read_csv(os.path.join(data_dir, 'train_subset_00.csv'))
    df['filepath'] = df['id'].apply(lambda x: os.path.join(data_dir, 'train', x[0], x[1], x[2], f'{x}.jpg'))
    df_sub = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

    df_test = df_sub[['id']].copy()
    df_test['filepath'] = df_test['id'].apply(lambda x: os.path.join(data_dir, 'test', x[0], x[1], x[2], f'{x}.jpg'))

    df = df[df.index % 10 == 0].iloc[500:1000].reset_index(drop=True)
    df_test = df_test.head(101).copy()

    dataset_query = LandmarkDataset(df, 'test', 'test')
    query_loader = torch.utils.data.DataLoader(dataset_query, batch_size=batch_size, num_workers=num_workers)

    dataset_test = LandmarkDataset(df_test, 'test', 'test')
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers)

    print(len(dataset_query), len(dataset_test))

    return query_loader, test_loader, df_test, df_sub, df

## MODEL LOADING

class enet_arcface_FINAL(nn.Module):
    def __init__(self, enet_type, out_dim):
        super(enet_arcface_FINAL, self).__init__()
        self.enet = geffnet.create_model(enet_type.replace('-', '_'), pretrained=None)
        self.feat = nn.Linear(self.enet.classifier.in_features, 512)
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)
        self.enet.classifier = nn.Identity()
 
    def forward(self, x):
        x = self.enet(x)
        x = self.swish(self.feat(x))
        return F.normalize(x), self.metric_classify(x)

def load_model(model, model_file):
    state_dict = torch.load(model_file)
    state_dict = state_dict["model_state_dict"]
    state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
    model.load_state_dict(state_dict, strict=True)

    #model.load_state_dict(state_dict)
    print(f"loaded {model_file}")
    model.eval()    
    return model

def get_model(out_dim):
    model_b7 = enet_arcface_FINAL('tf_efficientnet_b7_ns', out_dim=out_dim).to(device)
    return load_model(model_b7, './weights/b7ns_final_64_300w_f0_20ep_fold0.pth')

def get_pred_mask(path, df):
    with open(path, 'rb') as fp:
        idx2landmark_id = pickle.load(fp)
        landmark_id2idx = {idx2landmark_id[idx]: idx for idx in idx2landmark_id.keys()}
        
    return idx2landmark_id, pd.Series(df.landmark_id.unique()).map(landmark_id2idx).values

TOP_K = 5
CLS_TOP_K = 5

def val_epoch(model, valid_loader, criterion):

    model.eval()
    

def gen_preds(model, query_loader, test_loader, pred_mask, idx2landmark_id):
    with torch.no_grad():
        feats = []
        for i, img0 in enumerate(tqdm(query_loader)):
            img0 = img0.cuda()
            
            feat_b7, _ = model(img0)
            feats.append(feat_b7.detach().cpu())
        
        feats = torch.cat(feats)
        feats = feats.cuda()
        feat = F.normalize(feat_b7)


        val_loss = []
        PRODS_M = []
        PREDS_M = []
        TARGETS = []

        with torch.no_grad():
            for (data, target) in tqdm(test_loader):
                data, target = data.cuda(), target.cuda()

                logits_m = model(data)

                lmax_m = logits_m.max(1)
                probs_m = lmax_m.values
                preds_m = lmax_m.indices

                PRODS_M.append(probs_m.detach().cpu())
                PREDS_M.append(preds_m.detach().cpu())
                TARGETS.append(target.detach().cpu())

                loss = criterion(logits_m, target)
                val_loss.append(loss.detach().cpu().numpy())

            val_loss = np.mean(val_loss)
            PRODS_M = torch.cat(PRODS_M).numpy()
            PREDS_M = torch.cat(PREDS_M).numpy()
            TARGETS = torch.cat(TARGETS)

        acc_m = (PREDS_M == TARGETS.numpy()).mean() * 100.
        y_true = {idx: target if target >=0 else None for idx, target in enumerate(TARGETS)}
        y_pred_m = {idx: (pred_cls, conf) for idx, (pred_cls, conf) in enumerate(zip(PREDS_M, PRODS_M))}
        gap_m = global_average_precision_score(y_true, y_pred_m)
        return val_loss, acc_m, gap_m

        PRODS = []
        PREDS = []
        PRODS_M = []
        PREDS_M = []        
        for i, img0 in enumerate(tqdm(test_loader)):
            img0 = img0.cuda()
            
            probs_m = torch.zeros([4, out_dim], device=device)
            feat_b7, logits_m = model(img0); probs_m += logits_m
            feat = F.normalize(feat_b7)

            #probs_m[:, pred_mask] += 1.0
            #probs_m -= 1.0

            #print(probs_m.shape) 

            (values, indices) = torch.topk(probs_m, CLS_TOP_K, dim=1)
            probs_m = values
            preds_m = indices              
            PRODS_M.append(probs_m.detach().cpu())
            PREDS_M.append(preds_m.detach().cpu())            
            
            distance = feat.mm(feats.t())
            (values, indices) = torch.topk(distance, TOP_K, dim=1)
            probs = values
            preds = indices    
            PRODS.append(probs.detach().cpu())
            PREDS.append(preds.detach().cpu())

        PRODS = torch.cat(PRODS).numpy()
        PREDS = torch.cat(PREDS).numpy()
        PRODS_M = torch.cat(PRODS_M).numpy()
        PREDS_M = torch.cat(PREDS_M).numpy()

    gallery_landmark = df['landmark_id'].values
    PREDS = gallery_landmark[PREDS]
    PREDS_M = np.vectorize(idx2landmark_id.get)(PREDS_M)

    PRODS_F = []
    PREDS_F = []
    for i in tqdm(range(PREDS.shape[0])):
        tmp = {}
        classify_dict = {PREDS_M[i,j] : PRODS_M[i,j] for j in range(CLS_TOP_K)}
        for k in range(TOP_K):
            lid = PREDS[i, k]
            tmp[lid] = tmp.get(lid, 0.) + float(PRODS[i, k]) ** 9 * classify_dict.get(lid,1e-8)**10
        pred, conf = max(tmp.items(), key=lambda x: x[1])
        PREDS_F.append(pred)
        PRODS_F.append(conf)

    return PREDS_F, PRODS_F

if __name__ == '__main__':
    query_loader, test_loader, df_test, df_sub, df = load_data(data_dir, model_dir)
    model = get_model(out_dim)
    idx2landmark_id, pred_mask = get_pred_mask(pickle_path, df)

    print(pred_mask)

    PREDS_F, PRODS_F = gen_preds(model, query_loader, test_loader, pred_mask, idx2landmark_id)

    df_test['pred_id'] = PREDS_F
    df_test['pred_conf'] = PRODS_F

    df_sub['landmarks'] = df_test.apply(lambda row: f'{row["pred_id"]} {row["pred_conf"]}', axis=1)
    df_sub.to_csv('submission.csv', index=False)