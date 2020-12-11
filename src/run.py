import os
import time
import cv2
import pandas as pd
import numpy as np
import albumentations

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers, seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from models import *
from util import *

class Images(torch.utils.data.Dataset):
    def __init__(self, csv, args, transform = None, withlabel = True):
        self.csv = csv.reset_index()
        self.args = args
        self.transform = transform
        self.withlabel = withlabel
    
    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        if os.path.exists(row['filepath']):
            image = cv2.imread(row['filepath'])[:, :, ::-1]

            if self.transform is not None:
                res = self.transform(image = image)
                image = res['image'].astype(np.float32)
        else:
            append_to_log(self.args, 'Failed to locate image "' + row['filepath'] + '".', True)

            if self.withlabel:
                return torch.tensor(np.zeros(shape = (3, self.args.img_dim, self.args.img_dim), dtype=np.float32)), torch.tensor(0)
            else:
                return torch.tensor(np.zeros(shape = (3, self.args.img_dim, self.args.img_dim), dtype=np.float32))

        image = image.astype(np.float32).transpose(2, 0, 1)

        if self.withlabel: 
            return image, torch.tensor(row['landmark_id'])
        else:
            return image

class LandmarkDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.batch_size = args.batch_size

        self.train_transform = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ImageCompression(quality_lower=99, quality_upper=100),
            albumentations.Resize(args.img_dim, args.img_dim),
            albumentations.Normalize()
        ])

        self.val_transform = albumentations.Compose([
            albumentations.Resize(args.img_dim, args.img_dim),
            albumentations.Normalize()
        ])

        # Set of Images and Split Sets (Training, CV, Tets)
        self.images = None
        self.train_set = None
        self.cv_set = None
        self.test_set = None

        self.out_dim = 0

    def prepare_data(self):
        ########################################
        # DATA LOADING AND FILTERING
        ########################################
        df = pd.read_csv(os.path.join(self.args.img_dir, 'train.csv'))
        df['filepath'] = df['id'].apply(lambda x: os.path.join(self.args.img_dir, 'train', x[0], x[1], x[2], f'{x}.jpg'))

        # Get landmark ids that have more than min_landmark_count images.
        freq = pd.DataFrame(df['landmark_id'].value_counts())
        freq.reset_index(inplace = True)
        freq.columns = ['landmark_id', 'count']
        freq = freq[freq['count'] >= self.args.min_img_per_label]

        # Obtain filtered images.
        df_filtered = df.merge(freq, on=['landmark_id'], how='right')
        df_filtered.reset_index(inplace = True)

        landmark_id2idx = {landmark_id: idx for idx, landmark_id in enumerate(sorted(df_filtered['landmark_id'].unique()))}
        df_filtered['landmark_id'] = df_filtered['landmark_id'].map(landmark_id2idx)
        
        # Filter out old columns.
        self.images = df_filtered[['id', 'landmark_id', 'filepath']]

        # Set out_dim.
        self.out_dim = self.images['landmark_id'].nunique()

        append_to_log(self.args, f'Loaded {self.images.shape[0]} images with total {self.out_dim} unique labels.', True)

        ########################################
        # DATA SPLITTING
        ########################################
        train = 0.7
        cv = 0.2
        test = 0.1
        
        if train + cv + test >= 1:
            append_to_log(self.args, f'The requested size for train, cv, and test do not add up to 1. The ratios add up to {train + cv + test}.', True)
            return

        self.train_set, self.cv_set, self.test_set = np.split(self.images.sample(frac = 1.0, random_state = self.args.seed), [int(train * len(self.images)), int((train + cv) * len(self.images))])

        append_to_log(self.args, '', True)
        append_to_log(self.args, f'Training Set: {self.train_set.shape[0]} images.', True)
        append_to_log(self.args, f'CV Set: {self.cv_set.shape[0]} images.', True)
        append_to_log(self.args, f'Test Set: {self.test_set.shape[0]} images.', True)
        append_to_log(self.args, '', True)

        self.train_set.to_csv(os.path.join(self.args.log_dir, f'{self.args.name}-train_set.csv'), index=False)
        self.cv_set.to_csv(os.path.join(self.args.log_dir, f'{self.args.name}-cv_set.csv'), index=False)
        self.test_set.to_csv(os.path.join(self.args.log_dir, f'{self.args.name}-test_set.csv'), index=False)

    def setup(self, stage = None):
        if stage == "fit" or stage is None:
            self.train_img = Images(self.train_set, self.args, transform = self.train_transform, withlabel = True)
            self.cv_img = Images(self.cv_set, self.args, transform = self.val_transform, withlabel = True)

        if stage == "test" or stage is None:
            self.test_img = Images(self.test_set, self.args, transform = self.val_transform, withlabel = True)

    def train_dataloader(self):
        return DataLoader(self.train_img, batch_size = self.batch_size, num_workers = self.args.num_workers, drop_last = True)

    def val_dataloader(self):
        return DataLoader(self.cv_img, batch_size = self.batch_size, num_workers = self.args.num_workers, drop_last = True)

    def test_dataloader(self):
        return DataLoader(self.test_img, batch_size = self.batch_size, num_workers = self.args.num_workers, drop_last = True)
    
    def get_dim(self):
        return self.out_dim

class LandmarkClassifier(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.model = model
        self.args = args

        self.loss_fn = ArcFaceLoss()

        self.accuracy = pl.metrics.Accuracy()

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y) * 1.0

        return loss    
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)
        preds_conf, preds = torch.max(y_hat, 1)

        loss = self.loss_fn(y_hat, y) * 1.0
        acc = self.accuracy(y_hat, y)

        self.log('val_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        metrics = dict({
            'preds': preds,
            'preds_conf': preds_conf,
            'targets': y,
        })

        return metrics

    def validation_epoch_end(self, outputs):
        out_val = {}
        for key in outputs[0].keys():
                out_val[key] = torch.cat([o[key] for o in outputs])

        for key in out_val.keys():
            out_val[key] = out_val[key].detach().cpu().numpy().astype(np.float32)

        gap = global_average_precision_score(self.model.out_features, out_val["targets"], [out_val["preds"], out_val["preds_conf"]])

        self.log('val_mAP', gap, prog_bar=True, logger=True, sync_dist=True)
        append_to_log(self.args, time.ctime() + ' ' + f'Epoch {self.current_epoch}, Val mAP: {(gap):.6f}', False)
    
    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)
        preds_conf, preds = torch.max(y_hat.softmax(1),1)

        loss = self.loss_fn(y_hat, y) * 1.0
        acc = self.accuracy(y_hat, y)

        self.log('test_acc', acc, prog_bar=True, logger=True, sync_dist=True)

        metrics = dict({
                'preds': preds,
                'preds_conf':preds_conf,
                'targets': y
            })

        return metrics

    def test_epoch_end(self, outputs):
        out_val = {}
        for key in outputs[0].keys():
                out_val[key] = torch.cat([o[key] for o in outputs])

        for key in out_val.keys():
            out_val[key] = out_val[key].detach().cpu().numpy().astype(np.float32)

        gap = global_average_precision_score(self.model.out_features, out_val["targets"], [out_val["preds"], out_val["preds_conf"]])            

        df = pd.DataFrame(out_val).sort_values(by = ["preds_conf"], ascending = False)
        df = df[df['preds'] != df['targets']]

        df.to_csv(os.path.join(self.args.log_dir, f'{self.args.name}-test_results.csv'), index=True)

        self.log('test_mAP', gap, prog_bar=True, logger=True, sync_dist=True)
        append_to_log(self.args, time.ctime() + ' ' + f'Test Micro AP: {(gap):.6f}', True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr = self.args.lr, momentum = self.args.momentum, weight_decay = self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.args.epochs, verbose = True)

        return [optimizer], [scheduler]


if __name__ == '__main__':
    args = getargs()

    print("Running run.py...")
    print('\n'.join([key +': '+ str(vars(args)[key]) for key in vars(args).keys()]))
    print()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    seed_everything(args.seed)

    data_module = LandmarkDataModule(args)
    data_module.prepare_data()
    
    # Define Model
    effnet = EffnetLandmark(args, data_module.get_dim())
    model = LandmarkClassifier(args, effnet)
    model.hparams.batch_size = args.batch_size

    # Logger
    tb_logger = loggers.TensorBoardLogger(args.log_dir)

    # Model Saving Setup
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mAP',
        dirpath=args.model_dir,
        mode = 'max',
        filename=args.name + '-{epoch:02d}-{val_mAP:.4f}')

    if args.gpus > 1:
        accelerator = 'ddp'
    else:
        accelerator = None

    # Define trainer and fit to data
    trainer = Trainer(
        gpus=args.gpus, 
        logger = tb_logger, 
        accelerator=accelerator,
        #auto_scale_batch_size = 'binsearch',
        default_root_dir=args.model_dir, 
        callbacks=[checkpoint_callback], 
        max_epochs = args.epochs, 
        precision = 16,
        num_sanity_val_steps = 0 if args.mode == 'test' else 5,
        resume_from_checkpoint = args.checkpoint_path if args.mode == 'test' else None,
        progress_bar_refresh_rate = 5)

    trainer.fit(model, data_module)

    if args.mode == "train":
        trainer.test(datamodule = data_module)
    elif args.mode == "test":
        trainer.test(datamodule = data_module, ckpt_path=args.checkpoint_path)