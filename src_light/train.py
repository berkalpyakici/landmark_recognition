import os
import time
import cv2
import pandas as pd
import numpy as np
import albumentations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import pytorch_lightning as pl
from pytorch_lightning import loggers, seed_everything, Trainer
from pytorch_lightning import loggers
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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = image.astype(np.float32)

            if self.transform is not None:
                res = self.transform(image = image)
                image = res['image'].astype(np.float32)
        else:
            append_to_log(self.args, 'Failed to locate image "' + row['filepath'] + '".', True)

            if self.withlabel:
                return self.to_torch_tensor(np.zeros(shape = (3, self.args.img_dim, self.args.img_dim), dtype=np.int8)), self.to_torch_tensor(0)
            else:
                return self.to_torch_tensor(np.zeros(shape = (3, self.args.img_dim, self.args.img_dim), dtype=np.int8))
        
        image = image.astype(np.float32).transpose(2, 0, 1)

        if self.withlabel: 
            return image, torch.tensor(row['landmark_id'])
        else:
            return image

    def to_torch_tensor(self,img):
        return torch.from_numpy(img.transpose((2, 0, 1)))

class LandmarkData():
    def __init__(self, args):
        self.args = args

        # Set of Images and Split Sets (Training, CV, Tets)
        self.images = None
        self.train_set = None
        self.cv_set = None
        self.test_set = None

        self.out_dim = 0

        #self.set_seed(self.args.seed)
    
    def get_datasets(self):
        self.load_images()
        self.split_images()

        if self.train_set is None:
            print('Training set is not loaded.')
            return

        # Load Dataset
        train_dataset = Images(self.train_set, self.args, transform = make_transform_train(self.args.img_dim, self.args.img_dim), withlabel = True)
        valid_dataset = Images(self.cv_set, self.args, transform = make_transform_val(self.args.img_dim, self.args.img_dim), withlabel = True)
        test_dataset = Images(self.test_set, self.args, transform = make_transform_val(self.args.img_dim, self.args.img_dim), withlabel = True)

        return train_dataset, valid_dataset, test_dataset

    def load_images(self):
        # Load all images.
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
    
    def split_images(self, train = 0.7, cv = 0.2, test = 0.1):
        if self.images is None:
            append_to_log(self.args, 'Images are not loaded.', True)
            return
        
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

    def set_seed(self, seed=303):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

class LandmarkDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

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
        # Load all images.
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

        ### DATA SPLITTING
        if self.images is None:
            append_to_log(self.args, 'Images are not loaded.', True)
            return
            
        # Begin Data Splitting
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
            self.train = Images(self.train_set, self.args, transform = self.train_transform, withlabel = True)
            self.valid = Images(self.cv_set, self.args, transform = self.val_transform, withlabel = True)

            self.dims = tuple(self.train[0][0].shape)

        if stage == "test" or stage is None:
            self.test = Images(self.test_set, self.args, transform = self.val_transform, withlabel = True)
            self.dims = tuple(self.test[0][0].shape)        

    def train_dataloader(self):
        return DataLoader(self.train, batch_size = self.args.batch_size, num_workers = self.args.num_workers, drop_last = True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size = self.args.batch_size, num_workers = self.args.num_workers, drop_last = True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size = self.args.batch_size, num_workers = self.args.num_workers, drop_last = True)
    
    def get_dim(self):
        return self.out_dim

class LandmarkClassifier(pl.LightningModule):
    def __init__(self, args, model, out_dim):
        super().__init__()
        self.model = model
        self.args = args

        self.learning_rate = self.args.lr

        self.accuracy = pl.metrics.Accuracy()
        self.out_dim = out_dim

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)
        #loss = ArcFaceLoss()(y_hat, y)
        loss = nn.CrossEntropyLoss()(y_hat, y) * 1.0

        self.log('train_loss', loss, prog_bar = True)

        return loss    
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)

        preds_conf, preds = torch.max(y_hat.softmax(1),1)

        loss = nn.CrossEntropyLoss()(y_hat, y)

        acc = self.accuracy(y_hat, y)
        #gap = global_average_precision_score(y_hat, y)

        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)

        metrics = dict({
                'preds': preds,
                'preds_conf':preds_conf,
                'targets': y,
            })    

        return metrics

    def validation_epoch_end(self, outputs):
        out_val = {}
        for key in outputs[0].keys():
                out_val[key] = torch.cat([o[key] for o in outputs])

        for key in out_val.keys():
            out_val[key] = out_val[key].detach().cpu().numpy().astype(np.float32)

        val_score = global_average_precision_score(self.out_dim, out_val["targets"], [out_val["preds"], out_val["preds_conf"]])

        self.log('val_gap', val_score, prog_bar=True, logger=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)

        preds_conf, preds = torch.max(y_hat.softmax(1),1)

        loss = nn.CrossEntropyLoss()(y_hat, y)

        acc = self.accuracy(y_hat, y)
        #gap = global_average_precision_score(y_hat, y)

        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_acc', acc, prog_bar=True, logger=True)

        metrics = dict({
                'preds': preds,
                'preds_conf':preds_conf,
                'targets': y,
            })    

        return metrics

    def test_epoch_end(self, outputs):
        out_val = {}
        for key in outputs[0].keys():
                out_val[key] = torch.cat([o[key] for o in outputs])

        for key in out_val.keys():
            out_val[key] = out_val[key].detach().cpu().numpy().astype(np.float32)

        val_score = global_average_precision_score(self.out_dim, out_val["targets"], [out_val["preds"], out_val["preds_conf"]])

        self.log('test_gap', val_score, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr = self.args.lr, momentum = 0.9, weight_decay = 1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.args.epochs)

        return [optimizer], [scheduler]

if __name__ == '__main__':
    args = getargs()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    seed_everything(args.seed)

    # Load Dataset
    #data = LandmarkData(args)

    #train_dataset, valid_dataset, test_dataset = data.get_datasets()

    data_module = LandmarkDataModule(args)

    # Define Model
    effnet = Effnet_Landmark()
    model = LandmarkClassifier(args, effnet, data_module.get_dim())

    # Logger
    tb_logger = loggers.TensorBoardLogger(args.log_dir)

    # Model Saving Setup
    checkpoint_callback = ModelCheckpoint(
        monitor='val_gap',
        dirpath=args.model_dir,
        filename=args.name + '{epoch:02d}-{val_gap:.4f}')

    # Define trainer and fit to data
    trainer = Trainer(
        gpus=args.gpus, 
        logger = tb_logger, 
        #auto_scale_batch_size='binsearch', # automatically picks the batch size based on how much GPU memory we have available
        default_root_dir=args.model_dir, 
        callbacks=[checkpoint_callback], 
        max_epochs = args.epochs, 
        progress_bar_refresh_rate = 5)
    #trainer.tune(model, data_module)
    trainer.fit(model, data_module)

    trainer.test(datamodule = data_module)