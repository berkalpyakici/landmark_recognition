import os
import cv2
import random
import torch
import time
from tqdm import tqdm as tqdm
import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader, Dataset

from utilities import make_transform_train, make_transform_val, global_average_precision_score
from models import Effnet_Landmark, ArcMarginProduct

class Images(torch.utils.data.Dataset):
    def __init__(self, csv, img_dim, transform = None, withlabel = True):
        self.csv = csv.reset_index()
        self.img_dim = img_dim
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
            # TODO: FIX THIS. PYTORCH DOESNT LIKE EMPTY TENSORS
            if self.withlabel:
                return torch.tensor(np.zeros(shape = (self.img_dim, self.img_dim))), torch.tensor(0)
            else:
                return torch.tensor(np.zeros(shape = (self.img_dim, self.img_dim)))
        
        image = image.astype(np.float32).transpose(2, 0, 1)

        if self.withlabel: 
            return torch.tensor(image), torch.tensor(row['landmark_id'])
        else:
            return torch.tensor(image)

class Landmark():
    def __init__(self, args):
        self.args = args

        # Set of Images and Split Sets (Training, CV, Tets)
        self.images = None
        self.train_set = None
        self.cv_set = None
        self.test_set = None

        self.out_dim = 0

        self.seed()

        if self.args.cuda:
            torch.cuda.set_device(0)
    
    def train(self):
        if self.train_set is None:
            print('Training set is not loaded.')
            return

        # Load Dataset
        train_dataset = Images(self.train_set, self.args.img_dim, transform = make_transform_train(self.args.img_dim, self.args.img_dim), withlabel = True)
        valid_dataset = Images(self.cv_set, self.args.img_dim, transform = make_transform_val(self.args.img_dim, self.args.img_dim), withlabel = True)

        valid_loader = DataLoader(valid_dataset, batch_size = self.args.batch_size, num_workers = self.args.num_workers, drop_last = True)        


        # Init Model
        model = Effnet_Landmark('tf_efficientnet_b7_ns', out_dim = self.out_dim)

        if self.args.cuda:
            model = model.cuda()

        # Loss Function
        tmp = np.sqrt(1 / np.sqrt(self.train_set['landmark_id'].value_counts().sort_index().values))
        margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05

        def loss_fn(logits_m, target):
            return ArcFaceLossAdaptiveMargin(margins = margins, s = 80)(logits_m, target, self.out_dim)

        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr = self.args.lr, momentum = 0.9, weight_decay = 1e-5)
        
        # Adaptive LR
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.args.epochs)

        # Set model file name.
        model_file = os.path.join(self.args.model_dir, f'{self.args.name}.pth')

        # Set variable for previous Micro AP score.
        prev_val_map = 0.0

        # Run training!
        for epoch in range(1, self.args.epochs + 1):
            print(time.ctime(), 'Epoch:', epoch)

            lr_scheduler.step(epoch - 1)

            train_loader = DataLoader(train_dataset, batch_size = self.args.batch_size, num_workers = self.args.num_workers, drop_last = True)        

            train_loss = self.train_epoch(model, train_loader, optimizer, loss_fn)
            val_loss, val_ac, val_map = self.val_epoch(model, valid_loader, loss_fn)

            print(time.ctime() + ' ' + f'Epoch {epoch}, LR: {optimizer.param_groups[0]["lr"]:.7f}, Train Loss: {np.mean(train_loss):.5f}, Val Loss: {(val_loss):.5f}')
            print(time.ctime() + ' ' + f'Epoch {epoch}, Val Acc {(val_acc):.6f}, Val Micro AP: {(val_map):.6f}')
            
            with open(os.path.join(args.log_dir, f'{self.args.name}.txt'), 'a') as f:
                f.write(content + '\n')

            print('Val Micro AP ({:.6f} --> {:.6f})'.format(prev_val_map, val_map))

            prev_val_map = val_map
            
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, model_file)

            print('Saving model...')

    def train_epoch(self, model, loader, optimizer, loss_fn):
        #print("we doing things now")
        model.train()
        print("yay we trained")
        train_loss = []

        bar = tqdm(loader)

        #print("hel")
        # print(bar)

        for i, (data, target) in enumerate(bar):
            
            print("starting training")
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            print("zeroed")

            logits_m = model(data)
            print("got logits")
            loss = loss_fn(logits_m, target)
            print("computed loss")
            loss.backward()
            print("went backwards")
            optimizer.step()
            print("stepped forward")

            if self.args.cuda:
                torch.cuda.synchronize()
                
            loss_np = loss.detach().cpu().numpy()
            train_loss.append(loss_np)
            print("added loss")
            bar.set_description('Loss: %.5f' % (loss_np))
        
        return train_loss
    
    def val_epoch(self, model, loader, loss_fn):
        model.eval()
        val_loss = []
        PRODS_M = []
        PREDS_M = []
        TARGETS = []

        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(loader)):
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()

                logits_m = model(data)

                lmax_m = logits_m.max(1)
                probs_m = lmax_m.values
                preds_m = lmax_m.indices

                PRODS_M.append(probs_m.detach().cpu())
                PREDS_M.append(preds_m.detach().cpu())
                TARGETS.append(target.detach().cpu())

                loss = loss_fn(logits_m, target)
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

        print(f'Loaded {self.images.shape[0]} images.')
        print(f'Out dimension is {self.out_dim}.')
    
    def split_images(self, train = 0.7, cv = 0.2, test = 0.1):
        if self.images is None:
            print("Images are not loaded.")
            return
        
        self.train_set = self.images.sample(frac = train).reset_index(drop = True)
        self.cv_set = self.images.sample(frac = cv).reset_index(drop = True)
        self.test_set = self.images.sample(frac = test).reset_index(drop = True)

        print(f'Training set contains {self.train_set.shape[0]} images.')
        print(f'CV set contains {self.cv_set.shape[0]} images.')
        print(f'Test set contains {self.test_set.shape[0]} images.')

    def seed(self, s = 0):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

        if self.args.cuda:
            torch.cuda.manual_seed(s)
            torch.cuda.manual_seed_all(s)
            torch.backends.cudnn.deterministic = True