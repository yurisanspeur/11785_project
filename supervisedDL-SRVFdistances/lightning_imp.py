import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, CosineAnnealingLR
from matplotlib import pyplot as plt
import os
import math
wandb_logger = WandbLogger(project="Oblika", name="cyclic_pad_model_more_patient")
#data = np.load(
#    "massive_training_data_distances.pickle",
#    allow_pickle=True,
#)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')
torch.manual_seed(0)
#c1_xs = []
#c1_gs = []
#c2_xs = []
#c2_gs = []
#dists = []
#for i in range(len(data)): # We will write this data once to disk at the sample level (a pair of shape and its corresponding distance)
    # if len(np.array(data[i]['c1']['G'])) == 99 and len(np.array(data[i]['c2']['G'])) == 99:
#    c1_xs.append(np.array(data[i]["c1"]["x"]))
    # c1_gs.append(np.array(data[i]['c1']['G']))
#    c2_xs.append(np.array(data[i]["c2"]["x"]))
    # c2_gs.append(np.array(data[i]['c2']['G']))
#    dists.append(np.array(data[i]["dist"]))
#    X = torch.cat((torch.Tensor(data[i]["c1"]["x"]), torch.Tensor(data[i]["c2"]["x"])), dim=1) # This should be (100,4)
#    dist = torch.Tensor([data[i]["dist"]])
#    torch.save((X,dist), f"Fab_data/data_{i}.pt")
    # c1_xs = np.array(c1_xs)
    # c1_gs = np.array(c1_gs)
    # c2_xs = np.array(c2_xs)
    # c2_gs = np.array(c2_gs)
#cxy_data = np.concatenate((c1_xs, c2_xs), axis=2)
#dists = torch.FloatTensor(np.array([dists]).T)

#cxy_data = torch.FloatTensor(cxy_data)  # cast as a torch.float32 tensor
#label = (
#    dists.squeeze()
#)  # cast as torch.float32 for compatility with layer kernels and should have only batch_size dimension

#generator = torch.Generator()
#generator.manual_seed(0)
class ShapeDataset(torch.utils.data.Dataset):
    def __init__(self, path='Fab_data'):
        self.path = path

    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, ind):
        X, dist = torch.load(f"Fab_data/data_{ind}.pt")
        return X, dist


#class ShapeTestDataset(torch.utils.data.Dataset):
#    def __init__(self, feature_data, label):
#        self.feature_data = feature_data
#        self.label = label
#
#    def __len__(self):
#        return len(self.feature_data)
#
#    def __getitem__(self, ind):
#        return self.feature_data[ind], self.label[ind]
cxy_data = ShapeDataset(path="Fab_data")
len_data = len(cxy_data)
train_len = int(0.8 * len_data)
val_len = int(0.1 * len_data)
test_len = len_data - train_len - val_len
train_data, val_data, test_data = torch.utils.data.random_split(
    cxy_data, [train_len, val_len, test_len]
)


BATCH_SIZE = 128

#train_loader = torch.utils.data.DataLoader(
#    train_data, num_workers=8, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True
#)
#val_loader = torch.utils.data.DataLoader(
#    val_data, num_workers=8, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False
#)
#test_loader = torch.utils.data.DataLoader(
#    test_data, num_workers=8, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False
#)
class MyDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size,
        train_set,                                                                          
        val_set,                                                                            
        test_set,                                                                           
        train_collate_fn=None,                                                              
        val_collate_fn=None,                                                                
        test_collate_fn=None,                                                               
    ):                                                                                                                                                                                   
        self.train_set = train_set                                                          
        self.val_set = val_set                                                              
        self.test_set = test_set                                                            
        self.batch_size = batch_size                                                        
        self.train_collate_fn = train_collate_fn                                            
        self.val_collate_fn = val_collate_fn                                                
        self.test_collate_fn = test_collate_fn                                              
                                                                                            
    def train_dataloader(self):                                                                                                                                                          
        return DataLoader(                                                                  
            self.train_set,                                                                 
            batch_size=self.batch_size,                                                                                                                                                  
            shuffle=True,                                                                   
            pin_memory=True,                                                                
            num_workers=8,                                                                                                                                                               
            persistent_workers=True,                                                        
            collate_fn=self.train_collate_fn,                                               
        )                                                                                   
                                                                                            
    def val_dataloader(self):                                                               
        return DataLoader(                                                                  
            self.val_set,                                                                                                                                                                
            batch_size=self.batch_size,                                                     
            shuffle=False,                                                                                                                                                               
            pin_memory=True,                                                                
            persistent_workers=True,                                                        
            num_workers=8,                                                                  
            collate_fn=self.val_collate_fn,                                                 
        )                                                                                   
                                                                                            
    def test_dataloader(self):                                                              
        return DataLoader(                                                                  
            self.test_set,                                                                  
            batch_size=self.batch_size,                                                     
            shuffle=False,                                                                                                                                                               
            pin_memory=True,                                                                
            persistent_workers=True,                                                        
            num_workers=8,
            collate_fn=self.test_collate_fn,
        )



class CyclicPad(nn.Module):
    def __init__(self, **kwargs):
        self.kernelsize = 5
        super(CyclicPad, self).__init__(**kwargs)

    def forward(self, inputs):
        length = inputs.shape[1] # number of samples 100
        n = math.floor(self.kernelsize / 2) # 2
        a = inputs[:, length - n : length, :]
        b = inputs[:, 0:n, :]

        return tf.concat([a, inputs, b], 1)


class LayerNorm(nn.Module):
    def __init__(self, c1, c2):

        self.c1 = c1
        self.c2 = c2
    def forward(self):
        return (self.c1 - self.c2) ** 2



class Network(LightningModule):
    def __init__(self, dim, length):
        super(Network, self).__init__()
        self.dim = dim
        self.conv_layer = nn.Sequential(
            nn.Conv1d(dim, 2 * dim, 5, padding="same", bias=False),
            nn.BatchNorm1d(2 * dim),
            nn.ReLU(),
            nn.Conv1d(2 * dim, 4 * dim, 5, padding="same", bias=False),
            nn.BatchNorm1d(4 * dim),
            nn.ReLU(),
            # nn.MaxPool1d(2, padding="same"),
            nn.Conv1d(4 * dim, 8 * dim, 5, padding="same", bias=False),
            nn.BatchNorm1d(8 * dim),
            nn.ReLU(),
            # nn.MaxPool1d(2, padding="same"),
            nn.Conv1d(8 * dim, 16 * dim, 5, padding="same", bias=False),
            nn.BatchNorm1d(16 * dim),
            nn.ReLU(),
            # nn.MaxPool1d(2, padding="same"),
            nn.Conv1d(16 * dim, 32 * dim, 5, padding="same", bias=False),
            nn.BatchNorm1d(32 * dim),
            nn.ReLU(),
            # nn.MaxPool1d(2, padding="same"),
            nn.Conv1d(32 * dim, 64 * dim, 5, padding="same", bias=False),
            nn.BatchNorm1d(64 * dim),
            nn.ReLU(),
            # nn.MaxPool1d(2, padding="same"),
            nn.Conv1d(64 * dim, 128 * dim, 5, padding="same", bias=False),
            nn.BatchNorm1d(128 * dim),
            nn.ReLU(),
            # nn.MaxPool1d(2, padding="same"),
            nn.Conv1d(128 * dim, 256 * dim, 5, padding="same", bias=False),
            nn.BatchNorm1d(256 * dim),
            nn.ReLU(),
            # nn.MaxPool1d(2, padding="same"),
            # nn.LeakyReLU(negative_slope=0.2),
            #            nn.LeakyReLU(negative_slope=0.2),
            #            nn.Conv1d(128, 256, 5, padding="same", bias=False),
            #            nn.BatchNorm1d(256),
            #            nn.LeakyReLU(negative_slope=0.2),
            #            #            nn.LeakyReLU(negative_slope=0.2),
            #            nn.Conv1d(256, 512, 5, padding="same", bias=False),
            #            nn.BatchNorm1d(512),
            #            nn.LeakyReLU(negative_slope=0.2)
            #            nn.LeakyReLU(negative_slope=0.2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                256 * dim * length * 2, 256 * dim
            ),  # 2 here is to accomodate for the concatenation of shape representations
            nn.ReLU(),
            nn.Linear(256 * dim, 64 * dim),
            nn.ReLU(),
            nn.Linear(64 * dim, 16 * dim),
            nn.ReLU(),
            nn.Linear(16 * dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, 1),
        )

    def forward(self, x):
        c1 = x[:, :, : self.dim]
        c2 = x[:, :, self.dim :]
        c1_perm = c1.permute(0, 2, 1)
        c2_perm = c2.permute(0, 2, 1)
        out_c1 = self.conv_layer(c1_perm)
        out_c2 = self.conv_layer(c2_perm)
        # concatenate the two representations
        out_cat = torch.cat((out_c1.permute(0, 2, 1), out_c2.permute(0, 2, 1)), axis=2)

        #        out_to_fc = nn.AvgPool1d(2)(out_cat).squeeze() #FIXME: Think this is destroying all the featurization ; flatten instead
        dist = self.fc(out_cat).squeeze()

        return dist
