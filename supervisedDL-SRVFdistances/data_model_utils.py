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

class ShapeDataset(torch.utils.data.Dataset):
    def __init__(self, path='Fab_data'):
        self.path = path

    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, ind):
        X, dist = torch.load(f"Fab_data/data_{ind}.pt")
        return X, dist


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
        super(CyclicPad, self).__init__(**kwargs)
        self.kernelsize = 5

    def forward(self, inputs): # (B, F, N)

        length = inputs.shape[2] # number of samples 100
        n = math.floor(self.kernelsize / 2) # 2
        #a = inputs[:, length - n : length, :]
        #b = inputs[:, 0:n, :]
        a = inputs[:, :,length - n : length]
        b = inputs[:, :, 0:n]

        return torch.cat([a, inputs, b], dim=2)


class LayerNorm(nn.Module):

    def __init__(self):
        super(LayerNorm, self).__init__()

    def forward(self, c1, c2):
        return torch.sum((c1 - c2) ** 2,dim=(1,2))


class ConvBlock(nn.Module):

    def __init__(self, in_features, out_features):
        super(ConvBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cyclic_pad = CyclicPad()
        self.conv = nn.Conv1d(in_features, out_features,5, padding="same",bias=False)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.bn(self.conv(self.cyclic_pad(x))))
        return x

class LinearBlock(nn.Module):

    def __init__(self, in_features, out_features):
        super(LinearBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lin = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0,2,1) # so that batchnorm acts on feature dimension
        x = self.lin(x)
        x = x.permute(0,2,1) # B,F,N
        x = self.bn(x)
        x = self.relu(x)
        return x


class DistanceModel(LightningModule):
    def __init__(self, dim, num_conv_blocks):
        super(DistanceModel, self).__init__()
        self.dim = dim
        conv_blocks = [ConvBlock(dim, 2*dim)]
        for i in range(num_conv_blocks - 1):
            in_features = conv_blocks[i].out_features
            conv_blocks.append(ConvBlock(in_features, 2*in_features))

        self.conv_layer = nn.Sequential(*conv_blocks)
        linear_blocks = [LinearBlock(conv_blocks[-1].out_features,int(conv_blocks[-1].out_features/2))]

        lin_features = int(linear_blocks[-1].out_features)
        while lin_features >= 4:
            lin_features = int(linear_blocks[-1].out_features)
            linear_blocks.append(LinearBlock(lin_features, int(lin_features/2)))
            lin_features /= 2

        self.linear_layer = nn.Sequential(*linear_blocks)

        #self.conv_layer = nn.Sequential(
        #    CyclicPad(),
        #    nn.Conv1d(dim, 2 * dim, 5, padding="same", bias=False),
        #    nn.BatchNorm1d(2 * dim),
        #    nn.ReLU(),
        #    CyclicPad(),
        #    nn.Conv1d(2 * dim, 4 * dim, 5, padding="same", bias=False),
        #    nn.BatchNorm1d(4 * dim),
        #    nn.ReLU(),
        #    #nn.MaxPool1d(kernel_size=5,padding=2),
        #    CyclicPad(),
        #    nn.Conv1d(4 * dim, 8 * dim, 5, padding="same", bias=False),
        #    nn.BatchNorm1d(8 * dim),
        #    nn.ReLU(),
        #    #nn.MaxPool1d(kernel_size=5,padding=2),
        #    # nn.MaxPool1d(2, padding="same"),
        #    CyclicPad(),
        #    nn.Conv1d(8 * dim, 16 * dim, 5, padding="same", bias=False),
        #    nn.BatchNorm1d(16 * dim),
        #    nn.ReLU(),
        #    #nn.MaxPool1d(kernel_size=5,padding=2),
        #    # nn.MaxPool1d(2, padding="same"),
        #    CyclicPad(),
        #    nn.Conv1d(16 * dim, 32 * dim, 5, padding="same", bias=False),
        #    nn.BatchNorm1d(32 * dim),
        #    nn.ReLU(),
        #    #nn.MaxPool1d(kernel_size=5,padding=2),
        #    # nn.MaxPool1d(2, padding="same"),
        #    CyclicPad(),
        #    nn.Conv1d(32 * dim, 64 * dim, 5, padding="same", bias=False),
        #    nn.BatchNorm1d(64 * dim),
        #    nn.ReLU(),
        #    #nn.MaxPool1d(kernel_size=5,padding=2),
        #    # nn.MaxPool1d(2, padding="same"),
        #    CyclicPad(),
        #    nn.Conv1d(64 * dim, 128 * dim, 5, padding="same", bias=False),
        #    nn.BatchNorm1d(128 * dim),
        #    nn.ReLU(),
        #    #nn.MaxPool1d(kernel_size=5,padding=2),
        #    # nn.MaxPool1d(2, padding="same"),
        #    CyclicPad(),
        #    nn.Conv1d(128 * dim, 256 * dim, 5, padding="same", bias=False),
        #    nn.BatchNorm1d(256 * dim),
        #    nn.ReLU(),
            #nn.MaxPool1d(kernel_size=5,padding=2),
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
        #)
        #self.fc = nn.Sequential(
        #    nn.Flatten(),
        #    nn.Linear(
        #        256 * dim * length * 2, 256 * dim
        #    ),  # 2 here is to accomodate for the concatenation of shape representations
        #    nn.ReLU(),
        #    nn.Linear(256 * dim, 64 * dim),
        #    nn.ReLU(),
        #    nn.Linear(64 * dim, 16 * dim),
        #    nn.ReLU(),
        #    nn.Linear(16 * dim, 4 * dim),
        #    nn.ReLU(),
        #    nn.Linear(4 * dim, 1),
        #)
        #self.linear = nn.Sequential(nn.Linear(512, 256),
        #                            #nn.BatchNorm1d(),
        #                            nn.ReLU(),
        #                            nn.Linear(256,128),
        #                            #nn.BatchNorm1d(128),
        #                            nn.ReLU(),
        #                            nn.Linear(128,64),
        #                            #nn.BatchNorm1d(64),
        #                            nn.ReLU(),
        #                            nn.Linear(64,32),
        #                            #nn.BatchNorm1d(32),
        #                            nn.ReLU(),
        #                            nn.Linear(32,16),
        #                            #nn.BatchNorm1d(16),
        #                            nn.ReLU(),
        #                            nn.Linear(16,8),
        #                            #nn.BatchNorm1d(8),
        #                            nn.ReLU(),
        #                            nn.Linear(8,4),
        #                            #nn.BatchNorm1d(4),
        #                            nn.ReLU(),
        #                            nn.Linear(4,2),
        #                            #nn.BatchNorm1d(2),
        #                            nn.ReLU()
        #                            )
        self.layer_norm = LayerNorm()
        

    def forward(self, x):
        c1 = x[0][:, :self.dim, :]
        c2 = x[0][:, self.dim:,:]
        #c1_perm = c1.permute(0, 2, 1)
        #c2_perm = c2.permute(0, 2, 1)
        out_c1 = self.conv_layer(c1)
        out_c2 = self.conv_layer(c2)
        # concatenate the two representations
        #out_cat = torch.cat((out_c1.permute(0, 2, 1), out_c2.permute(0, 2, 1)), axis=2)

        #        out_to_fc = nn.AvgPool1d(2)(out_cat).squeeze() #FIXME: Think this is destroying all the featurization ; flatten instead
        #dist = self.fc(out_cat).squeeze()
        # Bring the high dimensional shapes to a lower dimension before calculating norm
        #out_c1_linear = out_c1.permute(0, 2, 1)
        #out_c2_linear = out_c2.permute(0, 2, 1)
        #c1_low_dim = self.linear(out_c1_linear)
        #c2_low_dim = self.linear(out_c2_linear)
        out_c1_linear = self.linear_layer(out_c1)
        out_c2_linear = self.linear_layer(out_c2)
        dist = self.layer_norm(out_c1_linear, out_c2_linear)

        return dist

    def training_step(self, batch, batch_idx):
        preds = self(batch).squeeze()
        targs = batch[1].squeeze()
        loss = nn.L1Loss()(preds, targs.type(preds.dtype))
        self.log("training_loss", loss, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        preds = self(batch).squeeze()
        targs = batch[1].squeeze()
        loss = nn.L1Loss()(preds, targs.type(preds.dtype))
        self.log("val_loss", loss, on_epoch=True)
        return {"val_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer, mode="min", patience=50, factor=0.75
                ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }
