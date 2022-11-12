import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from torch.utils.data import Dataset, DataLoader

data = np.load(
    "massive_training_data_distances.pickle",
    allow_pickle=True,
)

c1_xs = []
c1_gs = []
c2_xs = []
c2_gs = []
dists = []
for i in range(len(data)):
    # if len(np.array(data[i]['c1']['G'])) == 99 and len(np.array(data[i]['c2']['G'])) == 99:
    c1_xs.append(np.array(data[i]["c1"]["x"]))
    # c1_gs.append(np.array(data[i]['c1']['G']))
    c2_xs.append(np.array(data[i]["c2"]["x"]))
    # c2_gs.append(np.array(data[i]['c2']['G']))
    dists.append(np.array(data[i]["dist"]))
    # c1_xs = np.array(c1_xs)
    # c1_gs = np.array(c1_gs)
    # c2_xs = np.array(c2_xs)
    # c2_gs = np.array(c2_gs)
    # dists = np.array([dists]).T

cxy_data = np.concatenate((c1_xs, c2_xs), axis=2)
dists = torch.FloatTensor(np.array([dists]).T)
cxy_data = torch.FloatTensor(cxy_data)  # cast as a torch.float32 tensor
label = (
    dists.squeeze()
)  # cast as torch.float32 for compatility with layer kernels and should have only batch_size dimension

len_data = cxy_data.shape[0]
train_len = int(0.7 * len_data)
val_len = int(0.1 * len_data)
test_len = len_data - train_len - val_len
generator = torch.Generator()
generator.manual_seed(0)

train_cxy_data, val_cxy_data, test_cxy_data = torch.utils.data.random_split(
    cxy_data, [train_len, val_len, test_len], generator=generator
)
train_label, val_label, test_label = torch.utils.data.random_split(
    label, [train_len, val_len, test_len], generator=generator
)


class ShapeTrainDataset(torch.utils.data.Dataset):
    def __init__(self, feature_data, label):
        self.feature_data = feature_data
        self.label = label

    def __len__(self):
        return len(self.feature_data)

    def __getitem__(self, ind):
        return self.feature_data[ind], self.label[ind]


class ShapeTestDataset(torch.utils.data.Dataset):
    def __init__(self, feature_data):
        self.feature_data = feature_data

    def __len__(self):
        return len(self.feature_data)

    def __getitem__(self, ind):
        return self.feature_data[ind]


train_data = ShapeTrainDataset(train_cxy_data, train_label)
val_data = ShapeTrainDataset(val_cxy_data, val_label)
test_data = ShapeTestDataset(test_cxy_data)


BATCH_SIZE = 64

train_loader = torch.utils.data.DataLoader(
    train_data, num_workers=8, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_data, num_workers=8, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    test_data, num_workers=8, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False
)
print("Batch size: ", BATCH_SIZE)
print(
    "Train dataset samples = {}, batches = {}".format(
        train_data.__len__(), len(train_loader)
    )
)
print(
    "Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader))
)
print(
    "Test dataset samples = {}, batches = {}".format(
        test_data.__len__(), len(test_loader)
    )
)
breakpoint()

# Define the model
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(100, 256, 5, padding="same", bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        c1 = x[:, :, :2]
        c2 = x[:, :, 2:]
        out_c1 = self.conv_layer(c1)
        out_c2 = self.conv_layer(c2)

        return out_c1, out_c2


model = Network()
out_c1, out_c2 = model(next(iter(train_loader))[0])

print(out_c1.shape, out_c2.shape)
