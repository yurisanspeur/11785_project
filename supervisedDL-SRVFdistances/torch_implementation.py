import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score
import gc
import wandb
from tqdm import tqdm

data = np.load(
    "massive_training_data_distances.pickle",
    allow_pickle=True,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

wandb.login(
    key="4a7d69aeafc0f3c0c442c337417b282a3231e52e"
)  # API Key is in your wandb account, under settings (wandb.ai/settings)
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


BATCH_SIZE = 512

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

# Define the model
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(100, 128, 5, padding="same", bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, 256, 5, padding="same", bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 512, 5, padding="same", bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU()
            #            nn.LeakyReLU(negative_slope=0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 8),
            nn.Linear(8, 4),
            nn.Linear(4, 2),
            nn.Linear(2, 1),
        )

    def forward(self, x):
        c1 = x[:, :, :2]
        c2 = x[:, :, 2:]
        out_c1 = self.conv_layer(c1)
        out_c2 = self.conv_layer(c2)
        # concatenate the two representations
        out_cat = torch.cat((out_c1, out_c2), axis=2)
        out_to_fc = nn.AvgPool1d(4)(out_cat).squeeze()
        dist = self.fc(out_to_fc).squeeze()

        return dist


def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


model = Network()
model.apply(initialize_weights)
model.to(device)
config = {"batch_size": BATCH_SIZE, "lr": 8e-3, "epochs": 500}
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=5e-15)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=1, verbose=True
)


def evaluate(data_loader, model):

    val_loss = 0
    batch_bar = tqdm(
        total=len(data_loader),
        dynamic_ncols=True,
        leave=False,
        position=0,
        desc="Val",
        ncols=3,
    )
    # TODO Fill this function out, if you're using it.
    with torch.no_grad():
        model.eval()
        for batch_idx, data in enumerate(data_loader):
            # Unpack data
            X, y = data
            X, y = (
                X.to(device),
                y.to(device),
            )
            output = model(X)
            loss = criterion(output, y)
            val_loss += loss
            batch_bar.set_postfix(
                loss="{:.04f}".format(val_loss / (batch_idx + 1)),
            )
            batch_bar.update()
        # Normalize the loss per the len of dataloader
        batch_bar.close()
        loss = float(val_loss / len(data_loader))

    return loss


def train_step(train_loader, model, optimizer, criterion, scheduler):
    batch_bar = tqdm(
        total=len(train_loader),
        dynamic_ncols=True,
        leave=False,
        position=0,
        desc="Train",
        ncols=3,
    )
    train_loss = 0
    model.train()

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        # TODO: Fill this with the help of your sanity check
        X, y = data
        X, y = (
            X.to(device),
            y.to(device),
        )
        output = model(X)

        loss = criterion(output, y)
        # HINT: Are you using mixed precision?
        loss.backward()
        optimizer.step()

        train_loss += loss
        batch_bar.set_postfix(
            loss=f"{train_loss/ (i+1):.4f}", lr=f"{optimizer.param_groups[0]['lr']}"
        )
        gc.collect()
        torch.cuda.empty_cache()
        batch_bar.update()
        del X
        del y
        del output

    batch_bar.close()
    train_loss /= len(train_loader)  # TODO

    return train_loss  # And anything else you may wish to get out of this function


torch.cuda.empty_cache()
gc.collect()

run = wandb.init(
    name="baseline_GPU",  ### Wandb creates random run names if you skip this field, we recommend you give useful names
    reinit=True,  ### Allows reinitalizing runs when you re-run this cell
    project="HW5_project",  ### Project should be created in your wandb account
    config=config,  ### Wandb Config for your run
    entity="rys",
)
best_val_loss = np.inf
# TODO: Please complete the training loop
breakpoint()
for epoch in range(config["epochs"]):

    # one training step
    train_loss = train_step(train_loader, model, optimizer, criterion, scheduler)
    print(
        "\nEpoch {}/{}: \n\t Train Loss {:.04f}\t Learning Rate {:.04f}".format(
            epoch + 1, config["epochs"], train_loss, optimizer.param_groups[0]["lr"]
        )
    )
    # one validation step (if you want)
    val_loss = evaluate(val_loader, model)
    print("Val Loss {:.04f}\t Lev. Distance {:.04f}".format(val_loss, val_loss))

    wandb.log(
        {
            "train loss": train_loss,
            "val_loss": val_loss,
            "current_lr": optimizer.param_groups[0]["lr"],
        }
    )
    # HINT: Calculating levenshtein distance takes a long time. Do you need to do it every epoch?
    # Does the training step even need it?

    # Where you have your scheduler.step depends on the scheduler you use.
    scheduler.step(metrics=val_loss)

    # Use the below code to save models
    if val_loss < best_val_loss:
        # path = os.path.join(root_path, model_directory, 'checkpoint' + '.pth')
        print("Saving model")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
            },
            "./HW5_project_GPU.pth",
        )
        best_val_loss = val_loss
        wandb.save("checkpoint_GPU.pth")
