import pickle, numpy as np

# import torch.nn as nn
import torch, os
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import BasePredictionWriter

# from tensorflow.keras.models import load_model
from data_model_utils import DistanceModel
import pytorch_lightning as pl

# model = load_model("OCR")  # Tensorflow
# Load the template DB - this is what we match against

if __name__ == "__main__":
    template = pickle.load(open("debug_template.pickle", "rb"))
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #    # Load the intake - this is what is unknown and needs to be recognized
    intake = pickle.load(open("list_resampled_chars.pickle", "rb"))
    breakpoint()
    # light_model = DistanceModel(2, num_conv_blocks=8).to(device)
    # bmp = "oblika_ckpts/32BS_8conv_block-epoch=548-step=983808-val_loss=0.1192.ckpt"
    # light_model.load_state_dict(torch.load(f"{bmp}")["state_dict"])
    intake = intake
    # intake_arr_b = np.empty((len(template) * len(intake), 2, 100))
    # template_arr_b = np.empty((len(template) * len(intake), 2, 100))
    # intake_results = [""] * len(intake)
    # for i, intake_char in enumerate(intake):
    #    # min_dist = np.inf
    #    # Create an empty matrix to house the i,j for all j
    #    for j, template_char in enumerate(
    #        template
    #    ):  # FIXME: Much more efficient to accumulate each pairs per intake and then feed it to predict as a batch, This should be a one time thing we save to disk and load from
    #        # disk as pairs of shapes whose index is the batch index who dim size if len(intake) * len(template)
    #        # intake_arr = np.asarray(intake_char["c__x"])
    #        intake_arr = np.asarray(intake_char["x"])
    #        print(intake_arr.shape)
    #        if intake_arr.shape != (100, 2):
    #            print("Skipping intake char because shape is messed up!")
    #            continue
    #        template_arr = np.asarray(template_char["c"]["x"])
    #        # intake_arr = np.asarray(intake[intake_char]['x'])
    #        intake_arr_b[i * len(template) + j, :, :] = intake_arr.T
    #        template_arr_b[i * len(template) + j, :, :] = template_arr.T
    #        c1 = torch.tensor(intake_arr.T)  # (2,100)
    #        c2 = torch.tensor(template_arr.T)  # (2,100)
    #        pair = torch.cat([c1, c2], dim=0)
    #        torch.save(pair, f"inference_folder/pair_{i * len(template) + j}.pt")

    # torch.save(
    # Add the batch dimension
    #        intake_arr = intake_arr[np.newaxis, ...]
    #        template_arr = template_arr[np.newaxis, ...]
    # intake_tensor = torch.FloatTensor(intake_arr)
    # template_tensor = torch.FloatTensor(template_arr)
    # input_tensor = torch.cat((intake_tensor, template_tensor), axis=2)
    #        if j == len(template) - 1: # predict
    #            breakpoint()
    #            dist = model.predict([template_arr, intake_arr]) #TF
    # dist = model(input_tensor)
    #        print(dist, template_char["char"])
    #        if dist < min_dist:
    #            intake_results[i] = template_char["char"]
    #            min_dist = dist
    # Push all the shape pairs to the GPU
    # c1s = torch.tensor(intake_arr_b)
    # c2s = torch.tensor(template_arr_b)
    # big_matrix = torch.cat([c1s, c2s], dim=1)
    # torch.save(big_matrix, "big_inference_matrix.pt")
    # breakpoint()
    # intake = pickle.load(open("screenshot_11785.pickle", "rb"))

    # class Network(nn.Module):
    #    def __init__(self, dim, length):
    #        super(Network, self).__init__()
    #        self.dim = dim
    #        self.conv_layer = nn.Sequential(
    #            nn.Conv1d(dim, 2 * dim, 5, padding="same", bias=False),
    #            nn.BatchNorm1d(2 * dim),
    #            nn.ReLU(),
    #            nn.Conv1d(2 * dim, 4 * dim, 5, padding="same", bias=False),
    #            nn.BatchNorm1d(4 * dim),
    #            nn.ReLU(),
    #            # nn.MaxPool1d(2, padding="same"),
    #            nn.Conv1d(4 * dim, 8 * dim, 5, padding="same", bias=False),
    #            nn.BatchNorm1d(8 * dim),
    #            nn.ReLU(),
    #            # nn.MaxPool1d(2, padding="same"),
    #            nn.Conv1d(8 * dim, 16 * dim, 5, padding="same", bias=False),
    #            nn.BatchNorm1d(16 * dim),
    #            nn.ReLU(),
    #            # nn.MaxPool1d(2, padding="same"),
    #            nn.Conv1d(16 * dim, 32 * dim, 5, padding="same", bias=False),
    #            nn.BatchNorm1d(32 * dim),
    #            nn.ReLU(),
    #            # nn.MaxPool1d(2, padding="same"),
    #            nn.Conv1d(32 * dim, 64 * dim, 5, padding="same", bias=False),
    #            nn.BatchNorm1d(64 * dim),
    #            nn.ReLU(),
    #            # nn.MaxPool1d(2, padding="same"),
    #            nn.Conv1d(64 * dim, 128 * dim, 5, padding="same", bias=False),
    #            nn.BatchNorm1d(128 * dim),
    #            nn.ReLU(),
    #            # nn.MaxPool1d(2, padding="same"),
    #            nn.Conv1d(128 * dim, 256 * dim, 5, padding="same", bias=False),
    #            nn.BatchNorm1d(256 * dim),
    #            nn.ReLU(),
    #            # nn.MaxPool1d(2, padding="same"),
    #            # nn.LeakyReLU(negative_slope=0.2),
    #            #            nn.LeakyReLU(negative_slope=0.2),
    #            #            nn.Conv1d(128, 256, 5, padding="same", bias=False),
    #            #            nn.BatchNorm1d(256),
    #            #            nn.LeakyReLU(negative_slope=0.2),
    #            #            #            nn.LeakyReLU(negative_slope=0.2),
    #            #            nn.Conv1d(256, 512, 5, padding="same", bias=False),
    #            #            nn.BatchNorm1d(512),
    #            #            nn.LeakyReLU(negative_slope=0.2)
    #            #            nn.LeakyReLU(negative_slope=0.2),
    #        )
    #        self.fc = nn.Sequential(
    #            nn.Flatten(),
    #            nn.Linear(
    #                256 * dim * length * 2, 256 * dim
    #            ),  # 2 here is to accomodate for the concatenation of shape representations
    #            nn.ReLU(),
    #            nn.Linear(256 * dim, 64 * dim),
    #            nn.ReLU(),
    #            nn.Linear(64 * dim, 16 * dim),
    #            nn.ReLU(),
    #            nn.Linear(16 * dim, 4 * dim),
    #            nn.ReLU(),
    #            nn.Linear(4 * dim, 1),
    #        )
    #
    #    def forward(self, x):
    #        c1 = x[:, :, : self.dim]
    #        c2 = x[:, :, self.dim :]
    #        c1_perm = c1.permute(0, 2, 1)
    #        c2_perm = c2.permute(0, 2, 1)
    #        out_c1 = self.conv_layer(c1_perm)
    #        out_c2 = self.conv_layer(c2_perm)
    #        # concatenate the two representations
    #        out_cat = torch.cat((out_c1.permute(0, 2, 1), out_c2.permute(0, 2, 1)), axis=2)
    #
    #        #        out_to_fc = nn.AvgPool1d(2)(out_cat).squeeze() #FIXME: Think this is destroying all the featurization ; flatten instead
    #        dist = self.fc(out_cat).squeeze()
    #
    #        return dist

    # model = Network(2, 100)
    # model.load_state_dict(
    #    torch.load("HW5_project_GPU_more_data_less_depth.pth")["model_state_dict"]
    # )
    # Load our surrogate model for shape distances

    # Create a Dataset class in order to be able to serve a batch through DataLoader to predict method of Lightning Module for multi-GPU inference
    class InferenceData(Dataset):
        def __init__(
            self,
            path="/home/jovyan/11785_project/supervisedDL-SRVFdistances/inference_folder",
        ):
            self.path = path
            self.inf_matrix = torch.load("big_inference_matrix.pt")

        def __len__(self):
            # return len(os.listdir(self.path))
            return self.inf_matrix.shape[0]

        def __getitem__(self, ind):
            return (
                self.inf_matrix[ind].type(light_model.conv_layer[0].conv.weight.dtype),
                ind,
            )
            # return (
            #    torch.load(f"inference_folder/pair_{ind}.pt").type(
            #        light_model.conv_layer[0].conv.weight.dtype
            #    ),
            #    ind,
            # )

    # inf_data = InferenceData(
    #    path="/home/jovyan/11785_project/supervisedDL-SRVFdistances/inference_folder"
    # )
    # inf_data = InferenceData()
    # testloader = torch.utils.data.DataLoader(
    #    inf_data, batch_size=8192, persistent_workers=True, num_workers=2
    # )
    # Define a trainer

    # trainer = pl.Trainer(
    #    strategy=DDPStrategy(find_unused_parameters=False), accelerator="gpu", devices=2
    # )
    # trainer.predict(light_model, testloader)

    # for test_batch in testloader:
    #    breakpoint()
    #    print("Test")

    # x = [
    #    torch.cat([c1s, c2s], dim=1)
    #    .to(device)
    #    .type(light_model.conv_layer[1].weight.dtype)
    # ]
    # dists = light_model(x).reshape(len(intake), len(template)).detach().cpu().numpy()
    dists = torch.load("results.pt").reshape(len(intake), len(template)).cpu().numpy()
    ## dists = model.predict([intake_arr_b, template_arr_b]).reshape((len(intake), len(template)))
    ## dists = model.predict([template_arr_b, intake_arr_b]).reshape((len(intake), len(template)))
    ## Find the closest match for unknown character relative to the characters that compose our database
    indices = np.argmin(dists, axis=1)
    ## Decode the characters per min indices and associated template characters
    intake_result_dict = {
        f"char_{j+1}": {
            "char": template[i]["char"],
            "x_pos": intake[j]["x_pos"],
            "w_pos": intake[j]["w_pos"],
            "y_pos": intake[j]["y_pos"],
            "h_pos": intake[j]["h_pos"],
        }
        for j, i in enumerate(indices)
    }
    intake_result = [template[i]["char"] for i in indices]
    breakpoint()
    with open(f"decoded_chars_intake.pickle", "wb") as fout:
        pickle.dump(intake_result, fout)
    breakpoint()

    # FIXME: Need to maintain spatial layout
    print("".join(intake_result))
