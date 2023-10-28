import pickle, numpy as np
#import torch.nn as nn
import torch

from tensorflow.keras.models import load_model
from lightning_imp import DistanceModel

# Load the template DB - this is what we match against
template = pickle.load(open("debug_template.pickle", "rb"))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the intake - this is what is unknown and needs to be recognized
intake = pickle.load(open("list_resampled_chars.pickle", "rb"))
light_model = DistanceModel(2).to(device)
bmp = 'oblika_ckpts/CP_more_patient-epoch=805-step=361088-val_MAE_loss=0.1184.ckpt'
light_model.load_state_dict(torch.load(f"{bmp}")['state_dict'])
#intake = pickle.load(open("screenshot_11785.pickle", "rb"))

#class Network(nn.Module):
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


#model = Network(2, 100)
#model.load_state_dict(
#    torch.load("HW5_project_GPU_more_data_less_depth.pth")["model_state_dict"]
#)
# Load our surrogate model for shape distances
#model = load_model("OCR")  # Tensorflow
intake = intake[:60]
intake_arr_b = np.empty((len(template) * len(intake),2,100))
template_arr_b = np.empty((len(template) * len(intake),2,100))
intake_results = [""] * len(intake)
for i, intake_char in enumerate(intake):
    # min_dist = np.inf
    # Create an empty matrix to house the i,j for all j
    for j, template_char in enumerate(template):  # FIXME: Much more efficient to accumulate each pairs per intake and then feed it to predict as a batch
        #intake_arr = np.asarray(intake_char["c__x"])
        intake_arr = np.asarray(intake_char["x"])
        print(intake_arr.shape)
        if intake_arr.shape != (100,2):
            print("Skipping intake char because shape is messed up!")
            continue
        template_arr = np.asarray(template_char["c"]["x"])
        #intake_arr = np.asarray(intake[intake_char]['x'])
        intake_arr_b[i * len(template) + j,:,:] = intake_arr.T
        template_arr_b[i * len(template) + j,:,:] = template_arr.T
        # Add the batch dimension
#        intake_arr = intake_arr[np.newaxis, ...]
#        template_arr = template_arr[np.newaxis, ...]
        #intake_tensor = torch.FloatTensor(intake_arr)
        #template_tensor = torch.FloatTensor(template_arr)
        #input_tensor = torch.cat((intake_tensor, template_tensor), axis=2)
#        if j == len(template) - 1: # predict
#            breakpoint()
#            dist = model.predict([template_arr, intake_arr]) #TF
        #dist = model(input_tensor)
#        print(dist, template_char["char"])
#        if dist < min_dist:
#            intake_results[i] = template_char["char"]
#            min_dist = dist
# Push all the shape pairs to the GPU
c1s = torch.tensor(intake_arr_b)
c2s = torch.tensor(template_arr_b)
x = [torch.cat([c1s, c2s], dim=1).to(device).type(light_model.conv_layer[1].weight.dtype)]
dists = light_model(x).reshape(len(intake), len(template)).detach().cpu().numpy()
#dists = model.predict([intake_arr_b, template_arr_b]).reshape((len(intake), len(template)))
#dists = model.predict([template_arr_b, intake_arr_b]).reshape((len(intake), len(template)))
# Find the closest match for unknown character relative to the characters that compose our database
indices = np.argmin(dists,axis=1)
# Decode the characters per min indices and associated template characters
intake_result = [template[i]['char'] for i in indices]
print("".join(intake_result))
breakpoint()


