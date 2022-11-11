import pickle
from matplotlib import pyplot as plt
import numpy as np
from DLsrvf import trainModelwSPDA, defineModel, trainAndValidateModelwSPDA
from sklearn.model_selection import train_test_split


# Read in the dataset
data = pickle.load(open("massive_training_data_distances.pickle", "rb"))
trData = np.empty(shape=(len(data), 100, 4))
trLabels = np.empty(shape=(len(data), 1))
# tr2Data = np.empty(shape=(len(data), 100, 2))
for i, d in enumerate(data):
    c1 = np.asarray(d["c1"]["x"])
    c2 = np.asarray(d["c2"]["x"])
    dist = d["dist"]
    #    trData = np.vstack(trData[i],
    trData[i][:][:] = np.hstack((c1, c2))
    trLabels[i] = dist
#    print(dist)
#    if dist < 0.5:
#        fig = plt.figure(figsize=(10, 10))
#        ax = fig.add_subplot(1, 1, 1)
#        breakpoint()
#        xs1 = c1[:, 0]
#        ys1 = c1[:, 1]
#        xs2 = c2[:, 0]
#        ys2 = c2[:, 1]
#        ax.plot(xs1, ys1)
#        ax.plot(xs2, ys2)
#        fig.savefig(f"shapes_{i}.pdf")

# Split the data into train and test


X_train, X_test, y_train, y_test = train_test_split(trData, trLabels, test_size=0.2)

# Define the model
model = defineModel(100, 2, closed=False)  # FIXME: This should be closed ?
# Call the train function providing the data in the correct shape format
# model = trainModelwSPDA(model, 100, 2, True, trData, trLabels, 300, 512)
model, trainMSE, testMSE = trainAndValidateModelwSPDA(
    model, 100, 2, False, X_train, y_train, X_test, y_test, 500, 512
)
