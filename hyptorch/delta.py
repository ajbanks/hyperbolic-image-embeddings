import torch
import torch.nn as nn
import torchvision
from scipy.spatial import distance_matrix
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def delta_hyp(dismat):
    """
    computes delta hyperbolicity value from distance matrix
    """

    p = 0
    # [0]  # means line 0 of your matrix
    # [(0, 0)]  # means cell at 0,0 of your matrix
    # [0: 1]  # means lines 0 to 1 excluded of your matrix
    # [: 1]  # excluding the first value means all lines until line 1 excluded
    # [1:]  # excluding the last param mean all lines starting form line 1 included
    # [:]  # excluding both means all lines
    # [::2]  # the addition of a second ':' is the sampling. (1 item every 2)
    # [::]  # exluding it means a sampling of 1
    # [:,:]  # simply uses a tuple (a single , represents an empty tuple) instead of an index.
    # np.newaxis adds a new axis to an arrway, essential giving it another dimention. a 1D rray will become 2D for example
    # https://stackoverflow.com/questions/29241056/how-does-numpy-newaxis-work-and-when-to-use-it

    row = dismat[p, :][np.newaxis, :]
    print("\n\n extracted row from dismat: ")
    print(row)

    col = dismat[:, p][:, np.newaxis]
    print("\n\n extracted column from dismat: ")
    print(col)

    XY_p = 0.5 * (row + col - dismat)
    print("\n\n performe calc: 0.5 * (row + col - dismat): \n")
    print(XY_p)

    minimum = np.minimum(XY_p[:, :, None], XY_p[None, :, :])
    print("\n\n performe calc: np.minimum(XY_p[:, :, None], XY_p[None, :, :]): \n")
    print(minimum)


    maxmin = np.max(minimum, axis=1)
    print("\n\n performe calc: np.max(minimum, axis=1): \n")
    print(maxmin)

    # maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)
    print("\n\n calculated maxmin")

    return np.max(maxmin - XY_p)


def batched_delta_hyp(X, n_tries=5, batch_size=1500):
    vals = []
    for i in tqdm(range(n_tries)):
        idx = np.random.choice(len(X), batch_size)
        X_batch = X[idx]
        distmat = distance_matrix(X_batch, X_batch)
        print("calcularted batch distance matrix \n")
        diam = np.max(distmat)
        delta_rel = delta_hyp(distmat) / diam
        print("\n\n calcularted batch delta: " + str(delta_rel))
        vals.append(delta_rel)
    return np.mean(vals) #, np.std(vals)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B = x.shape[0]
        return x.view(B, -1)


def get_delta(loader):
    """
    computes delta value for image data by extracting features using VGG network;
    input -- data loader for images
    """
    vgg = torchvision.models.vgg16(pretrained=True)
    vgg_feats = vgg.features
    vgg_classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])

    vgg_part = nn.Sequential(vgg_feats, Flatten(), vgg_classifier).to(device)
    vgg_part.eval()
    print("\n\n Loaded vgg \n\n")
    all_features = []
    for i, (batch, _) in enumerate(loader):
        with torch.no_grad():
            batch = batch.to(device)
            print([len(batch[0]), len(batch[0][1]), len(batch)])
            print(batch.shape)
            #[all_features.append([torch.flatten(bat).cpu().numpy()]) for bat in batch]
            all_features.append(vgg_part(batch).detach().cpu().numpy())
        print("processed " + str(i))

    # for data in loader:
    #     all_features.append(data[0].cpu().numpy())

    print("\n\n all features processed \n\n")

    all_features = np.concatenate(all_features)
    idx = np.random.choice(len(all_features), 50)
    all_features_small = all_features[idx]
    print(all_features.shape)
    # print(len(all_features[0]))
    # print(all_features_small[1499])
    # print(len(all_features_small))
    # print(len(all_features_small[0]))
    # print(len(all_features_small[1499]))
    print("\n\n calculating distance matrix \n\n")
    # dists = distance_matrix(all_features_small, all_features_small)
    # print(dists)
    print("\n\n calculating hyperbolicity \n\n")
    delta = batched_delta_hyp(all_features, batch_size=len(all_features))
    print("\n\n calculated hyperbolicity \n\n")
    # diam = np.max(dists)
    return delta #, diam

test_matrix = [[0,0],[0,1], [1,0],[1,1]]
test_matrix = np.random.rand(224,224)
print(test_matrix)
test_delta = delta_hyp(
    distance_matrix(test_matrix, test_matrix))
print("the test hyperbolicity is: " + str(test_delta))

