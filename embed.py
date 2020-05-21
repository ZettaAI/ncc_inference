import cc3d
import fastremap
import numpy as np
import time
from helpers import coarsen_mask

import torch
from helpers import create_model, get_np, normalize


def embed_cv(src_cv, x, y, z, model_name):
    data = np.array(src_cv[x[0]:x[1], y[0]:y[1], z]).squeeze()

    data = torch.cuda.FloatTensor(data)

    encoder = create_model(
        "model", checkpoint_folder="./models/{}".format(model_name)
    )

    processed_data = encoder(
        data.unsqueeze(0).unsqueeze(0)
    ).squeeze()
    processed_data[data == 0] = 0
    processed_data[data == -20] = 0
    result = get_np(processed_data).astype(
        np.float32
    )

    return result




