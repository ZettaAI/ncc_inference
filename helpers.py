import os
import artificery
import torch
import numpy as np

from scipy.ndimage import convolve

def coarsen_mask(mask, n=1, flip=True):
    kernel = np.ones((n, n))
    mask = convolve(mask, kernel) > 0
    mask = mask.astype(np.int16) > 0
    return mask

    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    assert isinstance(mask, np.ndarray)
    for _ in range(n):
        mask = convolve(mask, kernel) > 0
        mask = mask.astype(np.int16) > 0
    return mask



def get_np(pt):
    return pt.cpu().detach().numpy()


def normalize(
    img,
    per_feature_center=True,
    per_feature_var=False,
    eps=1e-5,
    mask=None,
    mask_fill=None,
):

    img_out = img.clone()
    if mask is not None:
        assert mask.shape == img.shape
    for i in range(1):
        for b in range(img.shape[0]):
            x = img_out[b]
            if per_feature_center and len(img.shape) == 4:
                for f in range(img.shape[1]):
                    if mask is not None:
                        m = mask[b, f]
                        x[f][m] = x[f][m].clone() - torch.mean(x[f][m].clone())
                    else:
                        x[f] = x[f].clone() - torch.mean(x[f].clone())
            else:
                if mask is not None:
                    m = mask[b]
                    x[m] = x[m].clone() - torch.mean(x[m].clone())
                else:
                    x[...] = x.clone() - torch.mean(x.clone())

            if per_feature_var and len(img.shape) == 4:
                for f in range(img.shape[1]):
                    if mask is not None:
                        m = mask[b, f]
                        var = torch.var(x[f][m].clone())
                        x[f][m] = x[f][m].clone() / (torch.sqrt(var) + eps)
                    else:
                        var = torch.var(x[f].clone())
                        x[f] = x[f].clone() / (torch.sqrt(var) + eps)
            else:
                if mask is not None:
                    m = mask[b]
                    var = torch.var(x[m].clone())
                    x[m] = x[m].clone() / (torch.sqrt(var) + eps)
                else:
                    var = torch.var(x.clone())
                    x[...] = x.clone() / (torch.sqrt(var) + eps)

    if mask is not None and mask_fill is not None:
        img_out[mask == False] = mask_fill

    return img_out


def create_model(name, checkpoint_folder):
    a = artificery.Artificery()

    spec_path = os.path.join(checkpoint_folder, "model_spec.json")
    my_p = a.parse(spec_path)

    checkpoint_path = os.path.join(checkpoint_folder, "{}.state.pth.tar".format(name))
    if os.path.isfile(checkpoint_path):
        my_p.load_state_dict(torch.load(checkpoint_path))
    my_p.name = name
    return my_p.cuda()
