import torch
@torch.no_grad
def get_sampled_data(train_dataset, idxs,return_kpt=False):
    return ([train_dataset.images[i] for i in idxs], 
            [train_dataset.masks[i] for i in idxs], 
            [train_dataset.cameras[i] for i in idxs],
            train_dataset.normals[idxs] if train_dataset.normals is not None else None,
            train_dataset.target_kpts[idxs] if train_dataset.target_kpts is not None else None )
