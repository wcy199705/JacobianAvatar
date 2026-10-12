import pickle
import os
import torch
def save_smpl_params(save_path, smpl_params_batch):
    for key in smpl_params_batch.keys():
        smpl_params_batch[key] = smpl_params_batch[key].detach()
        smpl_params_batch[key].requires_grad =False
    with open(save_path, 'wb') as f:
        pickle.dump(smpl_params_batch, f)


def save_smpl_mesh(save_path, smpl_mesh):
    smpl_mesh.save_smpl_mesh(save_path)


def load_smpl_params(smpl_param_path):
    with open(smpl_param_path, 'rb') as f:
        smpl_params_batch = pickle.load(f)
    for key in smpl_params_batch.keys():
        smpl_params_batch[key].requires_grad =False
    return smpl_params_batch

def preprocessing_save(smpl_mesh_path,smpl_param_path, smpl_mesh=None, smpl_params_batch=None):
    if smpl_mesh is not None:
        save_smpl_mesh(smpl_mesh_path, smpl_mesh)
    if smpl_params_batch is not None:
        save_smpl_params(smpl_param_path, smpl_params_batch)

def preprocessing_load(smpl_mesh_path,smpl_param_path,smpl_mesh=None):

    smpl_params_batch = load_smpl_params(smpl_param_path)
    if smpl_mesh is not None:
        smpl_mesh.load_ckpt(smpl_mesh_path)
    else:
        return smpl_params_batch
    return smpl_mesh, smpl_params_batch




def save_optimizer_outputs(Avatar_save_path, smpl_batch_params, Avatar, optimizer_ckpt):
    save_dir_name = os.path.dirname(Avatar_save_path)
    exp_name = os.path.basename(Avatar_save_path)
    smpl_batch_save_path = os.path.join(save_dir_name,exp_name+'_smplx_param.pkl')
    ckpt_save_path = os.path.join(save_dir_name,exp_name+'_adam_ckpts.pth')
    Avatar.save_model(Avatar_save_path)
    torch.save(optimizer_ckpt, ckpt_save_path)
    save_smpl_params(smpl_batch_save_path,smpl_batch_params )


def load_optimizer_outputs(Avatar, Avatar_save_path):
    Avatar.load_model(Avatar_save_path)
    save_dir_name = os.path.dirname(Avatar_save_path)
    exp_name = os.path.basename(Avatar_save_path)
    smpl_batch_save_path = os.path.join(save_dir_name,exp_name+'_smplx_param.pkl')
    ckpt_save_path = os.path.join(save_dir_name,exp_name+'_adam_ckpts.pth')
    optimizer_ckpt = torch.load(ckpt_save_path)

    smpl_params_batch = load_smpl_params(smpl_batch_save_path)
    return smpl_params_batch, Avatar, optimizer_ckpt
