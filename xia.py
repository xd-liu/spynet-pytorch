import os
import numpy as np
import cv2
import torch
import multiprocessing as mp
from torch.nn import functional as F

def readFlow(name):
    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)

def flow_warp(x, flo):
    """
    inverse warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(x.device)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(x.device)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid = torch.stack([
        2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0,
        2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    ],
                        dim=1)

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, mode='nearest', padding_mode='border')

    return output.squeeze().numpy()


def warping(fn_list):
    mask_fn, flow_fn, pred_fn = fn_list
    mask = cv2.imread(mask_fn, cv2.IMREAD_GRAYSCALE).astype(np.float)
    flow = readFlow(flow_fn)
    # new mask
    img = torch.Tensor(mask).unsqueeze(dim=0).unsqueeze(dim=1)
    flow = torch.Tensor(flow).permute(2, 0, 1).contiguous().unsqueeze(dim=0)

    new_mask = flow_warp(img, flow)
    print(pred_fn, new_mask.shape)
    if not os.path.isdir(os.path.split(pred_fn)[0]):
        os.makedirs(os.path.split(pred_fn)[0])
    a = cv2.imwrite(pred_fn, new_mask)
    print(a)


def main():
    gt_base = '/shared/xudongliu/code/semi-flow/mask'
    fl_base = '/shared/xudongliu/code/pytorch-spynet/predictions/bdd-KT-val'
    pd_base = 'pd_mask/bdd-KT-val'
    list_file = '/shared/xudongliu/code/pytorch-liteflownet/lists/seg_track_val_new.txt'
    
    try:
        if not os.path.exists(pd_base):
            os.makedirs(pd_base)
    except OSError as err:
        print(err)
    args = []

    with open(list_file) as f:
        pair_list = f.readlines()
    
    for i, line in enumerate(pair_list):
        gt_name = os.path.join(gt_base, line.strip(' \n').split(' ')[0].split('.')[0] + '.png')
        flow_name = os.path.join(fl_base, line.strip(' \n').split(' ')[0].split('.')[0] + '.flo')
        pd_name = os.path.join(pd_base, line.strip(' \n').split(' ')[1].split('.')[0] + '.png')
        args.append([gt_name, flow_name, pd_name])

    pool = mp.Pool(16)
    pool.map(warping, args)


if __name__ == "__main__":
    main()
