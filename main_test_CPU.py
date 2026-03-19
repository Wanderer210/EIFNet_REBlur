import os
import random
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

from config import Config
import utils
from dataset_RGB import *
from U_model import unet
from warmup_scheduler import GradualWarmupScheduler

CONFIG_YAML = 'training.yml'


def _read_yaml_scalar(path, key):
    try:
        with open(path, 'r') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith('#'):
                    line = line[1:].strip()
                if not line.startswith(f'{key}:'):
                    continue
                value = line.split(':', 1)[1].strip()
                if (len(value) >= 2) and ((value[0] == value[-1]) and value[0] in ("'", '"')):
                    value = value[1:-1]
                return value
    except OSError:
        return None
    return None


def _load_checkpoint(model, weights, device):
    checkpoint = torch.load(weights, map_location=device)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    return checkpoint


opt = Config(CONFIG_YAML)

test_root = _read_yaml_scalar(CONFIG_YAML, 'father_test_path_npz')
if not test_root:
    test_root = '/home/zy/data/zy/zhaoyue/Datasets/EIFNet_Gopro/test/'
opt.father_test_path_npz = test_root

result_root = _read_yaml_scalar(CONFIG_YAML, 'result_dir')
if result_root:
    opt.result_dir = result_root

# =========================
# 强制使用 CPU
# =========================
device = torch.device('cpu')
print('Using device:', device)

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)


def main():
    start_epoch = 1
    session = opt.MODEL.SESSION

    result_dir = os.path.join(opt.TRAINING.SAVE_DIR, 'results', session)
    model_dir = os.path.join(opt.TRAINING.SAVE_DIR, 'models', session)

    utils.mkdir(result_dir)
    utils.mkdir(model_dir)

    if getattr(opt, 'result_dir', None) in [None, './output', 'output']:
        opt.result_dir = result_dir

    ######### Model ###########
    model_restoration = unet.Restoration(3, 6, 3, opt)
    model_restoration = model_restoration.to(device)

    new_lr = opt.OPTIM.LR_INITIAL
    optimizer = optim.Adam(
        model_restoration.parameters(),
        lr=new_lr,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        opt.OPTIM.NUM_EPOCHS - warmup_epochs,
        eta_min=opt.OPTIM.LR_MIN
    )
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1,
        total_epoch=warmup_epochs,
        after_scheduler=scheduler_cosine
    )

    # 这里只是为了兼容你原来的逻辑，测试时其实 scheduler/optimizer 基本无关紧要
    scheduler.step()

    ######### Resume ###########
    if opt.TESTING.RESUME:
        path_chk_rest = utils.get_last_path(model_dir, '_best_psnr.pth')
        if len(path_chk_rest) == 0:
            path_chk_rest = utils.get_last_path(model_dir, '_best_ssim.pth')
        if len(path_chk_rest) == 0:
            path_chk_rest = utils.get_last_path(model_dir, '_best.pth')
        if len(path_chk_rest) == 0:
            path_chk_rest = utils.get_last_path(model_dir, 'latest.pth')
        if len(path_chk_rest) == 0:
            raise FileNotFoundError(f'No checkpoint found in: {model_dir}')

        print('path_chk_rest', path_chk_rest)
        checkpoint = _load_checkpoint(model_restoration, path_chk_rest[0], device)
        start_epoch = int(checkpoint.get('epoch', 0)) + 1

        if 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                print('Warning: failed to load optimizer state on CPU:', e)

        for _ in range(start_epoch):
            scheduler.step()

        try:
            new_lr = scheduler.get_last_lr()[0]
        except Exception:
            new_lr = scheduler.get_lr()[0]

        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    ## data prepare
    test_files_dirs = sorted(os.listdir(os.path.join(opt.father_test_path_npz, 'blur')))

    ######### DataLoaders ###########
    print('===> Loading datasets')

    epoch = 0

    #### Evaluation ####
    if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        ssim_val_rgb = []

        for test_file in test_files_dirs:
            single_psnr_val_rgb = []
            single_ssim_val_rgb = []

            out_path = os.path.join(opt.result_dir, test_file)
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            val_dataset = DataLoaderTest_npz(opt.father_test_path_npz, test_file, opt)
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,      # CPU 下建议 0，最稳
                drop_last=False,
                pin_memory=False    # CPU 下不需要
            )

            for ii, data_val in enumerate(tqdm(val_loader), 0):
                input_img = data_val[0].to(device)
                input_event = data_val[1].to(device)
                input_target = data_val[2].to(device)

                with torch.no_grad():
                    restored = model_restoration(input_img, input_event)

                res = torch.clamp(restored, 0, 1)[0, :, :, :]
                tar = input_target[0, :, :, :]

                input1 = res.cpu().numpy().transpose([1, 2, 0])
                input2 = tar.cpu().numpy().transpose([1, 2, 0])

                ssim_rgb = SSIM(input1, input2, multichannel=True)
                single_ssim_val_rgb.append(ssim_rgb)
                ssim_val_rgb.append(ssim_rgb)

                psnr_rgb = PSNR(input1, input2)
                single_psnr_val_rgb.append(psnr_rgb)
                psnr_val_rgb.append(psnr_rgb)

                output = restored[0, :, :, :] * 255
                output.clamp_(0.0, 255.0)
                output = output.byte().cpu().numpy().transpose([1, 2, 0])

                fname = f"{ii:04d}_psnr{psnr_rgb:.2f}_ssim{ssim_rgb:.4f}.png"
                cv2.imwrite(os.path.join(out_path, fname), output)

            print("Name: %s PSNR: %.4f SSIM: %.4f" % (
                test_file,
                np.mean(single_psnr_val_rgb),
                np.mean(single_ssim_val_rgb))
            )

        ssim_val_rgb = np.mean(ssim_val_rgb)
        psnr_val_rgb = np.mean(psnr_val_rgb)

        print('ALL_SSIM', ssim_val_rgb)
        print('ALL_PSNR', psnr_val_rgb)


if __name__ == '__main__':
    main()