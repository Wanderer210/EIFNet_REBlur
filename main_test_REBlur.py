import os
import cv2
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
from tqdm import tqdm

from config import Config
import utils
from U_model import unet
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

# 1. 修改配置文件为你正在使用的 REBlur 配置
CONFIG_YAML = 'REBlur_funtine.yml'

def _read_yaml_scalar(path, key):
    try:
        with open(path, 'r') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith('#'): continue
                if line.startswith('#'): line = line[1:].strip()
                if not line.startswith(f'{key}:'): continue
                value = line.split(':', 1)[1].strip()
                if (len(value) >= 2) and ((value[0] == value[-1]) and value[0] in ("'", '"')):
                    value = value[1:-1]
                return value
    except OSError:
        return None
    return None

opt = Config(CONFIG_YAML)

# 强制指向极速离线测试集目录
test_root = '/home/zy/data/zy/zhaoyue/Datasets/EIFNet_REBlur/REBlur_Fast/test'
opt.father_test_path_npz = test_root

result_root = _read_yaml_scalar(CONFIG_YAML, 'result_dir')
if result_root:
    opt.result_dir = result_root
else:
    opt.result_dir = os.path.join(opt.TRAINING.SAVE_DIR, 'results', opt.MODEL.SESSION)

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
torch.backends.cudnn.benchmark = True

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# 2. 专门为 REBlur 极速测试编写的数据集类 (按序列加载)
class DataLoaderTest_REBlur_Fast(Dataset):
    def __init__(self, seq_dir):
        self.samples = []
        blur_files = sorted(glob.glob(os.path.join(seq_dir, 'blur', '*.png')))
        for b_file in blur_files:
            frame_name = os.path.basename(b_file).replace('.png', '')
            s_file = os.path.join(seq_dir, 'sharp', f"{frame_name}.png")
            v_file = os.path.join(seq_dir, 'voxel', f"{frame_name}.npy")
            if os.path.exists(s_file) and os.path.exists(v_file):
                self.samples.append((b_file, v_file, s_file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        b_path, v_path, s_path = self.samples[idx]

        blur_img = cv2.imread(b_path)
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blur_img = blur_img.transpose(2, 0, 1)

        sharp_img = cv2.imread(s_path)
        sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        sharp_img = sharp_img.transpose(2, 0, 1)

        event_voxel = np.load(v_path)

        return torch.from_numpy(blur_img), torch.from_numpy(event_voxel), torch.from_numpy(sharp_img)

def main():
    session = opt.MODEL.SESSION
    result_dir = opt.result_dir
    model_dir = os.path.join(opt.TRAINING.SAVE_DIR, 'models', session)

    utils.mkdir(result_dir)

    ######### Model ###########
    model_restoration = unet.Restoration(3, 6, 3, opt)
    model_restoration.cuda()

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    ######### Load Checkpoint ###########
    path_chk_rest = utils.get_last_path(model_dir, '_best_psnr.pth')
    if len(path_chk_rest) == 0:
        path_chk_rest = utils.get_last_path(model_dir, 'model_latest.pth')
    if len(path_chk_rest) == 0:
        raise FileNotFoundError(f'No checkpoint found in: {model_dir}')

    print('==> Loading checkpoint:', path_chk_rest[0])
    utils.load_checkpoint(model_restoration, path_chk_rest[0])

    # 3. 同步训练代码中的 DataParallel Bug 修复补丁
    for module in model_restoration.modules():
        invalid_params = []
        for k, v in module._parameters.items():
            if not isinstance(v, nn.Parameter) and v is not None:
                invalid_params.append((k, v))
                
        for k, v in invalid_params:
            del module._parameters[k]
            if isinstance(v, nn.Module):
                module._modules[k] = v
            else:
                module.__dict__[k] = v

    if len(device_ids) > 1:
        model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

    model_restoration.eval()

    ## data prepare (获取所有序列文件夹)
    test_seq_dirs = sorted([d for d in glob.glob(os.path.join(test_root, '*')) if os.path.isdir(d)])
    if not test_seq_dirs:
        raise ValueError(f"在 {test_root} 下未找到测试序列文件夹！")

    print('===> Loading datasets from:', test_root)

    psnr_val_rgb = []
    ssim_val_rgb = []

    for seq_dir in test_seq_dirs:
        seq_name = os.path.basename(seq_dir)
        single_psnr_val_rgb = []
        single_ssim_val_rgb = []
        
        out_path = os.path.join(result_dir, seq_name)
        utils.mkdir(out_path)

        val_dataset = DataLoaderTest_REBlur_Fast(seq_dir)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
        
        print(f"Testing Sequence: {seq_name}")
        for ii, data_val in enumerate(tqdm(val_loader)):
            input_img = data_val[0].cuda()
            input_event = data_val[1].cuda()
            input_target = data_val[2].cuda()

            with torch.no_grad():
                restored = model_restoration(input_img, input_event)

            res = torch.clamp(restored, 0, 1)[0, :, :, :]
            tar = input_target[0, :, :, :]
            
            input1 = res.cpu().numpy().transpose([1, 2, 0])
            input2 = tar.cpu().numpy().transpose([1, 2, 0])

            # 4. 同步 SSIM 版本兼容性修复
            ssim_rgb = SSIM(input1, input2, channel_axis=2, data_range=1.0)
            single_ssim_val_rgb.append(ssim_rgb)
            ssim_val_rgb.append(ssim_rgb)

            psnr_rgb = PSNR(input1, input2)
            single_psnr_val_rgb.append(psnr_rgb)
            psnr_val_rgb.append(psnr_rgb)

            # 将网络输出转换回 uint8 [0, 255]
            output = restored[0, :, :, :] * 255
            output.clamp_(0.0, 255.0)
            output = output.byte().cpu().numpy().transpose([1, 2, 0]) # HWC, RGB

            # 5. RGB 转 BGR (供 OpenCV 正确保存颜色)
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            fname = f"{ii:04d}_psnr{psnr_rgb:.2f}_ssim{ssim_rgb:.4f}.png"
            cv2.imwrite(os.path.join(out_path, fname), output_bgr)

        print("Sequence: %s | PSNR: %.4f | SSIM: %.4f" % (seq_name, np.mean(single_psnr_val_rgb), np.mean(single_ssim_val_rgb)))

    print('================================================')
    print('ALL_TEST_SSIM:', np.mean(ssim_val_rgb))
    print('ALL_TEST_PSNR:', np.mean(psnr_val_rgb))

if __name__ == '__main__':
    main()