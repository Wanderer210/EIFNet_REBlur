import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from dataset_REBlur import REBlurH5Base, collect_h5_files

class DummyArgs:
    def __init__(self, num_bins):
        self.num_bins = num_bins

def preprocess_dataset(h5_dir, output_dir, num_bins=6):
    h5_files = collect_h5_files(h5_dir)
    if not h5_files:
        print(f"在 {h5_dir} 中没有找到 H5 文件！")
        return
        
    args = DummyArgs(num_bins=num_bins)
    dataset = REBlurH5Base(h5_files, args)

    os.makedirs(output_dir, exist_ok=True)
    print(f"开始处理 {h5_dir} -> {output_dir}")

    for idx in tqdm(range(len(dataset))):
        file_idx, frame_idx = dataset.samples[idx]
        # 直接通过 h5_files 列表获取当前文件路径
        seq_name = os.path.basename(dataset.h5_files[file_idx]).replace('.h5', '')
        
        seq_out_dir = os.path.join(output_dir, seq_name)
        blur_dir = os.path.join(seq_out_dir, 'blur')
        sharp_dir = os.path.join(seq_out_dir, 'sharp')
        voxel_dir = os.path.join(seq_out_dir, 'voxel')

        os.makedirs(blur_dir, exist_ok=True)
        os.makedirs(sharp_dir, exist_ok=True)
        os.makedirs(voxel_dir, exist_ok=True)

        # 加载数据 (图像已归一化为 CHW 格式的 float32，voxel 已计算完毕)
        blur_img, event_voxel, sharp_img = dataset.load_sample(file_idx, frame_idx)

        # 将图像转换回 HWC, uint8 格式并转为 BGR (因为 cv2.imwrite 默认写入 BGR)
        blur_img_uint8 = (blur_img.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        blur_img_uint8 = cv2.cvtColor(blur_img_uint8, cv2.COLOR_RGB2BGR)
        
        sharp_img_uint8 = (sharp_img.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        sharp_img_uint8 = cv2.cvtColor(sharp_img_uint8, cv2.COLOR_RGB2BGR)

        # 保存为 png 和 npy
        cv2.imwrite(os.path.join(blur_dir, f"{frame_idx:04d}.png"), blur_img_uint8)
        cv2.imwrite(os.path.join(sharp_dir, f"{frame_idx:04d}.png"), sharp_img_uint8)
        
        # event_voxel 直接保存为 numpy 数组文件，加载速度极快
        np.save(os.path.join(voxel_dir, f"{frame_idx:04d}.npy"), event_voxel)

if __name__ == '__main__':
    # 请确保路径对应你真实的 REBlur 存放路径
    # 处理训练集
    preprocess_dataset('/home/zy/data/zy/zhaoyue/Datasets/EIFNet_REBlur/REBlur_rawevents/train', '/home/zy/data/zy/zhaoyue/Datasets/EIFNet_REBlur/REBlur_Fast/train')
    # 处理测试/验证集
    preprocess_dataset('/home/zy/data/zy/zhaoyue/Datasets/EIFNet_REBlur/REBlur_rawevents/test', '/home/zy/data/zy/zhaoyue/Datasets/EIFNet_REBlur/REBlur_Fast/test')
    print("全部预处理完成！")