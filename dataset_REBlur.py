import os
import glob
import math
import random
import logging
import h5py
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import cv2
import utils


def binary_events_to_voxel_grid(events, num_bins, width, height):
    """
    Build voxel grid from raw events.
    events: [N, 4] = [timestamp, x, y, polarity]
    return: [num_bins, H, W]
    """
    assert events.shape[1] == 4
    assert num_bins > 0
    assert width > 0
    assert height > 0

    if len(events) == 0:
        return np.zeros((num_bins, height, width), dtype=np.float32)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    events = events.copy()

    first_stamp = events[0, 0]
    last_stamp = events[-1, 0]
    deltaT = last_stamp - first_stamp
    if deltaT == 0:
        deltaT = 1.0

    # normalize timestamps to [0, num_bins - 1]
    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT

    ts = events[:, 0]
    xs = events[:, 1].astype(np.int64)
    ys = events[:, 2].astype(np.int64)
    pols = events[:, 3].astype(np.float32)

    # convert polarity from {0,1} to {-1,+1}
    pols[pols == 0] = -1

    tis = ts.astype(np.int64)
    dts = ts - tis

    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_left = (tis >= 0) & (tis < num_bins) & \
                 (xs >= 0) & (xs < width) & \
                 (ys >= 0) & (ys < height)

    np.add.at(
        voxel_grid,
        xs[valid_left] + ys[valid_left] * width + tis[valid_left] * width * height,
        vals_left[valid_left]
    )

    valid_right = ((tis + 1) >= 0) & ((tis + 1) < num_bins) & \
                  (xs >= 0) & (xs < width) & \
                  (ys >= 0) & (ys < height)

    np.add.at(
        voxel_grid,
        xs[valid_right] + ys[valid_right] * width + (tis[valid_right] + 1) * width * height,
        vals_right[valid_right]
    )

    voxel_grid = voxel_grid.reshape(num_bins, height, width)
    return voxel_grid.astype(np.float32)


def read_h5_image(group, key):
    """
    HWC uint8 -> CHW float32 [0,1]
    """
    img = np.asarray(group[key])
    if img.ndim != 3:
        raise ValueError(f"Unexpected image shape: {img.shape}, key={key}")

    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return img


def collect_h5_files(path_or_list):
    if isinstance(path_or_list, (list, tuple)):
        return sorted(path_or_list)

    if os.path.isdir(path_or_list):
        return sorted(glob.glob(os.path.join(path_or_list, '*.h5')))

    if os.path.isfile(path_or_list) and path_or_list.endswith('.h5'):
        return [path_or_list]

    raise ValueError(f'Invalid input: {path_or_list}')


class REBlurH5Base(object):
    """
    基类：
    1. 扫描所有 h5 文件
    2. 建立 (file_idx, frame_idx) -> sample 的索引
    3. 每个文件内部按帧数均匀切分事件流
    """
    def __init__(self, h5_files, args):
        self.h5_files = sorted(h5_files)
        self.args = args
        self.num_bins = args.num_bins

        self.samples = []
        self.files_info = {}

        for file_idx, h5_path in enumerate(self.h5_files):
            with h5py.File(h5_path, 'r') as f:
                if 'images' not in f:
                    raise ValueError(f'{h5_path} missing /images')
                if 'sharp_images' not in f:
                    raise ValueError(f'{h5_path} missing /sharp_images')
                if 'events' not in f:
                    raise ValueError(f'{h5_path} missing /events')

                blur_keys = sorted(list(f['images'].keys()))
                sharp_keys = sorted(list(f['sharp_images'].keys()))

                if len(blur_keys) != len(sharp_keys):
                    raise ValueError(f'{h5_path}: images and sharp_images count mismatch')

                num_frames = len(blur_keys)

                # infer H/W from first blur image
                first_img = np.asarray(f['images'][blur_keys[0]])
                height, width = first_img.shape[0], first_img.shape[1]

                # only total event stream exists
                num_events = len(f['events']['ts'])

                # equal split: num_frames images -> num_frames event windows
                boundaries = np.linspace(0, num_events, num_frames + 1, dtype=np.int64)

                self.files_info[file_idx] = {
                    'path': h5_path,
                    'blur_keys': blur_keys,
                    'sharp_keys': sharp_keys,
                    'num_frames': num_frames,
                    'height': height,
                    'width': width,
                    'boundaries': boundaries,
                    'num_events': num_events,
                }

                for frame_idx in range(num_frames):
                    self.samples.append((file_idx, frame_idx))

        logging.info(f'Loaded {len(self.h5_files)} h5 files')
        logging.info(f'Total samples: {len(self.samples)}')

    def __len__(self):
        return len(self.samples)

    def load_sample(self, file_idx, frame_idx):
        info = self.files_info[file_idx]
        h5_path = info['path']

        with h5py.File(h5_path, 'r') as f:
            blur_img = read_h5_image(f['images'], info['blur_keys'][frame_idx])
            sharp_img = read_h5_image(f['sharp_images'], info['sharp_keys'][frame_idx])

            left = info['boundaries'][frame_idx]
            right = info['boundaries'][frame_idx + 1]

            ts = np.asarray(f['events']['ts'][left:right], dtype=np.float32)
            xs = np.asarray(f['events']['xs'][left:right], dtype=np.float32)
            ys = np.asarray(f['events']['ys'][left:right], dtype=np.float32)
            ps = np.asarray(f['events']['ps'][left:right], dtype=np.float32)

            if len(ts) == 0:
                event_voxel = np.zeros((self.num_bins, info['height'], info['width']), dtype=np.float32)
            else:
                event_window = np.stack([ts, xs, ys, ps], axis=1)
                event_voxel = binary_events_to_voxel_grid(
                    event_window,
                    num_bins=self.num_bins,
                    width=info['width'],
                    height=info['height']
                )

        return blur_img, event_voxel, sharp_img


class DataLoaderTrain_REBlur_h5(Dataset, REBlurH5Base):
    def __init__(self, rgb_dir, args):
        Dataset.__init__(self)
        h5_files = collect_h5_files(rgb_dir)
        REBlurH5Base.__init__(self, h5_files, args)

    def __getitem__(self, index):
        file_idx, frame_idx = self.samples[index]
        blur_img, event_voxel, sharp_img = self.load_sample(file_idx, frame_idx)

        input_img, input_event, target = utils.image_proess(
            blur_img,
            event_voxel,
            sharp_img,
            self.args.TRAINING.TRAIN_PS,
            self.args
        )

        return input_img, input_event, target


class DataLoaderVal_REBlur_h5(Dataset, REBlurH5Base):
    def __init__(self, rgb_dir, args):
        Dataset.__init__(self)
        h5_files = collect_h5_files(rgb_dir)
        REBlurH5Base.__init__(self, h5_files, args)

    def __getitem__(self, index):
        file_idx, frame_idx = self.samples[index]
        blur_img, event_voxel, sharp_img = self.load_sample(file_idx, frame_idx)

        blur_img = torch.from_numpy(blur_img).float()
        event_voxel = torch.from_numpy(event_voxel).float()
        sharp_img = torch.from_numpy(sharp_img).float()

        return blur_img, event_voxel, sharp_img


class DataLoaderTest_REBlur_h5(Dataset):
    def __init__(self, h5_path, args):
        super().__init__()
        self.h5_path = h5_path
        self.args = args
        self.num_bins = args.num_bins

        with h5py.File(h5_path, 'r') as f:
            self.blur_keys = sorted(list(f['images'].keys()))
            self.sharp_keys = sorted(list(f['sharp_images'].keys()))
            self.num_frames = len(self.blur_keys)

            first_img = np.asarray(f['images'][self.blur_keys[0]])
            self.height, self.width = first_img.shape[0], first_img.shape[1]

            self.num_events = len(f['events']['ts'])
            self.boundaries = np.linspace(0, self.num_events, self.num_frames + 1, dtype=np.int64)

        print(f'[DataLoaderTest_REBlur_h5] file={h5_path}')
        print(f'[DataLoaderTest_REBlur_h5] frames={self.num_frames}, total_events={self.num_events}')

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        with h5py.File(self.h5_path, 'r') as f:
            blur_img = read_h5_image(f['images'], self.blur_keys[index])
            sharp_img = read_h5_image(f['sharp_images'], self.sharp_keys[index])

            left = self.boundaries[index]
            right = self.boundaries[index + 1]

            ts = np.asarray(f['events']['ts'][left:right], dtype=np.float32)
            xs = np.asarray(f['events']['xs'][left:right], dtype=np.float32)
            ys = np.asarray(f['events']['ys'][left:right], dtype=np.float32)
            ps = np.asarray(f['events']['ps'][left:right], dtype=np.float32)

            if len(ts) == 0:
                event_voxel = np.zeros((self.num_bins, self.height, self.width), dtype=np.float32)
            else:
                event_window = np.stack([ts, xs, ys, ps], axis=1)
                event_voxel = binary_events_to_voxel_grid(
                    event_window,
                    num_bins=self.num_bins,
                    width=self.width,
                    height=self.height
                )

        blur_img = torch.from_numpy(blur_img).float()
        event_voxel = torch.from_numpy(event_voxel).float()
        sharp_img = torch.from_numpy(sharp_img).float()

        return blur_img, event_voxel, sharp_img


class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def create_data_loader(data_set, opts, mode='train'):
    dataset_len = len(data_set)
    if dataset_len == 0:
        raise ValueError(f'Empty dataset for mode={mode}. Please check data paths and preprocessing outputs.')

    total_samples = opts.train_iters * opts.OPTIM.BATCH_SIZE
    num_epochs = int(math.ceil(float(total_samples) / dataset_len))

    indices = np.random.permutation(dataset_len)
    indices = np.tile(indices, num_epochs)
    indices = indices[:total_samples]

    sampler = SubsetSequentialSampler(indices)
    data_loader = DataLoader(
        dataset=data_set,
        num_workers=4,
        batch_size=opts.OPTIM.BATCH_SIZE,
        sampler=sampler,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )
    return data_loader


import glob

class DataLoaderTrain_Fast(Dataset):
    """极速版训练集：直接读取预计算的 npy 和 png，无需实时计算 voxel"""
    def __init__(self, processed_dir, args):
        self.args = args
        self.samples = []
        seq_dirs = sorted(glob.glob(os.path.join(processed_dir, '*')))
        
        for seq in seq_dirs:
            if not os.path.isdir(seq): continue
            blur_files = sorted(glob.glob(os.path.join(seq, 'blur', '*.png')))
            for b_file in blur_files:
                frame_name = os.path.basename(b_file).replace('.png', '')
                s_file = os.path.join(seq, 'sharp', f"{frame_name}.png")
                v_file = os.path.join(seq, 'voxel', f"{frame_name}.npy")
                
                if os.path.exists(s_file) and os.path.exists(v_file):
                    self.samples.append((b_file, v_file, s_file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        b_path, v_path, s_path = self.samples[idx]

        # 读取图像并转换回 RGB 和 CHW 的 Float32 格式
        blur_img = cv2.imread(b_path)
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blur_img = blur_img.transpose(2, 0, 1)

        sharp_img = cv2.imread(s_path)
        sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        sharp_img = sharp_img.transpose(2, 0, 1)

        # 极速读取体素张量 (无需通过复杂的插值计算)
        event_voxel = np.load(v_path)

        # 进行随机裁剪、翻转等数据增强操作
        input_img, input_event, target = utils.image_proess(
            blur_img, event_voxel, sharp_img, self.args.TRAINING.TRAIN_PS, self.args
        )
        return input_img, input_event, target

class DataLoaderVal_Fast(Dataset):
    """极速版验证集：返回全尺寸 Tensor，不进行裁切增强"""
    def __init__(self, processed_dir, args):
        # 初始化逻辑与 DataLoaderTrain_Fast 完全一致
        self.args = args
        self.samples = []
        seq_dirs = sorted(glob.glob(os.path.join(processed_dir, '*')))
        for seq in seq_dirs:
            if not os.path.isdir(seq): continue
            blur_files = sorted(glob.glob(os.path.join(seq, 'blur', '*.png')))
            for b_file in blur_files:
                frame_name = os.path.basename(b_file).replace('.png', '')
                s_file = os.path.join(seq, 'sharp', f"{frame_name}.png")
                v_file = os.path.join(seq, 'voxel', f"{frame_name}.npy")
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