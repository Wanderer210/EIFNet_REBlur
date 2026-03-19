import os
import glob
import math
import random
import logging
import bisect
import h5py
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

import utils


def binary_events_to_voxel_grid(events, num_bins, width, height):
    """
    events: [N, 4] -> [t, x, y, p]
    return: [num_bins, H, W]
    """
    assert events.shape[1] == 4
    assert num_bins > 0
    assert width > 0 and height > 0

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    if len(events) == 0:
        return np.zeros((num_bins, height, width), dtype=np.float32)

    events = events.copy()

    first_stamp = events[0, 0]
    last_stamp = events[-1, 0]
    deltaT = last_stamp - first_stamp
    if deltaT == 0:
        deltaT = 1.0

    # 时间归一化到 [0, num_bins - 1]
    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT

    ts = events[:, 0]
    xs = events[:, 1].astype(np.int64)
    ys = events[:, 2].astype(np.int64)
    pols = events[:, 3].astype(np.float32)

    # 0/1 极性转成 -1/+1
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


def try_read_frame_timestamps(h5_file):
    """
    尝试读取图像帧时间戳。
    由于你提供的 h5ls 没显示时间戳，这里做鲁棒兼容。
    如果找不到，则返回 None。
    """
    candidate_keys = [
        "image_ts",
        "images_ts",
        "image_timestamps",
        "images_timestamps",
        "frame_ts",
        "frame_timestamps",
        "timestamps",
        "img_ts",
    ]

    # 先查 dataset
    for k in candidate_keys:
        if k in h5_file:
            arr = np.asarray(h5_file[k])
            if arr.ndim == 1:
                return arr

    # 再查 attrs
    for k in candidate_keys:
        if k in h5_file.attrs:
            arr = np.asarray(h5_file.attrs[k])
            if arr.ndim == 1:
                return arr

    # 有些数据集可能把时间戳存在 images group 的 attrs 中
    if "images" in h5_file:
        g = h5_file["images"]
        for k in candidate_keys:
            if k in g.attrs:
                arr = np.asarray(g.attrs[k])
                if arr.ndim == 1:
                    return arr

    return None


def get_event_indices_by_equal_split(num_events, num_frames):
    """
    当没有帧时间戳时，按帧数把整个事件流均匀切分。
    返回长度为 num_frames+1 的边界数组：
    idx[i] ~ idx[i+1] 对应第 i 帧的事件窗口
    """
    boundaries = np.linspace(0, num_events, num_frames + 1, dtype=np.int64)
    return boundaries


def get_event_indices_by_timestamps(event_ts, frame_ts):
    """
    根据帧时间戳，把事件按时间划分到每一帧。
    返回长度为 num_frames+1 的边界数组。
    第 i 帧对应 [boundary[i], boundary[i+1]) 之间的事件。

    这里采用相邻帧时间中点作为分界。
    """
    frame_ts = np.asarray(frame_ts).reshape(-1)
    num_frames = len(frame_ts)

    if num_frames == 1:
        return np.array([0, len(event_ts)], dtype=np.int64)

    # 用相邻帧中点作为窗口分界
    mids = (frame_ts[:-1] + frame_ts[1:]) / 2.0

    boundaries = [event_ts[0] - 1e-6]
    boundaries.extend(list(mids))
    boundaries.append(event_ts[-1] + 1e-6)

    event_indices = []
    for b in boundaries:
        idx = np.searchsorted(event_ts, b, side='left')
        event_indices.append(idx)

    event_indices = np.array(event_indices, dtype=np.int64)
    event_indices[0] = 0
    event_indices[-1] = len(event_ts)
    return event_indices


def read_h5_image(group, key):
    """
    读取 HWC uint8 图像，转成 CHW float32 [0,1]
    """
    img = np.asarray(group[key])
    if img.ndim != 3:
        raise ValueError(f"Unexpected image shape for key={key}: {img.shape}")

    # 数据格式是 {H, W, 3}
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # CHW
    return img


class REBlurH5Base(object):
    def __init__(self, h5_files, args):
        self.h5_files = sorted(h5_files)
        self.args = args
        self.num_bins = args.num_bins

        self.samples = []
        self.seq_info = {}

        for file_idx, h5_path in enumerate(self.h5_files):
            with h5py.File(h5_path, 'r') as f:
                if 'images' not in f or 'sharp_images' not in f or 'events' not in f:
                    raise ValueError(f"{h5_path} missing required groups: images/sharp_images/events")

                blur_keys = sorted(list(f['images'].keys()))
                sharp_keys = sorted(list(f['sharp_images'].keys()))
                if len(blur_keys) != len(sharp_keys):
                    raise ValueError(f"{h5_path}: images and sharp_images count mismatch")

                num_frames = len(blur_keys)

                # 从第一张图推断 H/W
                first_img = np.asarray(f['images'][blur_keys[0]])
                h, w = first_img.shape[0], first_img.shape[1]

                event_ts = np.asarray(f['events']['ts']).reshape(-1)
                event_xs = np.asarray(f['events']['xs']).reshape(-1)
                event_ys = np.asarray(f['events']['ys']).reshape(-1)
                event_ps = np.asarray(f['events']['ps']).reshape(-1)

                if not (len(event_ts) == len(event_xs) == len(event_ys) == len(event_ps)):
                    raise ValueError(f"{h5_path}: event fields length mismatch")

                frame_ts = try_read_frame_timestamps(f)

                if frame_ts is not None and len(frame_ts) == num_frames:
                    event_boundaries = get_event_indices_by_timestamps(event_ts, frame_ts)
                    split_mode = "timestamp"
                else:
                    event_boundaries = get_event_indices_by_equal_split(len(event_ts), num_frames)
                    split_mode = "equal_split"

                self.seq_info[file_idx] = {
                    "path": h5_path,
                    "num_frames": num_frames,
                    "height": h,
                    "width": w,
                    "blur_keys": blur_keys,
                    "sharp_keys": sharp_keys,
                    "event_boundaries": event_boundaries,
                    "split_mode": split_mode,
                }

                for frame_idx in range(num_frames):
                    self.samples.append((file_idx, frame_idx))

        logging.info(f"[REBlurH5Base] total files: {len(self.h5_files)}")
        logging.info(f"[REBlurH5Base] total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def load_sample(self, file_idx, frame_idx):
        info = self.seq_info[file_idx]
        h5_path = info["path"]

        with h5py.File(h5_path, 'r') as f:
            blur_img = read_h5_image(f['images'], info["blur_keys"][frame_idx])
            sharp_img = read_h5_image(f['sharp_images'], info["sharp_keys"][frame_idx])

            left = info["event_boundaries"][frame_idx]
            right = info["event_boundaries"][frame_idx + 1]

            ts = np.asarray(f['events']['ts'][left:right], dtype=np.float32)
            xs = np.asarray(f['events']['xs'][left:right], dtype=np.float32)
            ys = np.asarray(f['events']['ys'][left:right], dtype=np.float32)
            ps = np.asarray(f['events']['ps'][left:right], dtype=np.float32)

            if len(ts) == 0:
                event_voxel = np.zeros((self.num_bins, info["height"], info["width"]), dtype=np.float32)
            else:
                event_window = np.stack([ts, xs, ys, ps], axis=1)
                event_voxel = binary_events_to_voxel_grid(
                    event_window,
                    num_bins=self.num_bins,
                    width=info["width"],
                    height=info["height"]
                )

        return blur_img, event_voxel, sharp_img


class DataLoaderTrain_REBlur_h5(Dataset, REBlurH5Base):
    """
    训练集：
    返回 (input_img, input_event, target)
    并调用 utils.image_proess 做随机裁剪/增强
    """
    def __init__(self, h5_files, args):
        Dataset.__init__(self)
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
    """
    验证集：
    返回 torch tensor，不做随机 crop
    """
    def __init__(self, h5_files, args):
        Dataset.__init__(self)
        REBlurH5Base.__init__(self, h5_files, args)

    def __getitem__(self, index):
        file_idx, frame_idx = self.samples[index]
        blur_img, event_voxel, sharp_img = self.load_sample(file_idx, frame_idx)

        blur_img = torch.from_numpy(blur_img).float()
        event_voxel = torch.from_numpy(event_voxel).float()
        sharp_img = torch.from_numpy(sharp_img).float()

        return blur_img, event_voxel, sharp_img


class DataLoaderTest_REBlur_h5(Dataset):
    """
    测试指定单个 h5 文件
    """
    def __init__(self, h5_path, args):
        super().__init__()
        self.args = args
        self.num_bins = args.num_bins
        self.h5_path = h5_path

        with h5py.File(h5_path, 'r') as f:
            self.blur_keys = sorted(list(f['images'].keys()))
            self.sharp_keys = sorted(list(f['sharp_images'].keys()))
            self.num_frames = len(self.blur_keys)

            first_img = np.asarray(f['images'][self.blur_keys[0]])
            self.height, self.width = first_img.shape[0], first_img.shape[1]

            event_ts = np.asarray(f['events']['ts']).reshape(-1)
            frame_ts = try_read_frame_timestamps(f)

            if frame_ts is not None and len(frame_ts) == self.num_frames:
                self.event_boundaries = get_event_indices_by_timestamps(event_ts, frame_ts)
                self.split_mode = "timestamp"
            else:
                self.event_boundaries = get_event_indices_by_equal_split(len(event_ts), self.num_frames)
                self.split_mode = "equal_split"

        print(f"[DataLoaderTest_REBlur_h5] file={h5_path}")
        print(f"[DataLoaderTest_REBlur_h5] num_frames={self.num_frames}, split_mode={self.split_mode}")

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        with h5py.File(self.h5_path, 'r') as f:
            blur_img = read_h5_image(f['images'], self.blur_keys[index])
            sharp_img = read_h5_image(f['sharp_images'], self.sharp_keys[index])

            left = self.event_boundaries[index]
            right = self.event_boundaries[index + 1]

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
    total_samples = opts.train_iters * opts.OPTIM.BATCH_SIZE
    num_epochs = int(math.ceil(float(total_samples) / len(data_set)))

    indices = np.random.permutation(len(data_set))
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


def collect_h5_files(root_or_list):
    """
    输入可以是：
    1. h5 文件列表
    2. 目录路径（自动搜 *.h5）
    """
    if isinstance(root_or_list, (list, tuple)):
        return sorted(root_or_list)

    if os.path.isdir(root_or_list):
        return sorted(glob.glob(os.path.join(root_or_list, "*.h5")))

    if os.path.isfile(root_or_list) and root_or_list.endswith(".h5"):
        return [root_or_list]

    raise ValueError(f"Invalid input for collect_h5_files: {root_or_list}")