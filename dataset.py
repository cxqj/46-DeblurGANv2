import os
from copy import deepcopy
from functools import partial
from glob import glob
from hashlib import sha1
from typing import Callable, Iterable, Optional, Tuple

import cv2
import numpy as np
from glog import logger
from joblib import Parallel, cpu_count, delayed
from skimage.io import imread
from torch.utils.data import Dataset
from tqdm import tqdm

import aug

# 返回采样后的图片对
def subsample(data: Iterable, bounds: Tuple[float, float], hash_fn: Callable, n_buckets=100, salt='', verbose=True):  # bounds:(0,0.9)
    data = list(data)  # [(path_a,path_b),(path_a,path_b),....(path_a,path_b)]  2103x2
    # 将图片对编号随机置为0-100的整数
    buckets = split_into_buckets(data, n_buckets=n_buckets, salt=salt, hash_fn=hash_fn)  # (2103,)  [46,61,30,35,....,25,96]??

    lower_bound, upper_bound = [x * n_buckets for x in bounds]  # 0, 90.0
    msg = f'Subsampling buckets from {lower_bound} to {upper_bound}, total buckets number is {n_buckets}'
    if salt:
        msg += f'; salt is {salt}'
    if verbose:
        logger.info(msg)
    return np.array([sample for bucket, sample in zip(buckets, data) if lower_bound <= bucket < upper_bound])  # samples between 0-90

# 随机生成hash地址
def hash_from_paths(x: Tuple[str, str], salt: str = '') -> str:
    path_a, path_b = x
    names = ''.join(map(os.path.basename, (path_a, path_b)))  # 000047.png000047.png
    return sha1(f'{names}_{salt}'.encode()).hexdigest()


def split_into_buckets(data: Iterable, n_buckets: int, hash_fn: Callable, salt=''):
    hashes = map(partial(hash_fn, salt=salt), data)
    return np.array([int(x, 16) % n_buckets for x in hashes])


def _read_img(x: str):
    img = cv2.imread(x)   # (720,1280,3)
    if img is None:
        logger.warning(f'Can not read image {x} with OpenCV, switching to scikit-image')
        img = imread(x)
    return img


class PairedDataset(Dataset):
    def __init__(self,
                 files_a: Tuple[str],
                 files_b: Tuple[str],
                 transform_fn: Callable,
                 normalize_fn: Callable,
                 corrupt_fn: Optional[Callable] = None,
                 preload: bool = True,
                 preload_size: Optional[int] = 0,
                 verbose=True):

        assert len(files_a) == len(files_b)

        self.preload = preload   # False
        self.data_a = files_a   # list (258,)  ['/media/../000047.png',...]
        self.data_b = files_b   # list (258,)  ['/media/../000047.png',...]
        self.verbose = verbose   # True
        self.corrupt_fn = corrupt_fn
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        logger.info(f'Dataset has been created with {len(self.data_a)} samples')

        if preload:  # 是否预加载图片对
            preload_fn = partial(self._bulk_preload, preload_size=preload_size)
            if files_a == files_b:
                self.data_a = self.data_b = preload_fn(self.data_a)
            else:
                self.data_a, self.data_b = map(preload_fn, (self.data_a, self.data_b))
            self.preload = True

    def _bulk_preload(self, data: Iterable[str], preload_size: int):
        jobs = [delayed(self._preload)(x, preload_size=preload_size) for x in data]
        jobs = tqdm(jobs, desc='preloading images', disable=not self.verbose)
        return Parallel(n_jobs=cpu_count(), backend='threading')(jobs)

    @staticmethod
    def _preload(x: str, preload_size: int):
        img = _read_img(x)
        if preload_size:
            h, w, *_ = img.shape
            h_scale = preload_size / h
            w_scale = preload_size / w
            scale = max(h_scale, w_scale)
            img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
            assert min(img.shape[:2]) >= preload_size, f'weird img shape: {img.shape}'
        return img

    def _preprocess(self, img, res):  #通道转换
        def transpose(x):
            return np.transpose(x, (2, 0, 1))

        return map(transpose, self.normalize_fn(img, res))

    def __len__(self):
        return len(self.data_a)

    def __getitem__(self, idx):
        a, b = self.data_a[idx], self.data_b[idx]   # a: /media/cxq/Elements/dataset/GOPRO/train/GOPR0372_07_00/blur/000076.png  b: /media/cxq/Elements/dataset/GOPRO/train/GOPR0372_07_00/sharp/000076.png
        if not self.preload:
            a, b = map(_read_img, (a, b))  # (720,1280,3), (720,1280,3)
        a, b = self.transform_fn(a, b)  # (256,256,3), (256,256,3)
        if self.corrupt_fn is not None:
            a = self.corrupt_fn(a)
        a, b = self._preprocess(a, b)  # (3,256,256), (3,256,256)
        return {'a': a, 'b': b}

    @staticmethod
    def from_config(config):
        config = deepcopy(config)
        # 获取模糊和清晰所有图像的路径
        files_a, files_b = map(lambda x: sorted(glob(config[x], recursive=True)), ('files_a', 'files_b'))
        # 图像增强
        transform_fn = aug.get_transforms(size=config['size'], scope=config['scope'], crop=config['crop'])
        # 归一化操作
        normalize_fn = aug.get_normalize()
        #裁剪函数
        corrupt_fn = aug.get_corrupt_function(config['corrupt'])

        hash_fn = hash_from_paths
        # ToDo: add more hash functions
        verbose = config.get('verbose', True)  # True
        data = subsample(data=zip(files_a, files_b),
                         bounds=config.get('bounds', (0, 1)),
                         hash_fn=hash_fn,
                         verbose=verbose)   # (1886,2)

        files_a, files_b = map(list, zip(*data))

        return PairedDataset(files_a=files_a,
                             files_b=files_b,
                             preload=config['preload'],  # False
                             preload_size=config['preload_size'],  # 0
                             corrupt_fn=corrupt_fn,
                             normalize_fn=normalize_fn,
                             transform_fn=transform_fn,
                             verbose=verbose)
