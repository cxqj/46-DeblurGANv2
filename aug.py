from typing import List
"""
albumentations 是一个给予 OpenCV的快速训练数据增强库，拥有非常简单且强大的可以用于多种任务（分割、检测）的接口，易于定制且添加其他框架非常方便。
它可以对数据集进行逐像素的转换，如模糊、下采样、高斯造点、高斯模糊、动态模糊、RGB转换、随机雾化等；也可以进行空间转换（同时也会对目标进行转换），
如裁剪、翻转、随机裁剪等。
"""
import albumentations as albu


def get_transforms(size: int, scope: str = 'geometric', crop='random'):
    augs = {'strong': albu.Compose([albu.HorizontalFlip(),
                                    """
                                    随机放射变换（ShiftScaleRotate）
                                    该方法可以对图片进行平移（translate）、缩放（scale）和旋转（roatate），其含有以下参数：
                                    shift_limit：图片宽高的平移因子
                                    scale_limit：图片缩放因子
                                    rotate_limit：图片旋转范围
                                    p：使用此转换的概率，默认值为 0.5
                                    """
                                    albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=20, p=.4),
                                    """
                                    在医学影像问题中非刚体装换可以帮助增强数据。albumentations 中主要提供了以下几种非刚体变换类：
                                    ElasticTransform、GridDistortion 和 OpticalDistortion。
                                    """
                                    albu.ElasticTransform(),  
                                    albu.OpticalDistortion(), 
                                    #按照概率P随机执行变换中的一个
                                    albu.OneOf([
                                        albu.CLAHE(clip_limit=2),
                                        albu.IAASharpen(),
                                        albu.IAAEmboss(),
                                        albu.RandomBrightnessContrast(),
                                        albu.RandomGamma()
                                    ], p=0.5),
                                    albu.OneOf([
                                        albu.RGBShift(),
                                        albu.HueSaturationValue(),
                                    ], p=0.5),
                                    ]),
            'weak': albu.Compose([albu.HorizontalFlip(),
                                  ]),
            # 几何变换
            'geometric': albu.OneOf([albu.HorizontalFlip(always_apply=True),
                                     albu.ShiftScaleRotate(always_apply=True),
                                     albu.Transpose(always_apply=True),
                                     albu.OpticalDistortion(always_apply=True),
                                     albu.ElasticTransform(always_apply=True),
                                     ])
            }

    aug_fn = augs[scope]   # scope = geometric 选取使用的是几何变换
    crop_fn = {'random': albu.RandomCrop(size, size, always_apply=True),
               'center': albu.CenterCrop(size, size, always_apply=True)}[crop]   # crop = random
    pad = albu.PadIfNeeded(size, size)

    # 将以上的aug,crop,pad组合在一起
    pipeline = albu.Compose([aug_fn, crop_fn, pad], additional_targets={'target': 'image'})

    def process(a, b):
        r = pipeline(image=a, target=b)
        return r['image'], r['target']

    return process

# 函数里面套函数这个写法有点高端
def get_normalize():
    normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = albu.Compose([normalize], additional_targets={'target': 'image'})

    def process(a, b):
        r = normalize(image=a, target=b)
        return r['image'], r['target']

    return process


def _resolve_aug_fn(name):
    d = {
        'cutout': albu.Cutout,
        'rgb_shift': albu.RGBShift,
        'hsv_shift': albu.HueSaturationValue,
        'motion_blur': albu.MotionBlur,
        'median_blur': albu.MedianBlur,
        'snow': albu.RandomSnow,
        'shadow': albu.RandomShadow,
        'fog': albu.RandomFog,
        'brightness_contrast': albu.RandomBrightnessContrast,
        'gamma': albu.RandomGamma,
        'sun_flare': albu.RandomSunFlare,
        'sharpen': albu.IAASharpen,
        'jpeg': albu.JpegCompression,
        'gray': albu.ToGray,
        # ToDo: pixelize
        # ToDo: partial gray
    }
    return d[name]

# 裁剪函数
def get_corrupt_function(config: List[dict]):
    augs = []
    for aug_params in config:
        name = aug_params.pop('name')
        cls = _resolve_aug_fn(name)
        prob = aug_params.pop('prob') if 'prob' in aug_params else .5
        augs.append(cls(p=prob, **aug_params))

    augs = albu.OneOf(augs)

    def process(x):
        return augs(image=x)['image']

    return process
