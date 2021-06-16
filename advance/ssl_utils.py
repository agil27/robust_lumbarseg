# util functions for the implementation of contrastive learning
# created by Yuanbiao Wang

from PIL import ImageFilter
import random
from jittor import transform
from PIL import Image, ImageEnhance
import numbers
import numpy as np
import jittor as jt
import jittor.nn as nn
import json
import random
import numpy as np
from PIL import Image
from jittor.dataset import Dataset
from os.path import join


class TwoCropsTransform(object):
    '''
    generates two random variations of the same image
    as the query and the key respectively
    '''
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k


# a series of transformations implemented by PIL
class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
    
normalize = transform.ImageNormalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

class RandomApply():
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
        
    def __call__(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)
    

def adjust_brightness(img, brightness_factor):
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):
    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


def adjust_gamma(img, gamma, gain=1):
    input_mode = img.mode
    img = img.convert('RGB')
    gamma_map = [(255 + 1 - 1e-3) * gain * pow(ele / 255., gamma) for ele in range(256)] * 3
    img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

    img = img.convert(input_mode)
    return img


class ColorJitter(nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def get_params(self, brightness, contrast, saturation, hue):
        fn_idx = jt.misc.randperm(4)
        b = None if brightness is None else float(jt.init.uniform([1], jt.float32, brightness[0], brightness[1]))
        c = None if contrast is None else float(jt.init.uniform([1], jt.float32, contrast[0], contrast[1]))
        s = None if saturation is None else float(jt.init.uniform([1], jt.float32, saturation[0], saturation[1]))
        h = None if hue is None else float(jt.init.uniform([1], jt.float32, hue[0], hue[1]))
        return fn_idx, b, c, s, h


    def execute(self, img):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = adjust_hue(img, hue_factor)
        return img
    

augmentation = transform.Compose([
    transform.RandomCropAndResize((512, 512), scale=(0.2, 1.0)), 
    RandomApply(ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8), 
    RandomApply(GaussianBlur([0.1, 2.0]), p=0.5), 
    transform.RandomHorizontalFlip(), 
    transform.ToTensor(),
    normalize
])


aug_for_unet = transform.Compose([
    RandomApply(ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8), 
    RandomApply(GaussianBlur([0.1, 2.0]), p=0.5), 
    transform.ToTensor(),
])


def isImageFile(filename):
    IMAGE_EXTENSIONS = ['.jpg','.png','.bmp','.tif','.tiff','.jpeg']
    return any([filename.lower().endswith(extension) for extension in IMAGE_EXTENSIONS])


def isNumpyFile(filename):
    return filename.lower().endswith('.npy')


# specified dataset for contrastive learning
class AugDataset(Dataset):
    def __init__(self, json_dir, img_dir, mask_dir, batch_size=32, shuffle=False, aug=augmentation):
        super(AugDataset, self).__init__()
        self.batch_size = batch_size
        self.aug = aug
        self.shuffle = shuffle
        self.img_file_names = []
        self.mask_file_names = []

        img_dir_dict = json.load(open(json_dir, "r"))
        
        for person_num in img_dir_dict:
            person = img_dir_dict[person_num]
            for ct in person:
                self.img_file_names.extend([join(img_dir, file_name.replace('.npy', '.jpg')) for file_name in person[ct]["list"] if isNumpyFile(file_name)])
                self.mask_file_names.extend([join(mask_dir, file_name) for file_name in person[ct]["list"] if isNumpyFile(file_name)])
        assert(len(self.mask_file_names) == len(self.img_file_names))
        self.total_len = len(self.img_file_names)
        self.set_attrs(batch_size = self.batch_size, total_len=self.total_len, shuffle= self.shuffle)

    def __getitem__(self, index):
        img, mask = self.fetch(img_path = self.img_file_names[index], mask_path = self.mask_file_names[index])
        query, key = self.aug(img)
        return query, key, mask

    def fetch(self, img_path, mask_path):
        with open(img_path, 'rb') as fp:
            img = Image.open(fp).convert('RGB')
        with open(mask_path, 'rb') as fp:
            mask = np.load(mask_path)
        return img, mask