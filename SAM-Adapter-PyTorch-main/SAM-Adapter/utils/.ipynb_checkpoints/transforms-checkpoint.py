import torch
import random
import numpy as np
import ttach as tta
import albumentations as A

from scipy import ndimage
import torchvision.transforms as T
from scipy.ndimage.interpolation import zoom


class random_generator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() > 1:
            image, label = self.random_rot_flip(image, label)
            image = torch.tensor(image, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)
        elif random.random() > 0.5:
            image, label = self.random_rotate(image, label)
            image = torch.tensor(image, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)
        
        h, w = image.shape[1:]
        label_h, label_w = label.shape
        # if h != self.output_size[0] or w != self.output_size[1]:
        #     image = zoom(image, (self.output_size[0] / h, self.output_size[1] / w), order=3)
        #     label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        
        low_res_label = torch.tensor(low_res_label, dtype=torch.float32)
        data.update({'image': image, 'label': label, 'low_res_label': low_res_label.long()})
        return data
    
    def random_rot_flip(self, image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label
    
    def random_rotate(self, image, label):
        angle = np.random.randint(-20, 20)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
        return image, label


class data_augmentation(object):
    def __init__(self):
        self.transform = A.Resize(height=1024, width=1024)

    def __call__(self, image, mask):
        if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
            image, mask = np.array(image), np.array(mask)

        transformed = self.transform(image=image.copy(), mask=mask.copy())

        image, mask = transformed['image'], transformed['mask']
        return image, mask