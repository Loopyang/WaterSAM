import os
import torch
import numpy as np

from glob import glob
from skimage import io
from copy import deepcopy
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset, DistributedSampler


def get_image_mask_mapping(datasets, flag='valid'):
    dataset_info_list = []

    for i, dataset in enumerate(datasets):
        print(f"--->>> {flag} dataset {i + 1}/{len(datasets)} {dataset['name']} <<<---")

        # Construct and retrieve a list of images
        image_file_pattern = os.path.join(dataset["img_dir"], '*' + dataset["img_ext"])
        image_list = glob(image_file_pattern)
        print(f'-images- {dataset["name"]} {dataset["img_dir"]}: {len(image_list)}')

        # Construct a label list based on the list of images
        if dataset["gt_dir"]:
            ground_truth_list = [
                os.path.join(dataset["gt_dir"], os.path.basename(x).replace(dataset["img_ext"], dataset["gt_ext"])) for
                x in image_list]
            print(f'-labels- {dataset["name"]} {dataset["gt_dir"]}: {len(ground_truth_list)}')
        else:
            ground_truth_list = []
            print(f'-labels- {dataset["name"]} {dataset["gt_dir"]}: No Ground Truth Found')

        dataset_info_list.append({
            "dataset_name": dataset["name"],
            "img_path": image_list,
            "gt_path": ground_truth_list,
            "img_ext": dataset["img_ext"],
            "gt_ext": dataset["gt_ext"]
        })

    return dataset_info_list


class SegDataset(Dataset):
    def __init__(self, dataset_info_list, transforms=None, eval_original_resolution=False):
        self.transforms = transforms
        self.dataset = self._prepare_dataset(dataset_info_list)
        self.eval_original_resolution = eval_original_resolution

    def _prepare_dataset(self, dataset_info_list):
        dataset_items = {
            "data_name": [],  # dataset name per image
            "img_name": [],  # image name
            "img_path": [],  # image path
            "ori_img_path": [],  # original image path for backup
            "gt_path": [],  # ground truth path
            "ori_gt_path": [],  # original ground truth path for backup
            "img_ext": [],  # image extension
            "gt_ext": []  # ground truth extension
        }

        for info in dataset_info_list:
            dataset_items["data_name"].extend([info["dataset_name"] for x in info["img_path"]])
            dataset_items["img_name"].extend([x.split(os.sep)[-1].split(info["img_ext"])[0] for x in info["img_path"]])
            dataset_items["img_path"].extend(info["img_path"])
            dataset_items["gt_path"].extend(info["gt_path"])
            dataset_items["img_ext"].extend([info["img_ext"] for x in info["img_path"]])
            dataset_items["gt_ext"].extend([info["gt_ext"] for x in info["gt_path"]])

        dataset_items["ori_img_path"] = deepcopy(dataset_items["img_path"])
        dataset_items["ori_gt_path"] = deepcopy(dataset_items["gt_path"])

        return dataset_items

    def __len__(self):
        return len(self.dataset["img_path"])

    def __getitem__(self, idx):
        img_path = self.dataset["img_path"][idx]
        gt_path = self.dataset["gt_path"][idx]
        img = io.imread(img_path)
        gt = io.imread(gt_path)
        gt2D = gt
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        
        H, W = gt2D.shape
        x_min = max(0, x_min - np.random.randint(0, 20))

        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = np.array([x_min, y_min, x_max, y_max])

        if len(gt.shape) > 2:
            gt = gt[:, :, 0]
        if len(img.shape) < 3:
            img = img[:, :, np.newaxis]
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        if self.transforms:
            img, gt = self.transforms(img, gt)

        img = torch.tensor(img.copy(), dtype=torch.float32)
        img = torch.transpose(torch.transpose(img, 1, 2), 0, 1)
        gt = torch.tensor(gt / 255, dtype=torch.float32)
        # gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32), 0)
        # import pdb
        # pdb.set_trace()

        item = {
            "idx": torch.from_numpy(np.array(idx)),
            "image": img,
            "label": gt,
            "shape": torch.tensor(img.shape[-2:]),
            "box": bbox,
        }

        if self.eval_original_resolution:
            item["ori_label"] = gt.type(torch.uint8)
            item['ori_im_path'] = self.dataset["img_path"][idx]
            item['ori_gt_path'] = self.dataset["gt_path"][idx]
            

        return item