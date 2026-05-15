import os
import random
import torch
import numpy as np
import ttach as tta
from tqdm import tqdm
from datetime import datetime
from importlib import import_module
import torch.backends.cudnn as cudnn

from torcheval.metrics.functional import multiclass_confusion_matrix
from utils.py2cfg import py2cfg
from utils.metric import Evaluator
from utils.loss import dice_ce_loss
from utils.dataloader import SegDataset
from utils.dataloader import get_image_mask_mapping
from utils.transforms import data_augmentation as data_aug
from models.segment_anything import sam_model_registry
from models.segment_anything import SamAutomaticMaskGenerator

from torch.utils.data import DataLoader


def compute_multiclass_confusion_matrix(predictions, labels, num_classes):
    # Flatten tensors to 1D arrays
    predictions = predictions.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    
    # Compute the confusion matrix
    cm = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    return cm

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def write_to_log(log_file, data):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write(f"[{current_time}] Log file created.\n")
    
    print(f"[{current_time}] {data}\n")
    with open(log_file, 'a') as f:
        if data.startswith("Start"):
            f.write(f"\n")
        f.write(f"[{current_time}] {data}\n")


def train(config, model, train_dataloaders, valid_dataloaders, multimask_output): 
    base_lr = config.learning_rate
    num_classes = config.num_classes
    learning_rate = base_lr / config.warmup_period if config.warmup else base_lr

    # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        betas=config.adam_betas,
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop_epoch)
    lr_scheduler.last_epoch = config.start_epoch

    max_iterations = config.max_epoch_num * len(train_dataloaders)
    best_model_epoch, best_miou, best_f1, best_oa = 0, 0, 0, 0
    iter_num = 0

    for epoch in range(config.start_epoch, config.max_epoch_num):
        model.train()
        print("\n" + "+" * 36 + " Epoch: " + str(epoch).zfill(3) + " " + "+" * 36)
        # model.print_model_parameters_info()
        data = f"Start Training Epoch: {str(epoch).zfill(3)}, "
        data += f"Learning Rate: {optimizer.param_groups[0]['lr']}"
        write_to_log(f"{config.output_path}/train_logs.txt", data)
        iterator  = tqdm(train_dataloaders)

        for batch in iterator:
            images, labels, boxes = batch['image'], batch['label'], batch['box'] # box???
            # images, labels = batch['image'], batch['label'] # box???
            # if torch.cuda.is_available() and config.device != "cpu": #这里改了
            #     images, labels = images.cuda(), labels.cuda()
            images, labels, boxes = images.cuda(), labels.cuda(), boxes.cuda()
            outputs = model(images, multimask_output, config.image_size[0], boxes) # box
            loss, loss_ce, loss_dice = dice_ce_loss(outputs, labels, config.num_classes, config.dice_param)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if config.warmup and iter_num < config.warmup_period:
                lr = base_lr * ((iter_num + 1) / config.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                if config.warmup:
                    shift_iter = iter_num - config.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                # learning rate adjustment depends on the max iterations
                lr = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            iterator.set_postfix(loss=loss.item(), loss_ce=loss_ce.item(), loss_dice=loss_dice.item())

        if  epoch >= config.stop_epoch:
            # create the path to save the model & inference the model
            model_name = config.checkpoint_name.replace("epoch", str(epoch).zfill(3))
            if not os.path.exists(config.output_path):
                os.makedirs(config.output_path)
            model_path = f"{config.output_path}/{model_name}"
            # f1, miou, oa = inference(config, model, valid_dataloaders, multimask_output, box) # box
            f1, miou, oa = inference(config, model, valid_dataloaders, multimask_output) # box

            # save the model every config.model_save_fre epochs
            if epoch % config.model_save_fre == 0:
                # model.save_adapters_parameters(model_path)
                model.save_lora_parameters(model_path)
                write_to_log(f"{config.output_path}/SUIM_FV_train_logs.txt", f"save model to {model_path}")

            # save the best model according to the mIOU
            if miou > best_miou:
                best_model_epoch, best_miou, best_f1, best_oa = epoch, miou, f1, oa
                model_name = config.checkpoint_name.replace("epoch", "best")
                model_path = f"{config.output_path}/{model_name}"
                # model.save_adapters_parameters(model_path)
                model.save_lora_parameters(model_path)
                data = f"new best model epoch - {best_model_epoch}: F1:{f1}, mIOU:{miou}, OA:{oa}"
                write_to_log(f"{config.output_path}/train_logs.txt", data)


def inference(config, model, valid_dataloaders, multimask_output):
    model.eval()

    tta_transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Rotate90(angles=[90]),
        tta.Scale(scales=[0.5, 0.75, 1, 1.25, 1.5], interpolation='bicubic', align_corners=False)
    ])
    # if config.is_eval:
    #     model = tta.SegmentationTTAWrapper(model, tta_transforms, merge_mode='mean')

    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()
    iterator  = tqdm(valid_dataloaders)
    for batch in iterator:
        idx, images, labels, boxes = batch['idx'], batch['image'], batch['label'], batch['box']
        # if torch.cuda.is_available() and config.device != "cpu":
        #     images = images.cuda()
        images, boxes = images.cuda(), boxes.cuda() # 改了
        predictions = model(images, multimask_output, config.image_size[0], boxes)
        predictions = torch.nn.Softmax(dim=1)(predictions)
        masks = predictions.argmax(dim=1)
        for i in range(config.valid_batch_size):
            # mask = masks[i].cuda().numpy()
            # label = labels[i].cuda().numpy()
          
            
            mask = masks[i].cpu().numpy()
            label = labels[i].cpu().numpy()
            evaluator.add_batch(pre_image=mask, gt_image=label)

#             # 转换为 PyTorch 张量，确保数据类型为 int64
#             predi = torch.tensor(mask, dtype=torch.int64).cuda()  # 转换为 CUDA 张量
#             target = torch.tensor(label, dtype=torch.int64).cuda()  # 转换为 CUDA 张量

#             # 展平为一维张量
#             predi = predi.view(-1)
#             target = target.view(-1)

#             # 计算混淆矩阵
#             multi_confu_mtx = multiclass_confusion_matrix(predi, target, 2)

#             # 写入日志
#             multi_confu_mtx_str = str(multi_confu_mtx.cpu().numpy())

            # 将转换后的字符串传递给 write_to_log
            # write_to_log(f"{config.output_path}/multiclass_confusion_matrix.txt", multi_confu_mtx_str)
            
          
    confusion_matrix = evaluator.confusion_matrix
    print(confusion_matrix)
    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    OA = evaluator.OA()
    for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class, f1_per_class):
        data = f'F1_{class_name}:{class_f1}, IOU_{class_name}:{class_iou}'
        write_to_log(f"{config.output_path}/train_logs.txt", data)
    # data = f'F1:{np.nanmean(f1_per_class[:-1])}, mIOU:{np.nanmean(iou_per_class[:-1])}, OA:{OA}'
    data = f'F1:{np.nanmean(f1_per_class)}, mIOU:{np.nanmean(iou_per_class)}, OA:{OA}'
    write_to_log(f"{config.output_path}/train_logs.txt", data)
    return np.nanmean(f1_per_class), np.nanmean(iou_per_class), OA


if __name__ == "__main__":
    config = py2cfg("configs/adapter_sam.py")
    seed_everything(config.seed)

    cudnn.benchmark = not config.deterministic
    cudnn.deterministic = config.deterministic

    # config.device = torch.device("cuda")
    # config.device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[config.model_type](
        num_classes = config.num_classes,
        checkpoint = config.sam_weights_path,
        # pixel_mean = [0, 0, 0],
        # pixel_std = [1, 1, 1],
        pixel_mean = [123.675, 116.28, 103.53],
        pixel_std =   [58.395, 57.12, 57.375],
        image_size = config.image_size[0],
    )

    # Adapter:
    # pkg = import_module("models.sam_adapter")
    pkg = import_module("models.sam_lora")
    # net = pkg.Adapter_Sam(sam).to(config.device)
    net = pkg.LoRA_Sam(sam,64).to(config.device)
    if config.adapters_weights_path is not None:
        print(f"Loading Adapters Parameters From {config.adapters_weights_path}")
        net.load_lora_parameters(config.adapters_weights_path)
    net.to(config.device)
    # ==============================
    # sam.to(config.device)
    multimask_output = config.num_classes > 2

    mask_generator = SamAutomaticMaskGenerator(model=sam,
                                           points_per_side=32,
                                           points_per_batch=64,
                                           pred_iou_thresh=0.88,
                                           stability_score_thresh=0.95,
                                           stability_score_offset=1.0,
                                           box_nms_thresh=0.7,
                                           crop_n_layers=0,
                                           crop_nms_thresh=0.7,
                                           crop_overlap_ratio=0.34133,
                                           crop_n_points_downscale_factor=1,
                                           point_grids=None,
                                           min_mask_region_area=0,
                                           output_mode='binary_mask')


    # 加载和预处理训练/验证数据集
    print("======> create training dataloader <======")
    if not config.is_eval:
        train_image_mask_list = get_image_mask_mapping(config.train_datasets, flag="train")
        train_dataset = SegDataset(
            dataset_info_list = train_image_mask_list,
            transforms=data_aug(),
            eval_original_resolution = False, # not train
        )
        train_dataloaders = DataLoader(
            dataset = train_dataset,
            batch_size = config.train_batch_size,
            drop_last = True,
            num_workers = 0,
        )
    valid_image_mask_list = get_image_mask_mapping(config.valid_datasets, flag="valid")
    valid_dataset = SegDataset(
        dataset_info_list = valid_image_mask_list,
        transforms=data_aug(),
        eval_original_resolution = True,
    )
    valid_dataloaders = DataLoader(
        dataset = valid_dataset,
        batch_size = config.valid_batch_size,
        drop_last = True,
        num_workers = 0,
    )

    # 开始模型的训练或者评估
    if not config.is_eval:
        print("================> Start training <================")
        train(config, net, train_dataloaders, valid_dataloaders, multimask_output)
        # train(config, sam, train_dataloaders, valid_dataloaders, multimask_output)

        print("Training is done!")
    else:
        print("================> Start inference <================")
        inference(config, net, valid_dataloaders, multimask_output)
        # inference(config, sam, valid_dataloaders, multimask_output)

        print("Inference is done!")