# 训练参数配置
seed = 42                          # 为确保实验可复现性而设置的随机种子
# device = "cpu"                     # 指定训练使用的设备，此处为 GPU
device = "cuda"
start_epoch = 0                    # 指定训练开始时的轮次
stop_epoch = 0                     # 指定训练停止的轮次，用于提前终止训练过程
max_epoch_num = 100                 # 指定训练过程的最大轮次
warmup = False                     # 指定是否使用学习率热身策略，当此选项激活时，学习率会从较低值渐进到基础学习率
warmup_period = 250                # 指定热身期的迭代次数，只在热身策略激活时有效
dice_param = 0.8                   # 指定 Dice 系数参数，用于模型性能评估
train_batch_size = 4               # 训练阶段的批处理大小，定义了每批训练数据的数量
valid_batch_size = 2               # 验证阶段的批处理大小，定义了每批验证数据的数量

learning_rate = 1e-3               # 定义初始学习率，控制权重更新的幅度
lr_drop_epoch = 10                 # 定义学习率衰减发生的轮次阈值
adam_betas = (0.9, 0.999)          # 为 Adam 优化器定义的 beta 参数，负责动量和二阶动量的衰减率
adam_eps = 1e-08                   # 设定 Adam 优化器的 epsilon 值，增强数值稳定性，避免除零操作
weight_decay = 2.5e-4              # 设置 L2 正则化系数，用于减轻模型过拟合
sgd_momentum = 0.9                 # 用于随机梯度下降优化器的动量参数

image_size = (1024, 1024)          # 定义输入数据的尺寸，此处为图像尺寸
model_save_fre = 5                 # 定义模型保存频率，即每经过多少轮训练后保存一次模型
deterministic = True               # 设置为 1 以启用确定性训练，有助于实验结果的复现
is_eval = False                    # 指定是否仅进行模型评估而不进行训练

vit_dims_dict={
    'vit_b': 768,
    'vit_l': 1024,
    'vit_h': 1280,
}
sam_weights_paths_dict = {
    "vit_b": "/root/autodl-tmp/SAM-Adapter-PyTorch-main/pretrained/sam_vit_b_01ec64.pth",
    "vit_l": "/home/workspace/hongyang/SAM-Adapter-PyTorch-main/pretrained/sam_vit_l_0b3195.pth",
    "vit_h": "/home/workspace/hongyang/SAM-Adapter-PyTorch-main/pretrained/sam_vit_h_4b8939.pth",
}
adapters_weights_paths_dict = {
    # "vit_b": "/root/autodl-tmp/SAM-Adapter-PyTorch-main/SAM-Adapter/outputs/weights/sam-vit-b-sam-LoRA/sam_vit_b_sam_LoRA_best_score.pth",
    # "vit_b":'/root/autodl-tmp/SAM-Adapter-PyTorch-main/SAM-Adapter/outputs/weights/+95-sam-vit-b-sam-LoRA/sam_vit_b_sam_LoRA_best_score.pth',
    "vit_b":'/root/autodl-tmp/SAM-Adapter-PyTorch-main/SAM-Adapter/outputs/weights/+95-sam-vit-b-sam-LoRA/sam_vit_b_sam_LoRA_015_score.pth',
    "vit_l": None,
    "vit_h": None,
}

# name = "sam_adapter"
name = "sam_LoRA"
model_type = "vit_b"
root_path = "./outputs"
vit_dim = vit_dims_dict[model_type]
sam_weights_path = sam_weights_paths_dict[model_type]
adapters_weights_path = adapters_weights_paths_dict[model_type]
checkpoint_name = f"sam_{model_type}_{name}_epoch_score.pth"
output_path = f"{root_path}/weights/++++-sam-{model_type.replace('_', '-')}-{name.replace('_', '-')}"
logs_path = f"{root_path}/train_logs/++++-sam-{model_type.replace('_', '-')}-{name.replace('_', '-')}"
tensorboard_path = f"{root_path}/tensorboard/sam-{model_type.replace('_', '-')}-{name.replace('_', '-')}"

### --------------- Configuring the Train and Valid datasets ---------------
dataset_cod = {
    # "name": "COD10K",
    # "img_dir": "/home/workspace/hongyang/SAM-Adapter-PyTorch-main/load/COD10K/Train/Image",
    # "gt_dir": "/home/workspace/hongyang/SAM-Adapter-PyTorch-main/load/COD10K/Train/GT_Object_Aquatic",

    # "name": "SUIM",
    # "img_dir": "/home/workspace/hongyang/SAM-Adapter-PyTorch-main/load/SUIM/train_val/images",
    # "gt_dir": "/home/workspace/hongyang/SAM-Adapter-PyTorch-main/load/SUIM/train_val/train_gt",
    # "img_ext": ".jpg",
    # "gt_ext": ".bmp",
    
    "name": "SUIM",
    "img_dir": "/root/autodl-tmp/SAM-Adapter-PyTorch-main/load/SUIM/train_val_2class/image_val",
    "gt_dir": "/root/autodl-tmp/SAM-Adapter-PyTorch-main/load/SUIM/train_val_2class/mask/using",
    "img_ext": ".bmp",
    "gt_ext": ".bmp",

#     "name":"UIIS",
#     "img_dir": "/home/workspace/hongyang/SAM-Adapter-PyTorch-main/load/UIIS/UIIS/UDW/train",
#     "gt_dir": "/home/workspace/hongyang/SAM-Adapter-PyTorch-main/load/UIIS/UIIS/UDW/train_gt",
#     "img_ext": ".jpg",
#     "gt_ext": ".png",
}

dataset_cod_val = {
    # "name": "COD10K",
    # "img_dir": "/home/workspace/hongyang/SAM-Adapter-PyTorch-main/load/COD10K/Test/Image_Aquatic",
    # "gt_dir": "/home/workspace/hongyang/SAM-Adapter-PyTorch-main/load/COD10K/Test/GT_Object_Aquatic",

    "name": "SUIM",
    "img_dir": "/root/autodl-tmp/SAM-Adapter-PyTorch-main/load/TEST/FV_IMG",
    "gt_dir": "/root/autodl-tmp/SAM-Adapter-PyTorch-main/load/TEST/FV",
    "img_ext": ".jpg",
    "gt_ext": ".bmp",

    # "name":"UIIS",
    # "img_dir": "/home/workspace/hongyang/SAM-Adapter-PyTorch-main/load/UIIS/UIIS/UDW/val",
    # "gt_dir": "/home/workspace/hongyang/SAM-Adapter-PyTorch-main/load/UIIS/UIIS/UDW/val_gt",
    # "img_ext": ".jpg",
    # "gt_ext": ".png",
}

num_classes = 2
classes = ('0', '1')
train_datasets = [dataset_cod]
valid_datasets = [dataset_cod_val]
