import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .segment_anything.modeling import Sam
from .segment_anything import sam_model_registry

from .adapter import Adapter
from .adapter import _adapter_attn, _adapter_mlp


class Adapter_Sam(nn.Module):
    """Applies adapter to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        mlp_adapter: the adapter layer for mlp block.
        attn_adapter: the adapter layer for attention block.
        use_mask_decoder_adapter: whether to use mask decoder adapter.
            if false then the mask decoder will be full parameter fine-tuned.


    Examples:
        >>> model = sam_model_registry["vit_b"](checkpoint="xxx")
        >>> adapter_model = Adapter_Sam(model)
        >>> preds = adapter_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self,
            sam_model: Sam,
            encoder_mlp_adapter = Adapter,
            encoder_attn_adapter = Adapter,
            decoder_mlp_adapter = Adapter,
            decoder_attn_adapter = Adapter,
            use_mask_decoder_adapter : bool = True,
        ):

        super(Adapter_Sam, self).__init__()
        self.sam = sam_model
        self.encoder_mlp_adapter = encoder_mlp_adapter
        self.encoder_attn_adapter = encoder_attn_adapter
        self.decoder_mlp_adapter = decoder_mlp_adapter
        self.decoder_attn_adapter = decoder_attn_adapter
        self.use_mask_decoder_adapter = use_mask_decoder_adapter

        # create storage for fine-tuning parameters
        self.image_encoder_adapters = []
        self.mask_decoder_adapters = []
        self.final_attn_adapter = None

        # freeze the original parameters of the model first
        if self.use_mask_decoder_adapter:
            # freeze all parameters in sam
            for param in self.sam.parameters():
                param.requires_grad = False
        else:
            # only freeze image encoder
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False

        # do the surgery on the image encoder section
        for _index, block in enumerate(self.sam.image_encoder.blocks): # 遍历sam中image encoder
            """
            Block(
                (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
                (attn): Attention(
                    (qkv): Linear(in_features=768, out_features=2304, bias=True)
                    (proj): Linear(in_features=768, out_features=768, bias=True)
                )
                (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
                (mlp): MLPBlock(
                    (lin1): Linear(in_features=768, out_features=3072, bias=True)
                    (lin2): Linear(in_features=3072, out_features=768, bias=True)
                    (act): GELU(approximate='none')
                )
            )
            """

            # 获取维度信息
            attn_dim = block.attn.proj.out_features
            mlp_dim = block.mlp.lin2.out_features

            # 创建适配器实例
            adapter_attn = self.encoder_attn_adapter(attn_dim, attn_dim)
            adapter_mlp = self.encoder_mlp_adapter(attn_dim, mlp_dim, skip_connect=False)

            # 将创建的适配器实例添加到image encoder adapter模块
            self.image_encoder_adapters.append(adapter_attn)
            self.image_encoder_adapters.append(adapter_mlp)

            # 替换
            block.attn.proj = _adapter_attn( # _adapter_attn替换block中attention projection
                block_attn_proj = block.attn.proj,
                adapter_attn = adapter_attn,    # 1st
            )
            block.mlp = _adapter_mlp(
                block_mlp = block.mlp,
                adapter_mlp = adapter_mlp,    # 2nd
            )

        # do the surgery on the mask decoder section
        if self.use_mask_decoder_adapter:
            for _index, block in enumerate(self.sam.mask_decoder.transformer.layers):
                """
                TwoWayAttentionBlock(
                    (self_attn): Attention(
                        (q_proj): Linear(in_features=256, out_features=256, bias=True)
                        (k_proj): Linear(in_features=256, out_features=256, bias=True)
                        (v_proj): Linear(in_features=256, out_features=256, bias=True)
                        (out_proj): Linear(in_features=256, out_features=256, bias=True)
                    )
                    (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                    (cross_attn_token_to_image): Attention(
                        (q_proj): Linear(in_features=256, out_features=128, bias=True)
                        (k_proj): Linear(in_features=256, out_features=128, bias=True)
                        (v_proj): Linear(in_features=256, out_features=128, bias=True)
                        (out_proj): Linear(in_features=128, out_features=256, bias=True)
                    )
                    (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                    (mlp): MLPBlock(
                        (lin1): Linear(in_features=256, out_features=2048, bias=True)
                        (lin2): Linear(in_features=2048, out_features=256, bias=True)
                        (act): ReLU()
                    )
                    (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                    (norm4): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                    (cross_attn_image_to_token): Attention(
                        (q_proj): Linear(in_features=256, out_features=128, bias=True)
                        (k_proj): Linear(in_features=256, out_features=128, bias=True)
                        (v_proj): Linear(in_features=256, out_features=128, bias=True)
                        (out_proj): Linear(in_features=128, out_features=256, bias=True)
                    )
                )
                """
                self_attn_dim = block.self_attn.out_proj.out_features
                cross_attn_token_to_image_dim = block.cross_attn_token_to_image.out_proj.out_features
                cross_attn_image_to_token_dim = block.cross_attn_image_to_token.out_proj.out_features
                mlp_in_dim = block.mlp.lin1.in_features
                mlp_out_dim = block.mlp.lin2.out_features

                # 创建适配器实例
                adapter_self_attn = self.decoder_attn_adapter(self_attn_dim, self_attn_dim)
                adapter_cross_attn_token_to_image = self.decoder_attn_adapter(cross_attn_token_to_image_dim, cross_attn_token_to_image_dim)
                adapter_cross_attn_image_to_token = self.decoder_attn_adapter(cross_attn_image_to_token_dim, cross_attn_image_to_token_dim)
                adapter_mlp = self.decoder_mlp_adapter(mlp_in_dim, mlp_out_dim, skip_connect=False)

                self.mask_decoder_adapters.append(adapter_self_attn)
                self.mask_decoder_adapters.append(adapter_cross_attn_token_to_image)
                self.mask_decoder_adapters.append(adapter_cross_attn_image_to_token)
                self.mask_decoder_adapters.append(adapter_mlp)

                block.self_attn.out_proj = _adapter_attn(
                    block_attn_proj = block.self_attn.out_proj,
                    adapter_attn = adapter_self_attn,     # 3rd
                )
                block.cross_attn_token_to_image.out_proj = _adapter_attn(
                    block_attn_proj = block.cross_attn_token_to_image.out_proj,
                    adapter_attn = adapter_cross_attn_token_to_image,    # 4th
                )
                block.cross_attn_image_to_token.out_proj = _adapter_attn(
                    block_attn_proj = block.cross_attn_image_to_token.out_proj,
                    adapter_attn = adapter_cross_attn_image_to_token,   # 5th
                )
                block.mlp = _adapter_mlp(
                    block_mlp = block.mlp,
                    adapter_mlp = adapter_mlp,    # 6th
                )

            # do the surgery on the final_attn section in mask decoder
            """
            final_attn_token_to_image: Attention(
                (q_proj): Linear(in_features=256, out_features=128, bias=True)
                (k_proj): Linear(in_features=256, out_features=128, bias=True)
                (v_proj): Linear(in_features=256, out_features=128, bias=True)
                (out_proj): Linear(in_features=128, out_features=256, bias=True)
            )
            """
            final_attn = self.sam.mask_decoder.transformer.final_attn_token_to_image
            final_attn_dim = final_attn.out_proj.out_features
            self.final_attn_adapter = self.decoder_attn_adapter(final_attn_dim, final_attn_dim)
            final_attn.out_proj = _adapter_attn(
                block_attn_proj = final_attn.out_proj,
                adapter_attn = self.final_attn_adapter,  # 7th
            )

        # prints the parameter information of the model
        self.print_model_parameters_info(False)

    # 打印模型参数信息
    def print_model_parameters_info(self, print_more_info=False):
        # Calculate the total number of parameters
        total_params = sum(p.numel() for p in self.parameters()) # 获取所有参数，并计算参数总数。

        # Filter out trainable parameters
        trainable_params_list = [p for p in self.parameters() if p.requires_grad] # 筛选出可训练的参数
        trainable_params_count = sum(p.numel() for p in trainable_params_list)

        # Print detailed information about trainable parameters
        if print_more_info:
            trainable_params_detailed = [p for p in self.named_parameters() if p[1].requires_grad]
            print("Number of trainable parameters:", len(trainable_params_detailed))
            for name, param in trainable_params_detailed:
                print(f"{name}: {param.shape}")

        # Print the total number of trainable parameters and the trainable ratio
        print("trainable params: {} || all params: {} || trainable ratio: {:.2%}"
              .format(trainable_params_count, total_params, trainable_params_count / total_params))


    # 保存适配器参数
    def save_adapters_parameters(self, filename: str) -> None:
        """
        Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save adapters parameters.
        """
        assert filename.endswith(".pt") or filename.endswith('.pth') # 确保保存文件名以 .pt 或 .pth 结尾。

        # create storage for adapters
        adapter_tensors = {}
        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        for i in range(len(self.image_encoder_adapters) // 2):
            adapter_attn = self.image_encoder_adapters[2 * i]
            adapter_mlp = self.image_encoder_adapters[2 * i + 1]
            
            adapter_attn.save_parameters(adapter_tensors, f"adapter_enc_attn_{i:03d}")
            adapter_mlp.save_parameters(adapter_tensors, f"adapter_enc_mlp_{i:03d}")

        if self.use_mask_decoder_adapter:
            for i in range(len(self.mask_decoder_adapters) // 4):
                adapter_self_attn = self.mask_decoder_adapters[2 * i]
                adapter_t2img_attn = self.mask_decoder_adapters[2 * i + 1]
                adapter_img2t_attn = self.mask_decoder_adapters[2 * i + 2]
                adapter_mlp = self.mask_decoder_adapters[2 * i + 3]
                
                adapter_self_attn.save_parameters(adapter_tensors, f"adapter_dec_self_attn_{i:03d}")
                adapter_t2img_attn.save_parameters(adapter_tensors, f"adapter_dec_t2img_attn{i:03d}")
                adapter_img2t_attn.save_parameters(adapter_tensors, f"adapter_dec_img2t_attn{i:03d}")
                adapter_mlp.save_parameters(adapter_tensors, f"adapter_dec_mlp_{i:03d}")

            self.final_attn_adapter.save_parameters(adapter_tensors, f"adapter_dec_final_attn")

        else:
            # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
            if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
                state_dict = self.sam.module.state_dict()
            else:
                state_dict = self.sam.state_dict()
            for key, value in state_dict.items():
                if 'prompt_encoder' in key:
                    prompt_encoder_tensors[key] = value
                if 'mask_decoder' in key:
                    mask_decoder_tensors[key] = value

        merged_dict = {**adapter_tensors, **prompt_encoder_tensors, **mask_decoder_tensors}
        torch.save(merged_dict, filename)



    def load_adapters_parameters(self, filename: str) -> None:
        """
        Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load adapters parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename)
        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        for i in range(len(self.image_encoder_adapters) // 2):
            adapter_attn = self.image_encoder_adapters[2 * i]
            adapter_mlp = self.image_encoder_adapters[2 * i + 1]
            
            adapter_attn.load_parameters(state_dict, f"adapter_enc_attn_{i:03d}")
            adapter_mlp.load_parameters(state_dict, f"adapter_enc_mlp_{i:03d}")

        if self.use_mask_decoder_adapter:
            for i in range(len(self.mask_decoder_adapters) // 4):
                adapter_self_attn = self.mask_decoder_adapters[2 * i]
                adapter_t2img_attn = self.mask_decoder_adapters[2 * i + 1]
                adapter_img2t_attn = self.mask_decoder_adapters[2 * i + 2]
                adapter_mlp = self.mask_decoder_adapters[2 * i + 3]
                
                adapter_self_attn.load_parameters(state_dict, f"adapter_dec_self_attn_{i:03d}")
                adapter_t2img_attn.load_parameters(state_dict, f"adapter_dec_t2img_attn{i:03d}")
                adapter_img2t_attn.load_parameters(state_dict, f"adapter_dec_img2t_attn{i:03d}")
                adapter_mlp.load_parameters(state_dict, f"adapter_dec_mlp_{i:03d}")

            self.final_attn_adapter.load_parameters(state_dict, f"adapter_dec_final_attn")

        else:
            # load prompt encoder
            prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
            prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
            prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
            sam_dict.update(prompt_encoder_new_state_dict)

            # load mask decoder
            mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
            mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
            mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
            sam_dict.update(mask_decoder_new_state_dict)

        self.sam.load_state_dict(sam_dict)


    def forward(self, batched_input, multimask_output, image_size):
        return self.sam(batched_input, multimask_output, image_size)


if __name__ == "__main__":
    sam = sam_model_registry["vit_b"](
        checkpoint="/root/SAM-Adapter/weights/sam_pretrain/sam_vit_b_01ec64.pth"
    )
    adapter_sam = Adapter_Sam(
        sam_model = sam,
        image_encoder_adapter = CAA_Adapter,
        use_mask_decoder_adapter = True
    )
    print(adapter_sam.sam.image_encoder(torch.rand(size=(1, 3, 1024, 1024))))