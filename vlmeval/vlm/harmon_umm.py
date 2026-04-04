import os

import numpy as np
import torch
from einops import rearrange
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from vlmeval.smp import get_cache_path, get_logger
from .base import BaseModel

logger = get_logger(__name__)

PROMPT_TEMPLATE = dict(
    INSTRUCTION='<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n',
)

MODEL_CONFIGS = {
    'Harmon-0.5B': dict(
        llm_name='Qwen/Qwen2.5-0.5B-Instruct',
        mar_type='mar_base',
        mar_kwargs=dict(diffloss_d=6, diffloss_w=1024),
    ),
    'Harmon-1.5B': dict(
        llm_name='Qwen/Qwen2.5-1.5B-Instruct',
        mar_type='mar_huge',
        mar_kwargs=dict(diffloss_d=12, diffloss_w=1536),
    ),
}

MAR_COMMON = dict(
    img_size=256, vae_stride=16, patch_size=1, vae_embed_dim=16,
    mask_ratio_min=0.7, label_drop_prob=0.1, class_num=1000,
    attn_dropout=0.1, proj_dropout=0.1, buffer_size=64,
    num_sampling_steps='100', diffusion_batch_mul=4, grad_checkpointing=False,
)


def expand2square(pil_img, background_color=(127, 127, 127)):
    w, h = pil_img.size
    if w == h:
        return pil_img
    elif w > h:
        result = Image.new(pil_img.mode, (w, w), background_color)
        result.paste(pil_img, (0, (w - h) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (h, h), background_color)
        result.paste(pil_img, ((h - w) // 2, 0))
        return result


class Harmon(BaseModel):
    """
    Harmon: Unified Multimodal Understanding and Generation
    https://arxiv.org/abs/2503.21979
    """

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(
        self,
        model_path='wusize/Harmon-0_5B',
        variant='Harmon-0.5B',
        image_size=512,
        max_new_tokens=1024,
        **kwargs,
    ):
        assert variant in MODEL_CONFIGS, f"variant must be one of {list(MODEL_CONFIGS.keys())}"
        cfg = MODEL_CONFIGS[variant]

        # resolve checkpoint path
        if os.path.exists(model_path):
            cache_path = model_path
        else:
            if get_cache_path(model_path, repo_type='models') is None:
                snapshot_download(repo_id=model_path)
            cache_path = get_cache_path(model_path, repo_type='models')

        self.model_path = cache_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        # register <image> special token
        num_added = self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<image>']}
        )
        assert num_added == 1

        device = torch.cuda.current_device()
        self.device = device
        self.model = AutoModel.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True).eval()
        self.model = self.model.to(device)

        self.image_token_idx = self.tokenizer.encode('<image>', add_special_tokens=False)[-1]

        self.image_size = image_size
        self.image_length = (image_size // 16) ** 2 + self.model.mar.buffer_size  # vae_stride=16, buffer_size=64
        self.max_new_tokens = max_new_tokens

    def _preprocess_image(self, img_path):
        with Image.open(img_path) as img:
            image = img.convert('RGB')
        image = expand2square(image)
        image = image.resize((self.image_size, self.image_size))
        image = torch.from_numpy(np.array(image)).to(dtype=self.model.dtype, device=self.model.device)
        image = rearrange(image, 'h w c -> c h w')[None]
        image = 2 * (image / 255) - 1
        return image

    def generate_inner(self, message, dataset=None):
        # Build interleaved prompt preserving image-text order
        prompt_parts = []
        images = []
        for x in message:
            if x['type'] == 'image':
                images.append(x['value'])
                prompt_parts.append('<image>\n')
            elif x['type'] == 'text':
                prompt_parts.append(x['value'])

        prompt_text = ' '.join(prompt_parts)
        prompt = PROMPT_TEMPLATE['INSTRUCTION'].format(input=prompt_text)
        prompt = prompt.replace('<image>', '<image>' * self.image_length)

        input_ids = self.tokenizer.encode(
            prompt, add_special_tokens=True, return_tensors='pt'
        ).cuda()

        with torch.inference_mode():
            if images:
                # Encode all images and concatenate features
                z_encs = []
                for img_path in images:
                    img_tensor = self._preprocess_image(img_path)
                    _, z_enc = self.model.extract_visual_feature(self.model.encode(img_tensor))
                    z_encs.append(z_enc.flatten(0, 1))  # (image_length, hidden_size)

                inputs_embeds = z_encs[0].new_zeros(*input_ids.shape, self.model.llm.config.hidden_size)
                # Fill text token embeddings
                inputs_embeds[input_ids != self.image_token_idx] = \
                    self.model.llm.get_input_embeddings()(input_ids[input_ids != self.image_token_idx])
                # Fill each image's features into its corresponding token positions in order
                img_token_positions = (input_ids[0] == self.image_token_idx).nonzero(as_tuple=True)[0]
                for i, z in enumerate(z_encs):
                    start = i * self.image_length
                    positions = img_token_positions[start:start + self.image_length]
                    inputs_embeds[0, positions] = z
            else:
                inputs_embeds = self.model.llm.get_input_embeddings()(input_ids)

            output = self.model.llm.generate(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=(self.tokenizer.pad_token_id
                              if self.tokenizer.pad_token_id is not None
                              else self.tokenizer.eos_token_id),
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
