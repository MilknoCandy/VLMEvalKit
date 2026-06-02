import logging
import os
import torch
from PIL import Image
from vlmeval.vlm.base import BaseModel

logging.getLogger('transformers').setLevel(logging.ERROR)


class LLaVA_OneVision2(BaseModel):
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(self, model_path="lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct", **kwargs):
        import sys

        # Enable TF32 for faster matmul on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        from transformers import AutoProcessor
        # Add local reference implementation to path so the model code
        # (LlavaOnevision2ForConditionalGeneration, Llava_Onevision2Processor)
        # can be loaded even when the checkpoint does not ship auto_map.
        ov2_impl = os.environ.get(
            'LLAVA_OV2_IMPL',
            '/path/to/LLaVA-OneVision-2/transformers_impl',
        )
        if ov2_impl not in sys.path:
            sys.path.insert(0, ov2_impl)

        from llavaonevision2 import LlavaOnevision2ForConditionalGeneration

        # Auto-detect Flash Attention 2
        attn_impl = None
        try:
            import flash_attn  # noqa: F401
            attn_impl = 'flash_attention_2'
        except ImportError:
            pass

        self.model = LlavaOnevision2ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        kwargs.setdefault('max_new_tokens', 1024)
        self.kwargs = kwargs

    def _parse_message(self, message):
        images = []
        video_path = None
        text_parts = []
        has_video = False

        for msg in message:
            if msg['type'] == 'text':
                text_parts.append(msg['value'])
            elif msg['type'] == 'image':
                images.append(msg['value'])
            elif msg['type'] == 'video':
                video_path = msg['value']
                has_video = True

        text = '\n'.join(text_parts)
        return text, images, video_path, has_video

    def _build_conversation(self, text, has_images, has_video):
        content = []
        if has_video:
            content.append({'type': 'video'})
        if has_images:
            for _ in range(has_images):
                content.append({'type': 'image'})
        content.append({'type': 'text', 'text': text})

        return [{'role': 'user', 'content': content}]

    def generate_inner(self, message, dataset=None):
        text, image_paths, video_path, has_video = self._parse_message(message)
        has_images = len(image_paths) > 0

        conversation = self._build_conversation(text, has_images, has_video)
        prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        processor_kwargs = dict(
            text=[prompt],
            padding=True,
            return_tensors='pt',
        )

        if image_paths:
            processor_kwargs['images'] = [Image.open(p).convert('RGB') for p in image_paths]
        if video_path:
            processor_kwargs['videos'] = video_path

        inputs = self.processor(**processor_kwargs).to(self.device)

        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.model.generate(
                **inputs,
                use_cache=True,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
                **self.kwargs,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if isinstance(output_text, list) and len(output_text) == 1:
            return output_text[0]
        return output_text
