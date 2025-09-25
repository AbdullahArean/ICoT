# import argparse
# import copy
# import json
# import os
# from dataclasses import dataclass
# from typing import Optional, Tuple, Union, Dict
#
# import torch
# from PIL import Image
# from ruamel.yaml import YAML
# from torch.nn import CrossEntropyLoss
# from tqdm import tqdm
# from transformers.cache_utils import Cache
# from transformers.modeling_outputs import CausalLMOutputWithPast
# from transformers.models.chameleon.modeling_chameleon import ChameleonForConditionalGeneration
# from transformers.models.chameleon.processing_chameleon import ChameleonProcessor
#
# # -----------------------
# # CLI / Config
# # -----------------------
# parser = argparse.ArgumentParser()
# parser.add_argument('--config', default='./config/config.yaml', help='global environment configs')
# args = parser.parse_args()
# yaml = YAML()
#
# with open(args.config, 'r') as file:
#     config = yaml.load(file)
#     print(config)
#
# # Optional: reduce allocator fragmentation (harmless if unknown)
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
#
# # -----------------------
# # Paths / Data
# # -----------------------
# IMG_FOLDER = './data/m3cot/images/'
# EVAL_FILE = './data/m3cot/test.jsonl'
# DATA_NAME = 'm3cot'
#
# dataset = [json.loads(d) for d in open(EVAL_FILE).readlines()]
# dataset = [x for x in dataset if x['image'] is not None]
#
# # -----------------------
# # Custom output type
# # -----------------------
# @dataclass
# class CausalLMOutputWithPastForInterCoT(CausalLMOutputWithPast):
#     selected_vokens: torch.LongTensor = None
#
# # -----------------------
# # InterCoT wrapper class
# # -----------------------
# class ChameleonForInterCoT(ChameleonForConditionalGeneration):
#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         sub_image_masks=None,
#         pixel_values: torch.FloatTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Cache] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         **kwargs,
#     ) -> Union[Tuple, CausalLMOutputWithPast]:
#
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         outputs = self.model(
#             input_ids=input_ids,
#             sub_image_masks=sub_image_masks,
#             pixel_values=pixel_values,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             cache_position=cache_position,
#         )
#
#         hidden_states = outputs[0]
#         logits = self.lm_head(hidden_states).float()
#
#         # Block raw image tokens outside image spans
#         image_tokens = self.model.vocabulary_mapping.image_tokens
#         logits[:, :, image_tokens] = torch.finfo(logits.dtype).min
#
#         loss = None
#         if labels is not None:
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1).to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)
#
#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (loss,) + output if loss is not None else output
#
#         return CausalLMOutputWithPastForInterCoT(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
#
# # -----------------------
# # Load processor & model (single load → no OOM)
# # -----------------------
# model_path = 'facebook/chameleon-7b'  # replace if different on HF Hub
# processor = ChameleonProcessor.from_pretrained(model_path)
#
# # Toggle if you have bitsandbytes and want more headroom
# USE_8BIT = config.get('use_8bit', False)
# USE_4BIT = config.get('use_4bit', False)
#
# if USE_8BIT or USE_4BIT:
#     # Requires bitsandbytes and a compatible GPU/driver
#     from transformers import BitsAndBytesConfig
#
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=USE_4BIT,
#         load_in_8bit=USE_8BIT,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4"
#     )
#
#     model = ChameleonForInterCoT.from_pretrained(
#         model_path,
#         attn_implementation=config.get('attn', 'eager'),
#         quantization_config=quantization_config,
#         device_map="auto",
#         low_cpu_mem_usage=config.get('low_cpu_mem_usage', True),
#         torch_dtype=torch.bfloat16,
#     )
# else:
#     # Keep weights in bf16 on a single GPU with memory optimizations
#     model = ChameleonForInterCoT.from_pretrained(
#         model_path,
#         attn_implementation=config.get('attn', 'eager'),
#         low_cpu_mem_usage=config.get('low_cpu_mem_usage', True),
#         torch_dtype=torch.bfloat16,
#     ).to(device='cuda')
#
#     # Enable gradient checkpointing to save memory
#     if config.get('gradient_checkpointing', False):
#         model.gradient_checkpointing_enable()
#
#     # Disable torch compile if specified
#     if not config.get('torch_compile', False):
#         torch._dynamo.config.suppress_errors = True
#
# # -----------------------
# # Generation config (trimmed for memory)
# # -----------------------
# generation_config = {
#     'do_sample': True,
#     'temperature': config.get('temperature', 0.7),
#     'top_p': config.get('top_p', 0.9),
#     'repetition_penalty': config.get('repetition_penalty', 1.2),
#     'min_new_tokens': config.get('min_new_tokens', 4),   # lower → less compute/memory
#     'max_new_tokens': config.get('max_new_tokens', 32)   # lower than before to avoid OOM
# }
#
# # Turn OFF attention outputs to save tons of memory
# MCOT = config.get('MCOT', False)  # MCOT can be enabled via config but uses more memory
#
# # -----------------------
# # Few-shot exemplar (NO REGION TAGS)
# # -----------------------
# TRAING_CASE_1 = {
#     'id': 'physical-commonsense-1426',
#     'question': 'What general conclusion can you draw about this kitchen?',
#     'choices': [
#         'This is the kitchen of a restaurant',
#         'The equipment in front has not been cleaned for a long time',
#         'Someone searched in this kitchen',
#         'All options are correct'
#     ],
#     'answer': 'D',
# }
#
# zero_shot_prompt_template = (
#     "<image>Question: {}\n"
#     "Options:\n"
# )
#
# mcot_induct = (
#     "<image>Question: {}\n"
#     "Options:\n"
#     "A. {}\n"
#     "B. {}\n"
#     "C. {}\n"
#     "D. {}\n"
#     "First, the image shows large ovens in a kitchen area that indicates it is a kitchen of a restaurant. "
#     "Therefore, option A is correct. "
#     "Second, there are grease stains on the front of appliances which are indicative of not being cleaned in a while. "
#     "So option B is correct answer. "
#     "Third, cabinet doors are opened up throughout the kitchen which shows someone was searching for something. "
#     "So option C is incorrect. Therefore, we can infer that option A, B and C are all correct.\n"
#     "Answer: {}"
# ).format(TRAING_CASE_1['question'], *TRAING_CASE_1['choices'], TRAING_CASE_1['answer'])
#
# # -----------------------
# # Robust generate helper
# # -----------------------
# def calculate_generated_text(prompt: str, vision_x, return_image_masks: bool = False) -> str:
#     """
#     - prompt must contain N "<image>" tags.
#     - vision_x must be a list of N PIL images.
#     - Keeps integer tensors as long/int; only pixel_values -> bfloat16.
#     - Passes sub_image_masks ONLY if it contains any True; else omit so model uses all patches.
#     """
#     # 1) Processor
#     inputs, sub_image_masks = processor(
#         text=prompt,
#         images=vision_x,
#         padding=True,
#         return_tensors="pt",
#         return_for_text_completion=False,
#         return_image_masks=True
#     )
#
#     # 2) Move to CUDA with strict dtypes
#     fixed: Dict[str, torch.Tensor] = {}
#     for k, v in inputs.items():
#         if k == "pixel_values":
#             # floats only
#             target_dtype = torch.bfloat16 if not USE_8BIT else v.dtype  # 8-bit path manages dtypes
#             fixed[k] = v.to(device="cuda", dtype=target_dtype)
#         elif k in ("input_ids", "position_ids"):
#             fixed[k] = v.to(device="cuda")  # keep long
#         elif k == "attention_mask":
#             fixed[k] = (v > 0).to(device="cuda", dtype=torch.long)  # 0/1 int mask
#         else:
#             fixed[k] = v.to(device="cuda")
#
#     # Let model derive position_ids
#     fixed.pop("position_ids", None)
#
#     # Number of images (N)
#     n_imgs = fixed["pixel_values"].shape[0] if "pixel_values" in fixed else prompt.count("<image>")
#
#     # 3) sub_image_masks → pass only if any True; else omit (use all 1024 by default)
#     if return_image_masks and sub_image_masks is not None:
#         sim = sub_image_masks.to(device="cuda")
#
#         # Normalize to [n_imgs, 1024]
#         if sim.ndim == 1:
#             sim = sim.unsqueeze(0)
#         if sim.shape[0] > n_imgs:
#             sim = sim[:n_imgs]
#         if sim.shape[1] > 1024:
#             sim = sim[:, :1024]
#         elif sim.shape[1] < 1024:
#             pad = torch.zeros(sim.shape[0], 1024 - sim.shape[1], dtype=sim.dtype, device=sim.device)
#             sim = torch.cat([sim, pad], dim=1)
#
#         sim = sim.bool()
#         if int(sim.sum().item()) > 0:
#             fixed["sub_image_masks"] = sim
#         # else: omit, so the model uses all patches per image
#
#     # 4) Invariants
#     iid = fixed["input_ids"]; am = fixed["attention_mask"]; pv = fixed["pixel_values"]
#     assert iid.dtype == torch.long, f"input_ids dtype {iid.dtype} must be long"
#     assert am.dtype == torch.long, f"attention_mask dtype {am.dtype} must be long"
#     if not USE_8BIT:
#         assert pv.dtype == torch.bfloat16, f"pixel_values dtype {pv.dtype} must be bfloat16"
#     assert iid.shape[0] == 1 and am.shape[0] == 1, \
#         f"text batch must be 1 (got input_ids {iid.shape[0]}, attention_mask {am.shape[0]})"
#     assert prompt.count("<image>") == n_imgs, \
#         f"#<image> in prompt ({prompt.count('<image>')}) ≠ pixel_values batch ({n_imgs})"
#
#     # 5) Disable attention outputs to save memory
#     fixed["output_attentions"] = False  # MCOT is off for memory
#
#     # 6) Generate under inference_mode (extra memory savings)
#     gen_cfg = dict(**generation_config, return_dict_in_generate=True)
#     with torch.inference_mode():
#         out = model.generate(**fixed, **gen_cfg)
#     out = out[0][fixed["input_ids"].shape[1]:]
#     return processor.decode(out, skip_special_tokens=True)
#
# # -----------------------
# # Main
# # -----------------------
# def main():
#     os.makedirs(f'./results/chameleon/{DATA_NAME}', exist_ok=True)
#     mcot_one_path = f'./results/chameleon/{DATA_NAME}/chameleon_mcot_one.json'
#     mcot_zero_path = f'./results/chameleon/{DATA_NAME}/chameleon_mcot_zero.json'
#     mcot_one_fh = open(mcot_one_path, 'a')
#     mcot_zero_fh = open(mcot_zero_path, 'a')
#
#     for data in tqdm(dataset):
#         # Build zero-shot prompt
#         mcot_input_str = zero_shot_prompt_template.format(data['question'])
#         for i, c in zip(['A', 'B', 'C', 'D', 'E', 'F'], data['choices']):
#             mcot_input_str += f'{i}. {c}\n'
#
#         # Images (one-shot: 2; zero-shot: 1)
#         def loadAndResizeImage(path):
#             img = Image.open(path)
#             if config.get('image_size', 512) != 512:
#                 # Resize to smaller size to save memory
#                 target_size = config.get('image_size', 512)
#                 img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
#             return img
#
#         one_shot_vision = [
#             loadAndResizeImage(os.path.join('./data/m3cot/images', TRAING_CASE_1['id'] + '.png')),  # exemplar
#             loadAndResizeImage(os.path.join(IMG_FOLDER, data['id'] + '.png' if DATA_NAME == 'm3cot' else data['image'])),  # query
#         ]
#         zero_shot_vision = [
#             loadAndResizeImage(os.path.join(IMG_FOLDER, data['id'] + '.png' if DATA_NAME == 'm3cot' else data['image']))
#         ]
#
#         # Prompts
#         one_shot_mcot_input_str = mcot_induct + '\n' + mcot_input_str      # two <image> tags total
#         zero_shot_mcot_input_str = mcot_input_str                           # one <image> tag total
#
#         # Generate
#         one_shot = calculate_generated_text(one_shot_mcot_input_str, one_shot_vision, return_image_masks=True)
#         zero_shot = calculate_generated_text(zero_shot_mcot_input_str, zero_shot_vision, return_image_masks=False)
#
#         # Write
#         oneshot_mcot_output = copy.deepcopy(data); oneshot_mcot_output['pred'] = one_shot
#         zeroshot_mcot_output = copy.deepcopy(data); zeroshot_mcot_output['pred'] = zero_shot
#
#         mcot_one_fh.write(json.dumps(oneshot_mcot_output) + '\n')
#         mcot_zero_fh.write(json.dumps(zeroshot_mcot_output) + '\n')
#
#         # Optional: free some memory between samples
#         torch.cuda.empty_cache()
#
#     mcot_one_fh.close()
#     mcot_zero_fh.close()
#
# if __name__ == '__main__':
#     main()
# chameleon_intercot_eval.py
# Fully running script for MCOT/zero-shot prediction on M3CoT with Chameleon

import argparse
import copy
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict

import torch
from PIL import Image
from ruamel.yaml import YAML
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.chameleon.modeling_chameleon import ChameleonForConditionalGeneration
from transformers.models.chameleon.processing_chameleon import ChameleonProcessor

# -----------------------
# CLI / Config
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/config.yaml', help='global environment configs')
args = parser.parse_args()
yaml = YAML()

with open(args.config, 'r', encoding='utf-8') as file:
    config = yaml.load(file)
    print("Loaded config:", config)

# Optional: reduce allocator fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.set_grad_enabled(False)

# -----------------------
# Paths / Data
# -----------------------
IMG_FOLDER = './data/m3cot/images/'
EVAL_FILE = './data/m3cot/test.jsonl'
DATA_NAME = 'm3cot'

assert os.path.exists(EVAL_FILE), f"Missing eval file: {EVAL_FILE}"
with open(EVAL_FILE, 'r', encoding='utf-8') as fh:
    dataset = [json.loads(d) for d in fh.readlines()]
dataset = [x for x in dataset if x.get('image') is not None]

# -----------------------
# Custom output type
# -----------------------
@dataclass
class CausalLMOutputWithPastForInterCoT(CausalLMOutputWithPast):
    selected_vokens: torch.LongTensor = None

# -----------------------
# InterCoT wrapper class
# -----------------------
class ChameleonForInterCoT(ChameleonForConditionalGeneration):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        sub_image_masks=None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            sub_image_masks=sub_image_masks,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()

        # Block raw image tokens outside image spans (guarded)
        image_tokens = getattr(getattr(self.model, "vocabulary_mapping", None), "image_tokens", None)
        if image_tokens:
            logits[:, :, image_tokens] = torch.finfo(logits.dtype).min

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastForInterCoT(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# -----------------------
# Load processor & model
# -----------------------
model_path = config.get('model_path', 'facebook/chameleon-7b')
processor = ChameleonProcessor.from_pretrained(model_path)

USE_8BIT = bool(config.get('use_8bit', False))
USE_4BIT = bool(config.get('use_4bit', False))

if USE_8BIT or USE_4BIT:
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=USE_4BIT,
        load_in_8bit=USE_8BIT,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = ChameleonForInterCoT.from_pretrained(
        model_path,
        attn_implementation=config.get('attn', 'eager'),
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=config.get('low_cpu_mem_usage', True),
        torch_dtype=torch.bfloat16,
    )
else:
    model = ChameleonForInterCoT.from_pretrained(
        model_path,
        attn_implementation=config.get('attn', 'eager'),
        low_cpu_mem_usage=config.get('low_cpu_mem_usage', True),
        torch_dtype=torch.bfloat16,
    ).to(device='cuda')

    if config.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()

    if not config.get('torch_compile', False):
        try:
            torch._dynamo.config.suppress_errors = True
        except Exception:
            pass

# Ensure pad_token_id exists (needed when using padding=True)
if getattr(model.generation_config, "pad_token_id", None) is None:
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

# -----------------------
# Generation config
# -----------------------
generation_config = {
    'do_sample': True,
    'temperature': float(config.get('temperature', 0.7)),
    'top_p': float(config.get('top_p', 0.9)),
    'repetition_penalty': float(config.get('repetition_penalty', 1.2)),
    'min_new_tokens': int(config.get('min_new_tokens', 4)),
    'max_new_tokens': int(config.get('max_new_tokens', 32)),
}

# -----------------------
# Few-shot exemplar
# -----------------------
TRAING_CASE_1 = {
    'id': 'physical-commonsense-1426',
    'question': 'What general conclusion can you draw about this kitchen?',
    'choices': [
        'This is the kitchen of a restaurant',
        'The equipment in front has not been cleaned for a long time',
        'Someone searched in this kitchen',
        'All options are correct'
    ],
    'answer': 'D',
}

zero_shot_prompt_template = (
    "<image>Question: {}\n"
    "Options:\n"
)

mcot_induct = (
    "<image>Question: {}\n"
    "Options:\n"
    "A. {}\n"
    "B. {}\n"
    "C. {}\n"
    "D. {}\n"
    "First, the image shows large ovens in a kitchen area that indicates it is a kitchen of a restaurant. "
    "Therefore, option A is correct. "
    "Second, there are grease stains on the front of appliances which are indicative of not being cleaned in a while. "
    "So option B is correct answer. "
    "Third, cabinet doors are opened up throughout the kitchen which shows someone was searching for something. "
    "So option C is incorrect. Therefore, we can infer that option A, B and C are all correct.\n"
    "Answer: {}"
).format(TRAING_CASE_1['question'], *TRAING_CASE_1['choices'], TRAING_CASE_1['answer'])

# -----------------------
# Dtype helper
# -----------------------
def _model_float_dtype(m) -> torch.dtype:
    for p in m.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float32

# -----------------------
# Generate helper (dtype-safe)
# -----------------------
def calculate_generated_text(prompt: str, vision_x, return_image_masks: bool = False) -> str:
    """
    - prompt must contain N "<image>" tags.
    - vision_x must be a list of N PIL images.
    - Ensures floating tensors match the model's dtype (bf16 in this setup),
      regardless of device_map setting.
    """
    # 1) Processor
    inputs, sub_image_masks = processor(
        text=prompt,
        images=vision_x,
        padding=True,
        return_tensors="pt",
        return_for_text_completion=True,
        return_image_masks=True
    )

    # 2) Determine dispatch mode & dtype
    using_auto_map = any(getattr(model, attr, None) for attr in ("hf_device_map", "device_map"))
    target_dtype = _model_float_dtype(model)  # typically torch.bfloat16

    fixed: Dict[str, torch.Tensor] = {}
    for k, v in inputs.items():
        is_float = torch.is_floating_point(v)
        if using_auto_map:
            # Let Accelerate move devices; cast floats to model dtype
            fixed[k] = v.to(dtype=target_dtype) if is_float else v
        else:
            if k == "pixel_values":
                fixed[k] = v.to(device="cuda", dtype=target_dtype)
            elif k in ("input_ids", "position_ids"):
                fixed[k] = v.to(device="cuda")
            elif k == "attention_mask":
                fixed[k] = (v > 0).to(device="cuda", dtype=torch.long)
            else:
                fixed[k] = v.to(device="cuda", dtype=target_dtype if is_float else None)

    # Let model derive position_ids
    fixed.pop("position_ids", None)

    # Invariants
    n_imgs_from_prompt = prompt.count("<image>")
    if "pixel_values" in fixed:
        assert fixed["pixel_values"].shape[0] == n_imgs_from_prompt, \
            f"pixel_values batch {fixed['pixel_values'].shape[0]} != #<image> {n_imgs_from_prompt}"

    # sub_image_masks only if any True
    if return_image_masks and sub_image_masks is not None:
        sim = sub_image_masks
        if not using_auto_map:
            sim = sim.to(device="cuda")
        if sim.ndim == 1:
            sim = sim.unsqueeze(0)
        if sim.shape[1] >= 1024:
            sim = sim[:, :1024]
        else:
            pad = torch.zeros(sim.shape[0], 1024 - sim.shape[1], dtype=sim.dtype, device=sim.device)
            sim = torch.cat([sim, pad], dim=1)
        sim = sim[:n_imgs_from_prompt].bool()
        if int(sim.sum().item()) > 0:
            fixed["sub_image_masks"] = sim

    # attention mask must be long
    if "attention_mask" in fixed:
        fixed["attention_mask"] = fixed["attention_mask"].to(dtype=torch.long)

    # No attention outputs (memory)
    fixed["output_attentions"] = False

    # Generate
    gen_cfg = dict(**generation_config, return_dict_in_generate=True)
    with torch.inference_mode():
        gen_out = model.generate(**fixed, **gen_cfg)

    # Extract sequences and slice off the prompt
    sequences = gen_out.sequences
    input_len = fixed["input_ids"].shape[1]
    gen_tokens = sequences[0, input_len:]
    text = processor.decode(gen_tokens, skip_special_tokens=True)
    return text.strip()

# -----------------------
# Helpers
# -----------------------
def load_and_resize_image(path: str, target_size: int) -> Image.Image:
    assert os.path.exists(path), f"Missing image: {path}"
    img = Image.open(path).convert("RGB")
    if target_size != 512:
        img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    return img

# -----------------------
# Main
# -----------------------
def main():
    os.makedirs(f'./results/chameleon/{DATA_NAME}', exist_ok=True)
    mcot_one_path = f'./results/chameleon/{DATA_NAME}/chameleon_mcot_one.json'
    mcot_zero_path = f'./results/chameleon/{DATA_NAME}/chameleon_mcot_zero.json'

    # overwrite each run for cleanliness
    mcot_one_fh = open(mcot_one_path, 'w', encoding='utf-8')
    mcot_zero_fh = open(mcot_zero_path, 'w', encoding='utf-8')

    image_target_size = int(config.get('image_size', 512))

    for data in tqdm(dataset, desc="Evaluating"):
        # Build zero-shot prompt
        mcot_input_str = zero_shot_prompt_template.format(data['question'])
        for i, c in zip(['A', 'B', 'C', 'D', 'E', 'F'], data['choices']):
            mcot_input_str += f'{i}. {c}\n'

        # Images (one-shot: 2; zero-shot: 1)
        exemplar_path = os.path.join('./data/m3cot/images', TRAING_CASE_1['id'] + '.png')
        query_path = os.path.join(
            IMG_FOLDER,
            data['id'] + '.png' if DATA_NAME == 'm3cot' else data['image']
        )

        one_shot_vision = [
            load_and_resize_image(exemplar_path, image_target_size),
            load_and_resize_image(query_path, image_target_size),
        ]
        zero_shot_vision = [load_and_resize_image(query_path, image_target_size)]

        # Prompts
        one_shot_mcot_input_str = mcot_induct + '\n' + mcot_input_str  # two <image> tags total
        zero_shot_mcot_input_str = mcot_input_str                      # one <image> tag total

        # Generate (with error capture per sample)
        try:
            one_shot = calculate_generated_text(one_shot_mcot_input_str, one_shot_vision, return_image_masks=True)
        except Exception as e:
            one_shot = f"[GENERATION_ERROR_one_shot: {repr(e)}]"

        try:
            zero_shot = calculate_generated_text(zero_shot_mcot_input_str, zero_shot_vision, return_image_masks=False)
        except Exception as e:
            zero_shot = f"[GENERATION_ERROR_zero_shot: {repr(e)}]"

        # Write
        oneshot_mcot_output = copy.deepcopy(data); oneshot_mcot_output['pred'] = one_shot
        zeroshot_mcot_output = copy.deepcopy(data); zeroshot_mcot_output['pred'] = zero_shot

        mcot_one_fh.write(json.dumps(oneshot_mcot_output, ensure_ascii=False) + '\n')
        mcot_zero_fh.write(json.dumps(zeroshot_mcot_output, ensure_ascii=False) + '\n')

        # Optional: free some memory between samples
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    mcot_one_fh.close()
    mcot_zero_fh.close()
    print(f"Saved:\n  {mcot_one_path}\n  {mcot_zero_path}")

if __name__ == '__main__':
    main()
