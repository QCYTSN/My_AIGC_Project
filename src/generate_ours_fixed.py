import torch
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor
import json
import numpy as np
from PIL import Image
import os

# === ⚙️ 配置区域 ===
device = "cuda"
model_path = "./sd-v1-5"

# 输入路径：确保同时包含猫和狗
input_dirs = [
    "results/baseline_eval_v5/cat_hat/json",
    "results/baseline_eval_v5/dog_scarf/json", 
]

# 输出路径 (依然覆盖这个文件夹)
output_root = "results/final_comparison_fixed"
os.makedirs(output_root, exist_ok=True)

# === 1. 核心处理器 (回退到温和增强版 V2.4) ===
class SpatialGateAttnProcessor_Balanced:
    def __init__(self, target_token_ids, bbox, width=512, height=512):
        self.target_token_ids = target_token_ids
        self.bbox = bbox
        self.W = width
        self.H = height
        
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None: encoder_hidden_states = hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # --- 💉 控制逻辑 ---
        spatial_pixels = attention_probs.shape[1]
        spatial_res = int(np.sqrt(spatial_pixels))
        
        # 针对 16x16 (256) 和 32x32 (1024) 进行控制
        if spatial_res in [16, 32]:
            # 1. 制作 Mask
            mask = torch.zeros((spatial_res, spatial_res), device=attention_probs.device)
            scale_x, scale_y = spatial_res / self.W, spatial_res / self.H
            
            x1 = int(self.bbox[0] * scale_x)
            y1 = int(self.bbox[1] * scale_y)
            x2 = int(self.bbox[2] * scale_x)
            y2 = int(self.bbox[3] * scale_y)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(spatial_res, x2), min(spatial_res, y2)
            
            mask[y1:y2, x1:x2] = 1.0
            mask_flat = mask.view(1, -1, 1) # [1, Pixels, 1]
            
            # 2. 温和控制
            for token_id in self.target_token_ids:
                current_map = attention_probs[:, :, token_id]
                
                # A. 抑制外部: 乘 0 (Hard Gating) - 这个保留，防止背景泄露
                masked_map = current_map * mask_flat.squeeze()
                
                # B. 温和增强内部: 乘 5.0 (Balanced Boosting)
                # 💡 回退点：从 20.0 改回 5.0，让猫显形
                attention_probs[:, :, token_id] = masked_map * 5.0 

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

# === 2. 动态 Prompt 管理 (保留之前的优化，用更具体的词) ===
def get_prompt(category):
    if category == "hat": 
        return "a photo of a cute cat wearing a red knitted beanie"
    
    if category == "scarf": 
        return "a photo of a dog wearing a blue winter scarf"
        
    return ""

def get_target_words(category):
    if category == "hat": 
        return ["red", "knitted", "beanie"]
        
    if category == "scarf": 
        return ["blue", "scarf"]
        
    return []

# === 3. 批量执行逻辑 ===
def run_batch():
    print(">>> 🚀 启动平衡版 (x5.0) 生成脚本 (猫+狗)...")
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, safety_checker=None).to(device)
    
    for json_dir in input_dirs:
        if not os.path.exists(json_dir): continue
        task_name = json_dir.split("/")[-2] 
        save_path = f"{output_root}/{task_name}"
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\n📂 处理任务组: {task_name}")
        
        files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])
        for f_name in files:
            with open(f"{json_dir}/{f_name}", 'r') as f:
                meta = json.load(f)
            
            seed = meta['seed']
            bbox = meta['target_bbox']
            category = meta['category']
            
            prompt = get_prompt(category)
            target_words = get_target_words(category)
            
            tokenizer = pipe.tokenizer
            tokens = tokenizer.encode(prompt)
            decoder = tokenizer.decode
            
            target_ids = []
            for idx, token in enumerate(tokens):
                decoded = decoder([token]).strip().lower()
                for word in target_words:
                    if word.lower() in decoded:
                        target_ids.append(idx)
            
            if not target_ids:
                print(f"⚠️ Warning: 没找到目标词 Tokens! (Prompt: {prompt})")
            
            # 挂载平衡版处理器
            processor = SpatialGateAttnProcessor_Balanced(target_ids, bbox)
            attn_procs = {}
            for name in pipe.unet.attn_processors.keys():
                if "attn2" in name: 
                    attn_procs[name] = processor
                else: 
                    attn_procs[name] = AttnProcessor()
            pipe.unet.set_attn_processor(attn_procs)
            
            generator = torch.Generator(device).manual_seed(seed)
            image = pipe(prompt, num_inference_steps=30, generator=generator).images[0]
            
            img_name = f_name.replace(".json", ".png")
            image.save(f"{save_path}/{img_name}")
            print(f"   ✅ 生成: {img_name}")

if __name__ == "__main__":
    run_batch()