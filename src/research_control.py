import torch
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor
import json
import numpy as np
from PIL import Image, ImageDraw
import os
import cv2

# === 配置 ===
device = "cuda"
model_path = "./sd-v1-5"
json_source_dir = "results/baseline_eval_v5/cat_hat/json"
output_dir = "results/research_week3_debug"
os.makedirs(output_dir, exist_ok=True)

# === 1. 处理器 V2.4 (含增强逻辑) ===
class SpatialGateAttnProcessorV2_4:
    def __init__(self, target_token_ids, bbox, width=512, height=512):
        self.target_token_ids = target_token_ids
        self.bbox = bbox
        self.W = width
        self.H = height
        self.debug_saved = False 
        self.trigger_count = 0

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args, **kwargs
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # --- 💉 手术开始 ---
        spatial_pixels = attention_probs.shape[1]
        spatial_res = int(np.sqrt(spatial_pixels))
        
        # 只要分辨率是 16 或 32，就执行控制
        if spatial_res in [16, 32]:
            self.trigger_count += 1
            
            mask = torch.zeros((spatial_res, spatial_res), device=attention_probs.device)
            scale_x = spatial_res / self.W
            scale_y = spatial_res / self.H
            
            x1 = int(self.bbox[0] * scale_x)
            y1 = int(self.bbox[1] * scale_y)
            x2 = int(self.bbox[2] * scale_x)
            y2 = int(self.bbox[3] * scale_y)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(spatial_res, x2), min(spatial_res, y2)
            
            # 1. 制作 Mask
            mask[y1:y2, x1:x2] = 1.0
            
            # Debug: 打印一次确认
            if not self.debug_saved and spatial_res == 16:
                print(f"🔥 DEBUG: 拦截 {spatial_res}x{spatial_res} 层 | 框内区域: x[{x1}:{x2}] y[{y1}:{y2}]")
                self.debug_saved = True

            mask_flat = mask.view(1, -1, 1) # [1, Pixels, 1]
            
            # 2. 核心控制逻辑 (增强 + 抑制)
            for token_id in self.target_token_ids:
                current_probs = attention_probs[:, :, token_id]
                
                # A. 抑制框外: 乘以 0 (Hard Gating)
                masked_probs = current_probs * mask_flat.squeeze()
                
                # B. 增强框内: 乘以 5.0 (Amplification)
                # 这一步是为了解决“猫没有帽子”的问题，强迫它在框内激活
                amplified_probs = masked_probs * 5.0
                
                attention_probs[:, :, token_id] = amplified_probs

        # --- 手术结束 ---

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

# === 2. 注册工具 (修复匹配逻辑) ===
def register_spatial_control(pipe, target_words, bbox):
    tokenizer = pipe.tokenizer
    prompt = "a photo of a cute cat wearing a red hat" 
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    
    target_ids = []
    print(f"\n🔎 Token 匹配:")
    for idx, token in enumerate(tokens):
        decoded = decoder([token]).strip().lower()
        for word in target_words:
            if word.lower() in decoded:
                print(f"   ✅ ID {idx} -> '{decoded}'")
                target_ids.append(idx)
    
    custom_processor = SpatialGateAttnProcessorV2_4(target_ids, bbox)
    
    from diffusers.models.attention_processor import AttnProcessor
    default_processor = AttnProcessor()

    attn_procs = {}
    hook_count = 0
    
    # 🚨 关键修复：遍历所有处理器，只要名字含 attn2 就替换
    print("\n🛠️ 开始挂载处理器...")
    for name in pipe.unet.attn_processors.keys():
        if "attn2" in name:
            attn_procs[name] = custom_processor
            hook_count += 1
            # 只打印前3个作为示例，避免刷屏
            if hook_count <= 3:
                print(f"   🔗 挂载到: {name}")
        else:
            attn_procs[name] = default_processor
            
    pipe.unet.set_attn_processor(attn_procs)
    print(f"🔌 最终结果: 已成功挂载控制器到 {hook_count} 个 Cross-Attention 层")
    
    if hook_count == 0:
        print("❌ 严重错误：依然没有挂载到任何层！请检查 Key 名称。")
        # 打印所有 Key 供调试
        print(list(pipe.unet.attn_processors.keys())[:5])
        
    return custom_processor

# === 3. 主程序 ===
if __name__ == "__main__":
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    
    # 读取第一张 JSON
    json_files = sorted(os.listdir(json_source_dir))
    target_json = json_files[0] 
    with open(f"{json_source_dir}/{target_json}", 'r') as f:
        meta = json.load(f)
    
    seed = meta['seed']
    bbox = meta['target_bbox'] 
    
    # 目标词
    target_words = ["red", "hat"] 
    prompt = "a photo of a cute cat wearing a red hat"
    
    # A. Baseline
    print("\n🧪 生成 Baseline...")
    pipe.unet.set_default_attn_processor()
    generator = torch.Generator(device).manual_seed(seed)
    img_baseline = pipe(prompt, num_inference_steps=30, generator=generator).images[0]
    img_baseline.save(f"{output_dir}/compare_baseline.png")
    
    # B. Ours
    print(f"\n💉 注入控制 (抑制外部 + 增强内部)...")
    processor = register_spatial_control(pipe, target_words, bbox)
    
    generator = torch.Generator(device).manual_seed(seed)
    img_ours = pipe(prompt, num_inference_steps=30, generator=generator).images[0]
    img_ours.save(f"{output_dir}/compare_ours_v2_4.png")
    
    print(f"\n📊 统计: 控制逻辑触发了 {processor.trigger_count} 次")
    
    # C. 画框
    img_vis = img_baseline.copy()
    draw = ImageDraw.Draw(img_vis)
    draw.rectangle(bbox, outline="green", width=5)
    img_vis.save(f"{output_dir}/vis_target_box.png")

    print(f"🎉 修复完成！请检查 {output_dir}")