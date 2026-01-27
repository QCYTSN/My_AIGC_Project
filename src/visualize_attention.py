import torch
from diffusers import StableDiffusionPipeline
import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt

# === 配置 ===
device = "cuda"
model_path = "./sd-v1-5" 
output_dir = "results/research_week2"
os.makedirs(output_dir, exist_ok=True)

# === 核心黑科技：Attention 钩子 ===
class AttentionStore:
    def __init__(self):
        self.step_store = {} 

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if not is_cross:
            return
        
        # 我们寻找 16x16 = 256 像素的层
        pixels = attn.shape[1]
        if pixels == 16 ** 2: 
            key = f"{place_in_unet}_{pixels}"
            if key not in self.step_store:
                self.step_store[key] = []
            self.step_store[key].append(attn)

    def reset(self):
        self.step_store = {}

# ✅ 稳健的注册方法
def register_attention_control(pipe, controller):
    def ca_forward(self, place_in_unet):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            is_cross = encoder_hidden_states is not None
            
            query = self.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            
            # 🚨 偷取 Attention Map
            controller(attention_probs, is_cross, place_in_unet)
            
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)
            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states
            
        return forward

    print(">>> 正在给 UNet 安装探针...")
    hook_count = 0
    for name, module in pipe.unet.named_modules():
        if name.endswith("attn2"):
            if "down" in name: place = "down"
            elif "mid" in name: place = "mid"
            elif "up" in name: place = "up"
            else: continue
            
            if hasattr(module, "to_q"):
                module.forward = ca_forward(module, place)
                hook_count += 1
    
    if hook_count == 0:
        print("❌ 警告：没有挂载到任何 Attention 层！请检查模型结构。")
    else:
        print(f"✅ 成功挂载了 {hook_count} 个 Attention 层。")

# === 可视化工具 (修复了 float16 问题) ===
def visualize_attention(pipe, prompt, target_word, seed=42):
    print(f"\n👀 正在探测 Prompt: '{prompt}' 中单词 '{target_word}' 的注意力...")
    
    controller = AttentionStore()
    register_attention_control(pipe, controller)
    
    generator = torch.Generator(device).manual_seed(seed)
    image = pipe(prompt, num_inference_steps=30, generator=generator).images[0]
    image.save(f"{output_dir}/vis_base_{target_word}.png")
    
    # 聚合 Attention Maps
    attention_maps = []
    for key in controller.step_store:
        attn = torch.cat(controller.step_store[key], dim=0) 
        # 只在 batch/heads 维度平均，保留 Pixels 维度
        attn = attn.mean(0) 
        attention_maps.append(attn)
    
    if not attention_maps:
        print("❌ 依然没有捕获到 Attention Map。这很奇怪。")
        return

    # 堆叠所有层的 map 并取平均
    global_attn = torch.stack(attention_maps).mean(0) 
    
    # 找到目标单词的 Token ID
    tokenizer = pipe.tokenizer
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    
    target_idx = -1
    print(f"Token列表: {[decoder([t]) for t in tokens]}")
    
    for idx, token in enumerate(tokens):
        decoded = decoder([token]).strip().lower()
        if target_word.lower() in decoded:
            target_idx = idx
            break
            
    if target_idx == -1:
        print(f"❌ 警告：没找到单词 '{target_word}' 的 token。")
        return

    print(f"✅ 锁定 Token ID: {target_idx} ('{decoder([tokens[target_idx]])}')")
    
    # 提取热力图
    # 🚨🚨🚨 【修复核心】强制转为 float32，防止 OpenCV 报错 🚨🚨🚨
    attn_map = global_attn[:, target_idx].reshape(16, 16).cpu().numpy().astype(np.float32)
    
    # 渲染
    attn_heatmap = cv2.resize(attn_map, (512, 512))
    # 归一化
    attn_heatmap = (attn_heatmap - attn_heatmap.min()) / (attn_heatmap.max() - attn_heatmap.min())
    
    # 画图
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image)
    axs[0].set_title("Generated Image")
    axs[0].axis("off")
    
    axs[1].imshow(image)
    axs[1].imshow(attn_heatmap, cmap='jet', alpha=0.6) 
    axs[1].set_title(f"Attention: {target_word}")
    axs[1].axis("off")
    
    save_path = f"{output_dir}/vis_heatmap_{target_word}.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"🎉 可视化完成：{save_path}")

# === 主程序 ===
if __name__ == "__main__":
    # 加载模型
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    
    # 实验 1：帽子
    prompt = "a photo of a cute cat wearing a red hat"
    visualize_attention(pipe, prompt, "hat", seed=2024)
    
    # 实验 2：围巾
    prompt_scarf = "a photo of a dog wearing a blue scarf"
    visualize_attention(pipe, prompt_scarf, "scarf", seed=2024)