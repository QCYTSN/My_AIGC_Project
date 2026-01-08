import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image, ImageFilter
from torchvision import transforms
import torch.nn.functional as F
import os

# === 1. 配置与加载 ===
base_model = "./sd-v1-5"
lora_dog = "./outputs/lora_dog"
lora_glass = "./outputs/lora_sunglasses"
mask_glass_path = "./data/masks/mask_sunglasses.png" 
# 注意：我们只需要墨镜的 mask，剩下的自动归为“背景/狗”

# 加载 Pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    base_model, torch_dtype=torch.float16, safety_checker=None
).to("cuda")

# 使用 DDIM 调度器
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# 加载 LoRA
print(">>> 📥 加载 LoRA...")
pipe.load_lora_weights(lora_dog, adapter_name="dog")
pipe.load_lora_weights(lora_glass, adapter_name="sunglasses")

# === 2. 准备 Mask (互斥逻辑) ===
def load_mask_tensor(path):
    mask = Image.open(path).convert("L").resize((512, 512))
    tensor = transforms.ToTensor()(mask).to("cuda", dtype=torch.float16)
    return tensor.unsqueeze(0) # [1, 1, 512, 512]

# 我们只需要加载墨镜的 Mask
mask_glass = load_mask_tensor(mask_glass_path)

# ✨ 核心修正 1：构建互斥 Mask ✨
# mask_glass: 墨镜区域 (1), 其他区域 (0)
# mask_bg: 其他区域 (1), 墨镜区域 (0) --> 这个区域用来画狗和背景
mask_bg = 1.0 - mask_glass

# === 3. 准备 Embeddings ===
def get_embeds(prompt):
    inputs = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    return pipe.text_encoder(inputs.input_ids.to("cuda"))[0]

neg_embeds = get_embeds("blur, low quality, distortion, ugly, bad anatomy") 
# 平行世界 A (背景+狗): 
embeds_dog = get_embeds("a photo of a sks dog in a garden, high quality, 8k")
# 平行世界 B (墨镜):
embeds_glass = get_embeds("a photo of a sks sunglasses, transparent glass, highly detailed, close up")

# === 4. 手写生成循环 ===
print(">>> 🚀 开始区域化生成 (修复版)...")

# 初始化随机噪声
latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16)

# ✨ 核心修正 2：缩放初始噪声 (防止电视雪花的关键！) ✨
latents = latents * pipe.scheduler.init_noise_sigma

pipe.scheduler.set_timesteps(50)

with torch.no_grad():
    for t in pipe.scheduler.timesteps:
        # --- 世界 A: 狗 + 背景 ---
        pipe.set_adapters(["dog"], adapter_weights=[1.0])
        
        input_cat = torch.cat([latents] * 2)
        input_cat = pipe.scheduler.scale_model_input(input_cat, t) # 缩放输入
        embeds_cat = torch.cat([neg_embeds, embeds_dog])
        
        noise_pred_A = pipe.unet(input_cat, t, encoder_hidden_states=embeds_cat).sample
        noise_uncond, noise_text_A = noise_pred_A.chunk(2)
        noise_pred_A = noise_uncond + 7.5 * (noise_text_A - noise_uncond)

        # --- 世界 B: 墨镜 ---
        pipe.set_adapters(["sunglasses"], adapter_weights=[1.0])
        
        input_cat = torch.cat([latents] * 2)
        input_cat = pipe.scheduler.scale_model_input(input_cat, t)
        embeds_cat = torch.cat([neg_embeds, embeds_glass])
        
        noise_pred_B = pipe.unet(input_cat, t, encoder_hidden_states=embeds_cat).sample
        noise_uncond, noise_text_B = noise_pred_B.chunk(2)
        noise_pred_B = noise_uncond + 7.5 * (noise_text_B - noise_uncond)

        # --- 融合 ---
        # 缩小 Mask 到 Latent 尺寸
        mask_glass_small = F.interpolate(mask_glass, size=(64, 64), mode="nearest")
        mask_bg_small = 1.0 - mask_glass_small # 确保无缝衔接
        
        # 拼接噪声: (狗噪声 * 狗区域) + (墨镜噪声 * 墨镜区域)
        merged_noise = (noise_pred_A * mask_bg_small) + (noise_pred_B * mask_glass_small)
        
        # 更新 Latents
        latents = pipe.scheduler.step(merged_noise, t, latents).prev_sample

# === 5. 解码并保存 ===
print(">>> 🖼️ 解码图像...")
image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
# ✨ 核心修正 3: 解除梯度锁
image = image.detach()
image = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=[True])[0]

save_path = "results/regional_result_fixed.png"
image.save(save_path)
print(f"✅ 成功！结果保存在 {save_path}")