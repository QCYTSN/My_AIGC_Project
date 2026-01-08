import torch
from diffusers import StableDiffusionPipeline
import os

# === 配置 ===
base_model = "./sd-v1-5"
lora_glass = "./outputs/lora_sunglasses"
save_dir = "results/check_lora"
os.makedirs(save_dir, exist_ok=True)

# === 加载模型 ===
pipe = StableDiffusionPipeline.from_pretrained(
    base_model, torch_dtype=torch.float16, safety_checker=None
).to("cuda")

print(">>> 📥 加载墨镜 LoRA...")
pipe.load_lora_weights(lora_glass)

# === 生成测试 ===
# 注意：Prompt 里去掉了 "close up"，加上了 "black frame" (黑框) 增加特征稳定性
prompt = "a photo of a sks sunglasses, black frame, transparent glass, white background, high quality"
negative_prompt = "low quality, blur, distortion"

print(">>> 🧪 正在单独测试 LoRA 质量...")
for i in range(4):
    image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30).images[0]
    image.save(f"{save_dir}/glass_test_{i}.png")
    print(f"saved: {save_dir}/glass_test_{i}.png")