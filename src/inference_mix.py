import torch
from diffusers import StableDiffusionPipeline
import os

# === 1. 配置路径 ===
base_model = "./sd-v1-5"
lora_dog_path = "./outputs/lora_dog"
lora_sunglasses_path = "./outputs/lora_sunglasses"

# === 2. 提示词 (关键！) ===
# 我们同时使用了两个触发词：sks dog 和 sks sunglasses
prompt = "a photo of a sks dog wearing sks sunglasses, in a garden"
negative_prompt = "blur, low quality, distortion, ugly, extra legs"

print("=== 🚀 正在加载底模... ===")
pipe = StableDiffusionPipeline.from_pretrained(
    base_model, 
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

# === 3. 加载两个 LoRA (重点) ===
print("=== 正在混合 LoRA... ===")

# 加载第一个：狗
pipe.load_lora_weights(lora_dog_path, adapter_name="dog")

# 加载第二个：墨镜
pipe.load_lora_weights(lora_sunglasses_path, adapter_name="sunglasses")

# 激活两个适配器，权重都设为 1.0 (你可以尝试调整这个比例，比如 [0.8, 1.0])
pipe.set_adapters(["dog", "sunglasses"], adapter_weights=[1.0, 1.0])

# === 4. 生成测试 ===
print(f"=== 正在生成: {prompt} ===")
save_dir = "results/mix_test"
os.makedirs(save_dir, exist_ok=True)

# 生成 4 张图看看效果
for i in range(4):
    seed = 2024 + i
    image = pipe(
        prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=50, 
        guidance_scale=7.5,
        cross_attention_kwargs={"scale": 1.0}, # 全局 LoRA 强度
        generator=torch.Generator("cuda").manual_seed(seed)
    ).images[0]
    
    save_path = f"{save_dir}/mix_result_{i}.png"
    image.save(save_path)
    print(f"✅ 图片已保存: {save_path}")

print("=== 🎉 实验结束，请去查看结果！ ===")