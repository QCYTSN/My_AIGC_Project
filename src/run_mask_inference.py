import torch
from PIL import Image
from torchvision import transforms
from pipeline_mask import MaskStableDiffusionPipeline # 引入刚才写好的 Pipeline
import os

# === 1. 准备路径 ===
base_model = "./sd-v1-5"
lora_dog = "./outputs/lora_dog"
lora_glass = "./outputs/lora_sunglasses"
mask_dog_path = "./data/masks/mask_dog.png"
mask_glass_path = "./data/masks/mask_sunglasses.png"

prompt = "a photo of a sks dog wearing sks sunglasses"
# ⚠️ 注意：这里我们简化了单词匹配，确保 mask 的 key (比如 "dog") 能在 prompt 里找到

# === 2. 加载 Pipeline ===
print(">>> 🚀 加载 Mask Pipeline...")
pipe = MaskStableDiffusionPipeline.from_pretrained(
    base_model, 
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

# 加载 LoRA
print(">>> 📥 加载双 LoRA...")
pipe.load_lora_weights(lora_dog, adapter_name="dog")
pipe.load_lora_weights(lora_glass, adapter_name="sunglasses")
pipe.set_adapters(["dog", "sunglasses"], adapter_weights=[1.0, 1.0])

# === 3. 准备 Masks ===
# 我们需要把图片变成 Tensor (1, 1, 512, 512)
def load_mask(path):
    mask = Image.open(path).convert("L") # 转黑白
    mask = mask.resize((512, 512))
    tensor = transforms.ToTensor()(mask) # 变成 [0, 1] 的 tensor
    tensor = tensor.unsqueeze(0) # [1, 1, 512, 512]
    return tensor

mask_config = {
    "dog": load_mask(mask_dog_path),
    "sunglasses": load_mask(mask_glass_path)
}

# === 4. 生成 ===
print(f">>> 🎨 开始生成: {prompt}")
output_dir = "results/mask_test"
os.makedirs(output_dir, exist_ok=True)

for i in range(4):
    seed = 2024 + i
    image = pipe(
        prompt=prompt,
        mask_config=mask_config, # <--- 传入我们的 Mask 配置
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=torch.Generator("cuda").manual_seed(seed)
    ).images[0]
    
    save_path = f"{output_dir}/mask_result_{i}.png"
    image.save(save_path)
    print(f"✅ 保存结果: {save_path}")

print(">>> 🎉 实验结束！快去 results/mask_test 看看有没有奇迹发生！")