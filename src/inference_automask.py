import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
# 🚨 修正点 1: 这里的类名修正为 OwlViT...
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image, ImageDraw
import numpy as np
import os

# === 配置 ===
base_model = "./sd-v1-5"
lora_dog = "./outputs/lora_dog"
lora_glass = "./outputs/lora_sunglasses"
save_dir = "results/automask_experiment"
os.makedirs(save_dir, exist_ok=True)

device = "cuda"

# === 1. 初始化模型群 ===
print(">>> 🧠 初始化模型群...")

# A. 生成模型 (T2I)
pipe_t2i = StableDiffusionPipeline.from_pretrained(
    base_model, torch_dtype=torch.float16, safety_checker=None
).to(device)
pipe_t2i.load_lora_weights(lora_dog, adapter_name="dog")
pipe_t2i.set_adapters(["dog"], adapter_weights=[1.0])

# B. 感知模型 (OWL-ViT)
# 指向刚才下载好的本地文件夹
local_owl_path = "./models/owlvit-base-patch32" 

print(f">>> 📂 从本地加载感知模型: {local_owl_path} ...")
processor = OwlViTProcessor.from_pretrained(local_owl_path)
model_owl = OwlViTForObjectDetection.from_pretrained(local_owl_path).to(device)

# === 2. 生成底图 ===
print(">>> 🐶 Step 1: 生成底图...")
# 稍微把 prompt 改得简单点，确保能画出正脸，提高检测成功率
prompt_dog = "a photo of a sks dog sitting, front view, looking at camera, high quality"
# 固定 Seed 方便调试
generator = torch.Generator(device).manual_seed(1024)
image_dog = pipe_t2i(prompt_dog, num_inference_steps=30, generator=generator).images[0]
image_dog.save(f"{save_dir}/base_dog.png")

del pipe_t2i
torch.cuda.empty_cache()

# === 3. 智能感知与几何计算 ===
print(">>> 👁️ Step 2: 视觉感知与Mask计算...")

# OWL-ViT 需要文本提示来找物体
texts = [["eyes", "face"]]
inputs = processor(text=texts, images=image_dog, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model_owl(**inputs)

target_sizes = torch.Tensor([image_dog.size[::-1]]).to(device)
# 降低一点阈值 (0.05) 确保能把眼睛找出来
results = processor.post_process_object_detection(outputs, threshold=0.05, target_sizes=target_sizes)[0]

boxes = results["boxes"].cpu().numpy()
labels = results["labels"].cpu().numpy()

eye_boxes = []
for box, label in zip(boxes, labels):
    if label == 0: # label 0 is "eyes"
        eye_boxes.append(box)

# --- 几何算法 ---
mask = Image.new("L", (512, 512), 0)
draw = ImageDraw.Draw(mask)

if len(eye_boxes) >= 1:
    print(f"✅ 检测到 {len(eye_boxes)} 只眼睛！")
    
    x1 = np.min([b[0] for b in eye_boxes])
    y1 = np.min([b[1] for b in eye_boxes])
    x2 = np.max([b[2] for b in eye_boxes])
    y2 = np.max([b[3] for b in eye_boxes])
    
    w = x2 - x1
    h = y2 - y1
    
    # 稍微调整一下扩张系数，防止画太大了
    pad_w = w * 0.3 
    pad_h = h * 0.5 
    
    final_box = [
        max(0, x1 - pad_w), 
        max(0, y1 - pad_h), 
        min(512, x2 + pad_w), 
        min(512, y2 + pad_h * 0.5)
    ]
    
    draw.rectangle(final_box, fill=255)
    print(f"    Mask 区域: {final_box}")
else:
    print("⚠️ 未检测到眼睛！使用兜底策略...")
    draw.rectangle([160, 160, 352, 220], fill=255)

mask.save(f"{save_dir}/auto_mask.png")

# === 4. 局部重绘 ===
print(">>> 🕶️ Step 3: 注入墨镜概念...")

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    base_model, torch_dtype=torch.float16, safety_checker=None
).to(device)

pipe_inpaint.load_lora_weights(lora_glass, adapter_name="sunglasses")
pipe_inpaint.set_adapters(["sunglasses"], adapter_weights=[1.0])

final_image = pipe_inpaint(
    prompt="a photo of a sks sunglasses on a dog face, black frame, transparent glass",
    negative_prompt="eyes, fur, messy",
    image=image_dog,
    mask_image=mask,
    strength=0.9,
    num_inference_steps=40
).images[0]

final_image.save(f"{save_dir}/final_result.png")
print(f"🎉 闭环完成！结果保存在 {save_dir}/final_result.png")