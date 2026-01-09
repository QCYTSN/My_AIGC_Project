import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from transformers import AutoProcessor, AutoModelForCausalLM

# === ⚙️ 全局配置 ===
device = "cuda"
base_model = "./sd-v1-5"
lora_dog = "./outputs/lora_dog"
lora_glass = "./outputs/lora_sunglasses"

# 继续使用 Large-FT
florence_model_path = "./models/Florence-2-large-ft"

save_dir = "results/florence_final"
os.makedirs(save_dir, exist_ok=True)

# === 1. 加载模型 ===
print(">>> 🧠 加载模型群 (Final Version)...")

# A. Florence-2
processor = AutoProcessor.from_pretrained(florence_model_path, trust_remote_code=True)
model_florence = AutoModelForCausalLM.from_pretrained(
    florence_model_path, 
    torch_dtype=torch.float16, 
    trust_remote_code=True
).to(device)

# B. SD T2I
pipe_t2i = StableDiffusionPipeline.from_pretrained(
    base_model, torch_dtype=torch.float16, safety_checker=None
).to(device)
pipe_t2i.load_lora_weights(lora_dog, adapter_name="dog")

print(">>> ✅ 模型加载完毕！")

# === 2. Step 1: 生成底图 ===
print(">>> 🐶 Step 1: 生成底图...")
prompt_dog = "a photo of a sks dog sitting, front view, looking at camera, high quality, 8k"
# 固定 Seed
generator = torch.Generator(device).manual_seed(2024) 

image_dog = pipe_t2i(prompt_dog, num_inference_steps=30, generator=generator).images[0]
image_path = f"{save_dir}/step1_base.png"
image_dog.save(image_path)

del pipe_t2i
torch.cuda.empty_cache()

# === 3. Step 2: Florence-2 寻找头部 ===
print(">>> 👁️ Step 2: 寻找头部定位...")

task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>"
text_input = "head" # 继续找头，因为头最准

inputs = processor(text=task_prompt + text_input, images=image_dog, return_tensors="pt").to(device, torch.float16)

# use_cache=False 保持兼容性
generated_ids = model_florence.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    do_sample=False,
    num_beams=3,
    use_cache=False 
)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
prediction = processor.post_process_generation(
    generated_text, 
    task=task_prompt, 
    image_size=(image_dog.width, image_dog.height)
)
segmentation_results = prediction[task_prompt]

print(f"✅ 检测目标: {segmentation_results['labels']}")

# === 4. 绘制“佐罗面具” Mask (核心逻辑修改) ===
# 创建全黑画布
mask = Image.new("L", image_dog.size, 0)
draw = ImageDraw.Draw(mask)

# 我们只用 Florence-2 的结果来计算位置，不再直接用它的多边形
# 先画一个临时的 Mask 来获取 bbox
temp_mask = Image.new("L", image_dog.size, 0)
temp_draw = ImageDraw.Draw(temp_mask)

for polygon in segmentation_results["polygons"]:
    points = np.array(polygon).reshape(-1, 2)
    points_tuple = [tuple(pt) for pt in points]
    temp_draw.polygon(points_tuple, fill=255)

# 获取头部的边界框 (Left, Top, Right, Bottom)
bbox = temp_mask.getbbox()

if bbox:
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    
    print(f"📐 头部位置: x={left}, y={top}, w={width}, h={height}")

    # --------- 🎨 核心魔法：绘制“佐罗面具” ---------
    # 我们不信任 Florence 的眼睛检测，我们信任“比例学”
    # 在狗的头部中，眼睛通常位于高度的 30% 到 55% 之间
    
    # 1. 设定墨镜区域的上边界 (避开额头)
    eye_top = top + height * 0.30 
    
    # 2. 设定墨镜区域的下边界 (避开鼻子)
    eye_bottom = top + height * 0.55
    
    # 3. 设定左右边界 (稍微往里收一点，避开耳朵根部)
    eye_left = left + width * 0.15
    eye_right = right - width * 0.15
    
    # 4. 绘制一个圆角矩形 (更像墨镜的形状，给 SD 更好的暗示)
    # 这种横条形状会强迫 SD 生成类似 aviator 或 wayfarer 的形状
    draw.rounded_rectangle(
        [(eye_left, eye_top), (eye_right, eye_bottom)], 
        radius=15, 
        fill=255
    )
    print("🕶️ 已生成“佐罗面具”Mask：基于头部比例推算眼睛区域。")
    
else:
    print("⚠️ 未检测到头部，使用全黑 Mask (将导致失败)。")

# 适度膨胀，让边缘融合更好
mask = mask.filter(ImageFilter.MaxFilter(9))
# 高斯模糊，让边界不要太生硬
mask = mask.filter(ImageFilter.GaussianBlur(radius=3))

mask.save(f"{save_dir}/step2_zorro_mask.png")

del model_florence
torch.cuda.empty_cache()

# === 5. Step 3: Inpainting ===
print(">>> 🕶️ Step 3: 佩戴墨镜...")

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    base_model, torch_dtype=torch.float16, safety_checker=None
).to(device)
pipe_inpaint.load_lora_weights(lora_glass, adapter_name="sunglasses")

final_image = pipe_inpaint(
    prompt="a photo of a sks sunglasses on dog eyes, black frame, dark lenses, highly detailed, professional photography",
    # 负面提示词非常重要
    negative_prompt="forehead, ears, nose, fur texture on glass, ugly, messy, distorted frame",
    image=image_dog,
    mask_image=mask,
    strength=1.0, 
    num_inference_steps=50 # 稍微增加步数提升质感
).images[0]

final_image.save(f"{save_dir}/final_result.png")
print(f"🎉 任务完成！结果保存在 {save_dir}/final_result.png")