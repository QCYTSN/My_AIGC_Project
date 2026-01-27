import os
import torch
import numpy as np
import json
import random
from PIL import Image, ImageDraw, ImageFilter
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, AutoencoderKL
from ultralytics import YOLO

print(">>> 🚀 启动 Baseline V5.0 (色块引导终极版) <<<")
print(">>> 策略：半透明染色 -> 降低生成难度 -> 仅做材质渲染 -> 杜绝人脸幻觉")

# === ⚙️ 全局配置 ===
device = "cuda"
base_model = "./sd-v1-5"
lora_dog = "./outputs/lora_dog"
vae_path = "./models/sd-vae-ft-mse"
yolo_path = "/root/My_AIGC_Project/models/yolo_result/eye_detector/weights/best.pt"

output_dir = "results/baseline_eval_v5"
os.makedirs(output_dir, exist_ok=True)

# === 🎯 任务定义 ===
TASKS = [
    {
        "name": "corgi_sunglasses",
        "category": "glasses",
        "t2i_prompt": "a photo of a sks dog sitting, front view, looking at camera, high quality, 8k",
        "inpaint_prompt": "black sunglasses, dark reflective lenses, glossy frame, realistic shading, 8k",
        "negative": "box, cloth, case, eyes, animal face, human, skin",
        "strategy": "sketch", 
        "strength": 0.65
    },
    {
        "name": "cat_hat",
        "category": "hat",
        "t2i_prompt": "a photo of a cute cat sitting, front view, looking at camera, high quality, 8k",
        # 💡 Prompt 变了：不再说 "wearing", 只说 "texture of red beanie"
        "inpaint_prompt": "close up texture of a red wool beanie hat, knitted fabric, realistic lighting, 8k",
        "negative": "face, eyes, mouth, human, skin, model, man, woman, hair, ear",
        "strategy": "color_guide", # 新策略
        "guide_color": (200, 20, 20, 160), # 红色，透明度 160/255
        "strength": 0.80
    },
    {
        "name": "dog_scarf",
        "category": "scarf",
        "t2i_prompt": "a photo of a golden retriever dog sitting, front view, looking at camera, high quality, 8k",
        # 💡 Prompt 变了：只强调材质
        "inpaint_prompt": "close up texture of a blue winter scarf, wool fabric, thick knitting, cozy, 8k",
        "negative": "face, eyes, mouth, human, skin, model, man, woman, chin, neck",
        "strategy": "color_guide", # 新策略
        "guide_color": (20, 50, 200, 160), # 蓝色，透明度 160/255
        "strength": 0.80
    }
]

TARGET_PER_TASK = 20

def load_models():
    print(">>> 🧠 初始化模型...")
    vae = AutoencoderKL.from_pretrained(vae_path).to(device, torch.float16)
    model_yolo = YOLO(yolo_path)
    
    pipe_t2i = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16, safety_checker=None, vae=vae
    ).to(device)
    pipe_t2i.load_lora_weights(lora_dog, adapter_name="dog")
    
    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16, safety_checker=None, vae=vae
    ).to(device)
    
    pipe_t2i.set_progress_bar_config(disable=True)
    pipe_inpaint.set_progress_bar_config(disable=True)
    return model_yolo, pipe_t2i, pipe_inpaint

# 🎨 1. 草图绘制 (墨镜专用)
def draw_synthetic_sketch(base_image, eyes_bbox):
    overlay = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    min_x, min_y, max_x, max_y = eyes_bbox
    pad_x = (max_x - min_x) * 0.15
    pad_y = (max_y - min_y) * 0.5
    box = [min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y]
    
    color = (40, 40, 40, 230)
    w = box[2] - box[0]
    h = box[3] - box[1]
    lens_w = w * 0.45
    draw.ellipse([box[0], box[1], box[0]+lens_w, box[3]], fill=color) 
    draw.ellipse([box[2]-lens_w, box[1], box[2], box[3]], fill=color) 
    draw.line([(box[0]+lens_w*0.8, box[1]+h*0.4), (box[2]-lens_w*0.8, box[1]+h*0.4)], fill=color, width=5)
    
    base_image = base_image.convert("RGBA")
    comp = Image.alpha_composite(base_image, overlay)
    return comp.convert("RGB"), box

# 🖌️ 2. 色块引导 (帽子/围巾专用)
def apply_color_guide(base_image, bbox, color):
    # 创建一个和原图一样大的透明层
    overlay = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # 画一个半透明的矩形 (或者圆角矩形) 覆盖在目标区域
    # 这会把狗的毛发“染色”，让 AI 觉得“这里已经是蓝色的了”
    draw.rounded_rectangle(bbox, radius=20, fill=color)
    
    # 叠加
    base_image = base_image.convert("RGBA")
    comp = Image.alpha_composite(base_image, overlay)
    return comp.convert("RGB")

def zoom_inpaint(pipe, base_image, mask_image, prompt, negative_prompt, strength=0.8):
    bbox = mask_image.getbbox()
    if not bbox: return base_image
    
    padding = 60
    left, top, right, bottom = bbox
    width, height = base_image.size
    crop_box = (max(0, left-padding), max(0, top-padding), min(width, right+padding), min(height, bottom+padding))
    
    img_crop = base_image.crop(crop_box).resize((512, 512), Image.Resampling.LANCZOS)
    mask_crop = mask_image.crop(crop_box).resize((512, 512), Image.Resampling.NEAREST)
    
    res_crop = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        image=img_crop, 
        mask_image=mask_crop,
        strength=strength,     
        num_inference_steps=50, 
        guidance_scale=9.0        
    ).images[0]
    
    res_crop = res_crop.resize(base_image.crop(crop_box).size, Image.Resampling.LANCZOS)
    final_img = base_image.copy()
    final_img.paste(res_crop, crop_box) 
    return final_img

def run_tasks():
    model_yolo, pipe_t2i, pipe_inpaint = load_models()
    
    for task in TASKS:
        task_name = task["name"]
        print(f"\n>>> 📦 任务: {task_name} | 策略: {task['strategy']} | Strength: {task['strength']}")
        
        save_path = f"{output_dir}/{task_name}"
        os.makedirs(f"{save_path}/images", exist_ok=True)
        os.makedirs(f"{save_path}/json", exist_ok=True)
        
        count = 0
        while count < TARGET_PER_TASK:
            seed = random.randint(0, 99999999)
            generator = torch.Generator(device).manual_seed(seed)
            
            # 1. 生成底图
            img_base = pipe_t2i(task["t2i_prompt"], num_inference_steps=30, generator=generator).images[0]
            W, H = img_base.size
            
            # 2. 检测眼睛
            results = model_yolo.predict(img_base, conf=0.3, verbose=False)
            boxes = results[0].boxes
            if len(boxes) < 2: continue 
            
            all_boxes = boxes.xyxy.cpu().numpy()
            min_x, min_y = np.min(all_boxes[:, 0]), np.min(all_boxes[:, 1])
            max_x, max_y = np.max(all_boxes[:, 2]), np.max(all_boxes[:, 3])
            
            eyes_bbox = [float(min_x), float(min_y), float(max_x), float(max_y)]
            eye_w, eye_h = max_x - min_x, max_y - min_y
            
            # 3. 计算目标区域
            target_bbox = []
            img_input = None 
            
            if task["strategy"] == "sketch": 
                img_input, target_bbox = draw_synthetic_sketch(img_base, eyes_bbox)
                
            elif task["strategy"] == "color_guide": 
                if task["category"] == "hat":
                    pad_x = eye_w * 0.4
                    hat_h = eye_h * 3.5 
                    target_bbox = [min_x - pad_x, min_y - hat_h, max_x + pad_x, min_y - eye_h*0.2]
                elif task["category"] == "scarf":
                    pad_x = eye_w * 0.5
                    neck_y = max_y + eye_h * 0.5
                    scarf_h = eye_h * 3.0
                    target_bbox = [min_x - pad_x, neck_y, max_x + pad_x, neck_y + scarf_h]
                
                # 🚨🚨🚨 【修复核心】双向坐标钳制 (Clamping) 🚨🚨🚨
                # 1. 确保 x0, y0 不小于 0
                x0 = max(0, float(target_bbox[0]))
                y0 = max(0, float(target_bbox[1]))
                
                # 2. 确保 x1, y1 既不小于 0，也不大于 W/H
                x1 = max(0, min(W, float(target_bbox[2])))
                y1 = max(0, min(H, float(target_bbox[3])))
                
                # 3. 【终极保险】确保 x1 >= x0 且 y1 >= y0
                # 如果帽子完全跑出去了导致 y1 < y0，就强制让它们相等（变成一条线，避免报错）
                if x1 < x0: x1 = x0
                if y1 < y0: y1 = y0
                
                target_bbox = [x0, y0, x1, y1]
                
                # 应用色块引导
                # 如果 box 高度或宽度为 0，rounded_rectangle 可能会有问题，加个判断
                if (x1 - x0) > 1 and (y1 - y0) > 1:
                    img_input = apply_color_guide(img_base, target_bbox, task["guide_color"])
                else:
                    # 如果目标区域无效（完全在图外），直接用原图，跳过引导
                    print("   ⚠️ 目标区域在图外，跳过染色引导...")
                    img_input = img_base

            # 4. 类型清洗 (虽然上面已经float过了，保持统一)
            target_bbox = [float(x) for x in target_bbox]

            # 5. 构建 Mask
            mask = Image.new("L", img_base.size, 0)
            draw = ImageDraw.Draw(mask)
            
            # 只有当 bbox 有效时才画 Mask
            if (target_bbox[2] - target_bbox[0]) > 1 and (target_bbox[3] - target_bbox[1]) > 1:
                draw.rounded_rectangle(target_bbox, radius=20, fill=255)
            
            blur_radius = 5 if task["category"] == "glasses" else 15
            mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
            
            # 6. Inpaint
            final_img = zoom_inpaint(
                pipe_inpaint, 
                img_input, 
                mask, 
                task["inpaint_prompt"], 
                task["negative"],
                strength=task["strength"]
            )
            
            final_img.save(f"{save_path}/images/{count:03d}.png")
            
            meta = {
                "seed": seed, "category": task["category"],
                "anchor_eyes": eyes_bbox, "target_bbox": target_bbox
            }
            with open(f"{save_path}/json/{count:03d}.json", "w") as f:
                json.dump(meta, f, indent=4)
                
            print(f"   ✅ [{task_name}] {count+1}/{TARGET_PER_TASK} | Seed: {seed}")
            count += 1
            
if __name__ == "__main__":
    run_tasks()