import torch
import os
import json
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, OwlViTProcessor, OwlViTForObjectDetection
from tqdm import tqdm

# === ⚙️ 配置区域 ===
device = "cuda" if torch.cuda.is_available() else "cpu"

baseline_dir = "results/baseline_eval_v5"       
ours_dir = "results/final_comparison_fixed"     
tasks = ["cat_hat", "dog_scarf"]

# 指向本地模型
clip_model_id = "./models/clip-vit-base-patch32"
owl_model_id = "./models/owlvit-base-patch32"

# === 1. 加载模型 ===
print(">>> 🧠 正在加载评估模型 (本地模式)...")
clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

owl_model = OwlViTForObjectDetection.from_pretrained(owl_model_id).to(device)
owl_processor = OwlViTProcessor.from_pretrained(owl_model_id)

# === 2. 工具函数 ===
def calculate_iou(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def get_clip_score(image, text):
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    return outputs.logits_per_image.item() / 100.0

def detect_object(image, text_query):
    inputs = owl_processor(text=[[text_query]], images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = owl_model(**inputs)
    
    target_sizes = torch.Tensor([image.size[::-1]]).to(device)
    # 🚨 关键修改：阈值降到极低 (0.001)，只求 Top-1
    results = owl_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.001)[0]
    
    if len(results["scores"]) > 0:
        # 无论分数多少，只取最高分的那一个框
        idx = results["scores"].argmax()
        best_box = results["boxes"][idx].cpu().numpy().tolist()
        best_score = results["scores"][idx].item()
        return best_box, best_score
    
    return None, 0.0

# === 3. 评测循环 ===
def evaluate_task(task_name):
    print(f"\n📊 正在评测任务: {task_name}")
    
    # 🚨 关键修改：使用更自然的 Query
    if task_name == "cat_hat":
        prompt_text = "a photo of a cute cat wearing a red knitted beanie"
        detect_query = "a photo of a hat" # 之前测试显示这个分数最高
    elif task_name == "dog_scarf":
        prompt_text = "a photo of a dog wearing a blue winter scarf"
        detect_query = "a photo of a scarf"
    else:
        return
        
    baseline_img_dir = f"{baseline_dir}/{task_name}/images"
    ours_img_dir = f"{ours_dir}/{task_name}"
    json_dir = f"{baseline_dir}/{task_name}/json"
    
    results = {"baseline": {"clip": [], "iou": [], "detected": 0}, 
               "ours": {"clip": [], "iou": [], "detected": 0}}
    
    files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])
    
    for f_name in tqdm(files):
        with open(f"{json_dir}/{f_name}", 'r') as f:
            meta = json.load(f)
        target_bbox = meta['target_bbox']
        img_name = f_name.replace(".json", ".png")
        
        # --- 评测 Baseline ---
        path_b = f"{baseline_img_dir}/{img_name}"
        if os.path.exists(path_b):
            img = Image.open(path_b).convert("RGB")
            results["baseline"]["clip"].append(get_clip_score(img, prompt_text))
            
            pred_box, conf = detect_object(img, detect_query)
            if pred_box:
                iou = calculate_iou(target_bbox, pred_box)
                results["baseline"]["iou"].append(iou)
                # 只要检测到了，且 IoU > 0.3 就算成功
                if iou > 0.3: results["baseline"]["detected"] += 1
            else:
                results["baseline"]["iou"].append(0.0)

        # --- 评测 Ours ---
        path_o = f"{ours_img_dir}/{img_name}"
        if os.path.exists(path_o):
            img = Image.open(path_o).convert("RGB")
            results["ours"]["clip"].append(get_clip_score(img, prompt_text))
            
            pred_box, conf = detect_object(img, detect_query)
            if pred_box:
                iou = calculate_iou(target_bbox, pred_box)
                results["ours"]["iou"].append(iou)
                if iou > 0.3: results["ours"]["detected"] += 1
            else:
                results["ours"]["iou"].append(0.0)

    # === 输出表格 ===
    print("\n" + "="*45)
    print(f"📝 最终定量评估报告: {task_name}")
    print("="*45)
    
    def print_metric(name, data):
        avg_clip = np.mean(data["clip"]) if data["clip"] else 0
        avg_iou = np.mean(data["iou"]) if data["iou"] else 0
        acc = data["detected"] / len(files) if files else 0
        print(f"[{name}]")
        print(f"  CLIP Score (Quality) : {avg_clip:.4f}")
        print(f"  Mean IoU   (Control) : {avg_iou:.4f}")
        print(f"  Accuracy   (Success) : {acc*100:.1f}%")
    
    print_metric("Baseline (Inpainting)", results["baseline"])
    print("-" * 20)
    print_metric("Ours (Attention Control)", results["ours"])
    print("="*45 + "\n")

if __name__ == "__main__":
    for task in tasks:
        evaluate_task(task)