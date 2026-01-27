import torch
import os
import json
import cv2
import numpy as np
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# === 配置 ===
device = "cuda" if torch.cuda.is_available() else "cpu"
# 指向你的本地模型
owl_model_id = "./models/owlvit-base-patch32"

# 随便找一张生成的图来测试 (Baseline 或 Ours 都可以)
# 假设我们用 cat_hat 里的第一张图
task_name = "cat_hat"
baseline_dir = "results/baseline_eval_v5"
json_files = sorted([f for f in os.listdir(f"{baseline_dir}/{task_name}/json") if f.endswith(".json")])
target_file = json_files[0] # 取第一张

# === 加载模型 ===
print(f">>> 正在加载 OwlViT 模型: {owl_model_id}")
processor = OwlViTProcessor.from_pretrained(owl_model_id)
model = OwlViTForObjectDetection.from_pretrained(owl_model_id).to(device)

# === 读取图片和数据 ===
with open(f"{baseline_dir}/{task_name}/json/{target_file}", 'r') as f:
    meta = json.load(f)

target_bbox = meta['target_bbox'] # GT 框 [x1, y1, x2, y2]
img_name = target_file.replace(".json", ".png")
img_path = f"{baseline_dir}/{task_name}/images/{img_name}"

print(f">>> 正在分析图片: {img_path}")
image = Image.open(img_path).convert("RGB")
W, H = image.size

# === 关键调试：尝试不同的 Prompt ===
# OwlViT 对 Prompt 很敏感，我们多试几个
queries = [
    "red hat",           # 简单词
    "a photo of a hat",  # 完整句
    "hat",               # 单名词
    "beanie"             # 具体名词
]

print(f"\n>>> 🔍 OwlViT 深度扫描开始...")
print(f"    图片尺寸: {W}x{H}")
print(f"    目标区域 (GT): {target_bbox}")

for query in queries:
    inputs = processor(text=[[query]], images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取原始结果，不设阈值
    target_sizes = torch.Tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.0)[0]
    
    scores = results["scores"]
    boxes = results["boxes"]
    
    # 找到最高分的框
    if len(scores) > 0:
        max_idx = scores.argmax()
        max_score = scores[max_idx].item()
        max_box = boxes[max_idx].cpu().numpy().tolist()
        
        print(f"\n📝 Query: '{query}'")
        print(f"   -> 最高置信度: {max_score:.4f}")
        print(f"   -> 预测框位置: {[int(x) for x in max_box]}")
        
        # 计算一下 IoU 看看
        def get_iou(boxA, boxB):
            xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
            xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
            boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
            return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

        iou = get_iou(target_bbox, max_box)
        print(f"   -> 当前 IoU  : {iou:.4f}")
        
        # 画图保存看看
        debug_img = cv2.imread(img_path)
        # 画 GT (绿色)
        cv2.rectangle(debug_img, (int(target_bbox[0]), int(target_bbox[1])), (int(target_bbox[2]), int(target_bbox[3])), (0, 255, 0), 2)
        cv2.putText(debug_img, "GT", (int(target_bbox[0]), int(target_bbox[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # 画 Pred (蓝色)
        cv2.rectangle(debug_img, (int(max_box[0]), int(max_box[1])), (int(max_box[2]), int(max_box[3])), (255, 0, 0), 2)
        cv2.putText(debug_img, f"Pred {max_score:.2f}", (int(max_box[0]), int(max_box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        save_name = f"results/debug_owl_{query.replace(' ', '_')}.jpg"
        cv2.imwrite(save_name, debug_img)
        print(f"   -> 调试图已保存: {save_name}")

print("\n>>> 分析结束。请查看上面的最高置信度。")
print("    如果最高分都低于 0.1，说明之前的 threshold=0.1 设太高了。")