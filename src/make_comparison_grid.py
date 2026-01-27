import os
from PIL import Image, ImageDraw, ImageFont

# 配置路径
baseline_root = "results/baseline_eval_v5"
ours_root = "results/final_comparison"
tasks = ["cat_hat", "dog_scarf"] # 我们重点关注这两组

def make_grid():
    for task in tasks:
        print(f"🖼️ 正在拼图: {task}...")
        
        # 获取文件名列表
        files = sorted(os.listdir(f"{ours_root}/{task}"))
        
        # 创建一个大图 (假设我们拼前 5 张作为展示)
        # 布局: 上排 Baseline, 下排 Ours
        num_show = 5
        w, h = 512, 512
        grid_img = Image.new('RGB', (w * num_show, h * 2 + 100), (255, 255, 255))
        
        # 字体 (可选)
        # font = ImageFont.truetype("arial.ttf", 40) 
        
        for i in range(min(num_show, len(files))):
            f_name = files[i]
            
            # 读取 Baseline (Inpainting)
            # 注意 Baseline 的图在 images 子目录里
            path_base = f"{baseline_root}/{task}/images/{f_name}"
            # 读取 Ours (Attention Control)
            path_ours = f"{ours_root}/{task}/{f_name}"
            
            if os.path.exists(path_base) and os.path.exists(path_ours):
                img_b = Image.open(path_base).resize((w, h))
                img_o = Image.open(path_ours).resize((w, h))
                
                # 贴图
                grid_img.paste(img_b, (i * w, 50))       # 上排
                grid_img.paste(img_o, (i * w, 50 + h))   # 下排
                
        # 保存网格
        grid_img.save(f"results/comparison_grid_{task}.jpg")
        print(f"✅ 保存对比图: results/comparison_grid_{task}.jpg")

if __name__ == "__main__":
    make_grid()