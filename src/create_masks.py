from PIL import Image, ImageDraw, ImageFilter
import os

# 配置路径
output_dir = "./data/masks"
os.makedirs(output_dir, exist_ok=True)

# 定义坐标 (方便复用)
dog_box = [100, 100, 412, 500]      # 狗的大框
glass_box = [180, 150, 332, 200]    # 墨镜的小框

# === 1. 画墨镜 Mask (保持不变) ===
mask_glass = Image.new("L", (512, 512), 0)
draw_glass = ImageDraw.Draw(mask_glass)
draw_glass.rectangle(glass_box, fill=255)

# ✨ 优化点 A: 给墨镜边缘一点羽化，让融合不那么生硬
mask_glass = mask_glass.filter(ImageFilter.GaussianBlur(radius=3))
mask_glass.save(f"{output_dir}/mask_sunglasses.png")
print("✅ Sunglasses mask saved (with blur).")

# === 2. 画狗 Mask (关键：挖孔！) ===
mask_dog = Image.new("L", (512, 512), 0)
draw_dog = ImageDraw.Draw(mask_dog)

# 第一步：画大框 (白色)
draw_dog.rectangle(dog_box, fill=255)

# ✨ 优化点 B: 挖孔！把墨镜区域涂黑 (0)
# 这样模型就知道：狗的纹理不要生成在墨镜区域
draw_dog.rectangle(glass_box, fill=0)

# 同样给一点羽化
mask_dog = mask_dog.filter(ImageFilter.GaussianBlur(radius=3))
mask_dog.save(f"{output_dir}/mask_dog.png")
print("✅ Dog mask saved (with hole created).")