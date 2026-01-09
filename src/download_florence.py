import os
import shutil
# 确保安装了 modelscope: pip install modelscope
from modelscope.hub.snapshot_download import snapshot_download

# === 配置 ===
# 使用 Large-FT 版本 (效果最强)
model_id = "AI-ModelScope/Florence-2-large-ft"
# 最终保存位置
target_dir = "./models/Florence-2-large-ft"

print(f">>> 🚀 正在从魔搭社区高速下载 {model_id} ...")

try:
    # 1. 下载到临时目录 (ModelScope 默认行为)
    # cache_dir 指定临时缓存位置
    temp_path = snapshot_download(model_id, cache_dir="./models_temp")
    
    print(f">>> ✅ 下载完成，原始路径: {temp_path}")
    print(f">>> 📦 正在搬运到: {target_dir} ...")

    # 2. 搬运文件 (为了目录结构整洁)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir) # 清理旧的
    
    shutil.copytree(temp_path, target_dir)
    
    # 3. 清理缓存
    if os.path.exists("./models_temp"):
        shutil.rmtree("./models_temp")

    print(f"🎉 完美！模型已就绪: {target_dir}")

except Exception as e:
    print(f"❌ 下载失败: {e}")
    print("可能是 ModelScope ID 变了，请尝试搜索 'Florence-2-large-ft'")