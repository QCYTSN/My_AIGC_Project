from modelscope.hub.snapshot_download import snapshot_download
import os
import shutil

print("=== 🚀 正在从 ModelScope (阿里云内网) 高速下载... ===")

# 1. 从 ModelScope 下载
# cache_dir='.' 表示下载到当前目录下
model_dir = snapshot_download('AI-ModelScope/stable-diffusion-v1-5', cache_dir='.', revision='v1.0.8')

print(f"✅ 下载完成！原始路径: {model_dir}")

# 2. 整理文件夹名称
# ModelScope 下载后的文件夹名字比较长，我们把它改名为简单的 'sd-v1-5'
target_dir = "./sd-v1-5"
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

# 将下载的文件夹重命名为 sd-v1-5
os.rename(model_dir, target_dir)

print("=== 🎉 恭喜！模型已就绪，文件夹名称: sd-v1-5 ===")
