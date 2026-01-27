import os
import time

# === 🚀 关键配置：设置镜像加速 ===
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

def download_model(repo_id, local_dir):
    print(f"\n>>> 正在下载: {repo_id} ...")
    print(f"    目标路径: {local_dir}")
    
    max_retries = 5
    for i in range(max_retries):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False, # 确保下载的是真实文件
                resume_download=True,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # 忽略不需要的格式，省流量
            )
            print("    ✅ 下载成功！")
            return
        except Exception as e:
            print(f"    ⚠️ 下载失败 (尝试 {i+1}/{max_retries}): {e}")
            time.sleep(2)
            
    print("    ❌ 最终下载失败，请检查网络。")

if __name__ == "__main__":
    # 1. 下载 CLIP
    download_model(
        repo_id="openai/clip-vit-base-patch32", 
        local_dir="./models/clip-vit-base-patch32"
    )
    
    # 2. 下载 OwlViT
    download_model(
        repo_id="google/owlvit-base-patch32", 
        local_dir="./models/owlvit-base-patch32"
    )