import os
# 1. 强制使用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

print("=== 开始下载模型 (使用截图中的版本) ===")
print("正在从 hf-mirror.com 下载...")

try:
    snapshot_download(
        repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5", # 换成你截图里这个官方镜像版
        local_dir="sd-v1-5",                                   # 下载到当前文件夹下的 sd-v1-5
        ignore_patterns=["*.ckpt", "*.h5", "*.safetensors"],   # 排除超大文件，只下必须的
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("✅ 模型下载成功！请继续训练！")
except Exception as e:
    print(f"❌ 下载失败: {e}")