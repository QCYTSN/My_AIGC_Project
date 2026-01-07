import os
import sys
# 强制使用国内最快镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

print("=== 🚀 开始下载 Stable Diffusion v1-5 (精简版) ===")

# 过滤规则：
# 1. 排除 safetensors/ckpt/h5 (我们用标准的 pytorch_model.bin)
# 2. 排除 fp16/non_ema (初学者训练不需要这些备份，能省 15GB 空间)
# 3. 排除 tensorflow/flax 权重
ignore_list = [
    "*.ckpt", 
    "*.h5", 
    "*.safetensors",
    "*.fp16.bin", 
    "*.non_ema.bin",
    "*.msgpack",
    "*.tflite"
]

try:
    snapshot_download(
        repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5", 
        local_dir="sd-v1-5", 
        ignore_patterns=ignore_list,
        local_dir_use_symlinks=False, # 确保下载的是真实文件
        resume_download=True          # 开启断点续传
    )
    print("SUCCESS_DONE") # 成功的特殊标记
except Exception as e:
    print(f"DOWNLOAD_ERROR: {e}")
    sys.exit(1) # 报错退出，让外部脚本捕获
