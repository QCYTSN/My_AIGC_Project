import os
import subprocess

# === 配置 ===
model_id = "google/owlvit-base-patch32"
# 镜像站的基础 URL (绕过官网)
base_url = f"https://hf-mirror.com/{model_id}/resolve/main"
target_dir = "./models/owlvit-base-patch32"

# OWL-ViT 运行必须的 6 个核心文件
files = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "pytorch_model.bin"  # 这是最重的主文件 (约600MB)
]

# === 创建文件夹 ===
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

print(f">>> 🚀 开始暴力下载 OWL-ViT (目标: {target_dir})...")

# === 循环下载 ===
for file in files:
    url = f"{base_url}/{file}"
    save_path = os.path.join(target_dir, file)
    
    print(f"\n⬇️ 正在下载: {file} ...")
    
    # 使用 wget 命令，-c 支持断点续传，-O 指定输出文件名
    # --no-check-certificate 防止 SSL 报错
    cmd = f"wget -c --no-check-certificate -O {save_path} {url}"
    
    # 执行命令
    result = os.system(cmd)
    
    if result != 0:
        print(f"❌ 下载失败: {file}")
        # 如果 pytorch_model.bin 失败，尝试 model.safetensors (有些模型用这个新格式)
        if file == "pytorch_model.bin":
            print("🔄 尝试下载 safetensors 格式...")
            alt_file = "model.safetensors"
            url = f"{base_url}/{alt_file}"
            save_path = os.path.join(target_dir, alt_file)
            cmd = f"wget -c --no-check-certificate -O {save_path} {url}"
            os.system(cmd)
    else:
        print(f"✅ 完成: {file}")

print("\n>>> 🎉 所有文件下载尝试结束！请检查上方是否有报错。")
print("👉 如果看起来都 OK，请直接运行 inference_automask.py")