import os
import subprocess

# === 配置 ===
target_dir = "./models"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

print(f">>> 🚀 开始暴力下载 GroundingDINO 和 SAM (目标: {target_dir})...")

# === 定义下载清单 (文件名: 镜像站直链) ===
files_to_download = {
    # 1. GroundingDINO 权重 (来自 ShilongLiu 的官方搬运)
    "groundingdino_swinb_cogcoor.pth": "https://hf-mirror.com/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth",
    
    # 2. GroundingDINO 配置文件
    "GroundingDINO_SwinB.cfg.py": "https://hf-mirror.com/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
    
    # 3. SAM 权重 (来自 ybelkada 的搬运，文件就是官方的 vit_h)
    "sam_vit_h_4b8939.pth": "https://hf-mirror.com/ybelkada/segment-anything/resolve/main/sam_vit_h_4b8939.pth"
}

# === 循环下载 ===
for filename, url in files_to_download.items():
    save_path = os.path.join(target_dir, filename)
    
    # 如果文件已经存在且很大(说明下完了)，跳过
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        print(f"✅ {filename} 已存在，跳过。")
        continue

    print(f"\n⬇️ 正在下载: {filename} ...")
    print(f"🔗 来源: {url}")
    
    # 使用 wget 命令
    # -c: 断点续传
    # --no-check-certificate: 防止 SSL 报错
    # -O: 指定保存路径
    cmd = f"wget -c --no-check-certificate -O {save_path} {url}"
    
    result = os.system(cmd)
    
    if result != 0:
        print(f"❌ 下载失败: {filename}")
    else:
        print(f"✅ 下载完成: {filename}")

print("\n>>> 🎉 所有任务结束！请检查 ./models 目录。")