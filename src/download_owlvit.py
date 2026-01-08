import os
import shutil
# 确保你已经 pip install modelscope
from modelscope.hub.snapshot_download import snapshot_download

# === 配置 ===
# 这里使用你搜索到的正确 ID
model_id = "google/owlvit-base-patch32"

# 这是一个临时缓存目录，下载完我们会搬运
temp_cache_dir = "./models_temp"

# 这是我们 Inference 脚本里写死的最终读取目录
final_target_dir = "./models/owlvit-base-patch32"

print(f">>> 🚀 正在从魔搭社区高速下载: {model_id} ...")

try:
    # 1. 下载 (ModelScope 会自动处理断点续传和加速)
    # cache_dir 指定下载到哪里
    download_path = snapshot_download(model_id, cache_dir=temp_cache_dir)
    
    print(f">>> ✅ 下载成功！原始路径: {download_path}")
    
    # 2. 搬运文件 (为了配合我们的推理代码)
    print(f">>> 📦 正在将模型移动到最终目录: {final_target_dir} ...")
    
    # 如果目标目录已经存在（可能是之前 wget 下了一半的空文件夹），先删掉，防止冲突
    if os.path.exists(final_target_dir):
        shutil.rmtree(final_target_dir)
        
    # 把下载好的文件夹复制过去
    shutil.copytree(download_path, final_target_dir)
    
    # 3. 清理临时缓存
    if os.path.exists(temp_cache_dir):
        shutil.rmtree(temp_cache_dir)

    print(f"🎉 完美！模型已就绪: {final_target_dir}")
    print("👉 现在的网络问题彻底解决了，请运行 inference_automask.py")

except Exception as e:
    print(f"❌ 下载出错: {e}")
    print("如果是 'revision' 错误，说明魔搭上这个模型可能没有 main 分支，但这种情况很少见。")