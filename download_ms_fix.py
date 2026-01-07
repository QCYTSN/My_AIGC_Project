from modelscope.hub.snapshot_download import snapshot_download
import os
import shutil

print("=== 🚀 ModelScope 精简下载模式 (只下核心文件) ===")

# 关键修改：设置过滤列表，排除所有巨大的冗余文件
# 这样下载量会从 25GB 骤降到 5GB 左右
ignore_list = [
    '*.ckpt',           # 排除旧版权重 (每个4GB+)
    '*.safetensors',    # 排除单文件权重 (每个4GB+)
    '*.h5',             # 排除 TensorFlow 权重
    '*.msgpack',        # 排除 Flax 权重
    '*.onnx',           # 排除 ONNX 权重
    '*.png',            # 排除示例图片
    'feature_extractor/*', # 排除这个非必要的文件夹(可选)
]

try:
    # 1. 下载
    model_dir = snapshot_download(
        'AI-ModelScope/stable-diffusion-v1-5', 
        cache_dir='.', 
        revision='v1.0.8',
        ignore_file_pattern=ignore_list  # <--- 加上这一行过滤
    )
    print(f"✅ 下载成功！原始路径: {model_dir}")

    # 2. 重命名整理
    target_dir = "./sd-v1-5"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    
    # 移动/重命名
    # 注意：modelscope 下载路径可能包含 repo 名字，我们做个判断
    if os.path.exists(model_dir):
        os.rename(model_dir, target_dir)
        print("=== 🎉 恭喜！模型已就绪，文件夹名称: sd-v1-5 ===")
    else:
        print("⚠️ 警告：下载目录结构可能有变，请手动检查 AI-ModelScope 文件夹")

except Exception as e:
    print(f"❌ 下载失败: {e}")
