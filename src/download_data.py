import os
from modelscope.msdatasets import MsDataset

# === 配置区域 ===
concept_name = "pink_sunglasses"
save_dir = f"/root/My_AIGC_Project/data/{concept_name}"

print(f"=== 🚀 [ModelScope最终版] 正在下载 {concept_name} 子集... ===")

try:
    # 1. 精准加载 "pink_sunglasses" 子集
    # subset_name 参数告诉它我们要下载哪个物体
    ds = MsDataset.load('google/dreambooth', subset_name=concept_name, split='train')

    # 2. 创建文件夹
    os.makedirs(save_dir, exist_ok=True)
    
    # 3. 遍历并保存
    count = 0
    for item in ds:
        # 这里的 item 已经是指定子集的数据了，直接拿 'image' 字段
        if 'image' in item:
            item['image'].save(os.path.join(save_dir, f"{count:02d}.jpg"))
            count += 1

    print(f"✅ 成功！已保存 {count} 张粉色墨镜图片到: {save_dir}")

except Exception as e:
    print(f"❌ 发生错误: {e}")
    # 如果出错，打印一下第一条数据长什么样，方便调试
    try:
        print(f"🔍 调试信息 - 数据集Keys: {ds[0].keys()}")
    except:
        pass

except Exception as e:
    print(f"❌ 发生错误: {e}")