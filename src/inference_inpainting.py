import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
import os

# 配置路径
base_model = "./sd-v1-5"
lora_dog = "./outputs/lora_dog"
lora_glass = "./outputs/lora_sunglasses"
save_dir = "results/final_tryon"
os.makedirs(save_dir, exist_ok=True)

# === 🐶 第一步：生成底图（狗） ===
print(">>> 🐶 Step 1: Generating Dog Base Image...")
pipe_t2i = StableDiffusionPipeline.from_pretrained(
    base_model, torch_dtype=torch.float16, safety_checker=None
).to("cuda")

pipe_t2i.load_lora_weights(lora_dog, adapter_name="dog")
pipe_t2i.set_adapters(["dog"], adapter_weights=[1.0])

# 关键 Prompt：Front view (正视图) 确保狗脸正对镜头，方便我们对齐 Mask
prompt_dog = "a photo of a sks dog looking at the camera, front view, in a garden, high quality"
# 固定种子，方便复现（如果这张狗不好看，可以改 seed）
generator = torch.Generator("cuda").manual_seed(42) 

dog_image = pipe_t2i(prompt_dog, num_inference_steps=30, generator=generator).images[0]
dog_image_path = f"{save_dir}/step1_dog.png"
dog_image.save(dog_image_path)
print(f"✅ Dog image saved to {dog_image_path}")

# 清理显存
del pipe_t2i
torch.cuda.empty_cache()

# === 🖌️ 第二步：制作 Mask (针对正脸狗) ===
print(">>> 🖌️ Step 2: Creating Mask...")
# 既然我们用了 seed=42 的正脸狗，我们可以预估眼睛的大致位置
# 对于 512x512 的正脸图，眼睛通常在垂直方向的中间偏上
mask = Image.new("L", (512, 512), 0)
draw = ImageDraw.Draw(mask)

# 画一个覆盖双眼的宽矩形
# [左, 上, 右, 下]
# 你可以打开 step1_dog.png 确认一下位置，如果歪了可以微调这里
draw.rectangle([140, 150, 372, 230], fill=255) 

mask_path = f"{save_dir}/step2_mask.png"
mask.save(mask_path)

# === 🕶️ 第三步：佩戴墨镜 (Inpainting) ===
print(">>> 🕶️ Step 3: Inpainting Sunglasses...")

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    base_model, torch_dtype=torch.float16, safety_checker=None
).to("cuda")

pipe_inpaint.load_lora_weights(lora_glass, adapter_name="sunglasses")
pipe_inpaint.set_adapters(["sunglasses"], adapter_weights=[1.0])

prompt_glass = "a photo of a sks sunglasses on a dog face, black frame, transparent glass, realistic"
negative_prompt = "cartoon, painting, low quality, bad anatomy, eyes closed"

final_image = pipe_inpaint(
    prompt=prompt_glass,
    image=dog_image,
    mask_image=mask,
    strength=0.95,  # 强度高一点，确保完全画成墨镜
    num_inference_steps=40,
    guidance_scale=8.0
).images[0]

final_path = f"{save_dir}/final_result.png"
final_image.save(final_path)
print(f"🎉 任务完成！结果保存在 {final_path}")