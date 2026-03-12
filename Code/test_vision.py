import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import cv2
import warnings
warnings.filterwarnings("ignore")

# 1. 绝对的本地路径，不再受网络鸟气！
model_path = "/mnt/data/VMR/cache/llava-1.5-7b-hf"

print(f"正在从本地路径 {model_path} 加载大模型...")

# 2. 【核心修复】：使用最新版规范的 4-bit 量化配置！
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

processor = AutoProcessor.from_pretrained(model_path)
model = LlavaForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    quantization_config=quantization_config  # 替换掉原来的 load_in_4bit=True
)

# 3. 抓取视频画面（确保这个视频在你的文件夹里存在哦）
video_path = "/mnt/data/VMR/Charades_Dataset/videos/Charades_v1_480/003WS.mp4"
print(f"\n正在从视频 {video_path} 中抽取第 4 秒的画面...")

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_MSEC, 4000) 
ret, frame = cap.read()
if not ret:
    print("视频读取失败，请检查视频文件是否存在！")
    exit()

image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

# 4. 提问与推理
prompt = "USER: <image>\nDescribe the action of the person in this image in one short, simple sentence.\nASSISTANT:"

print("\n正在让大模型看图并思考...")
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
generate_ids = model.generate(**inputs, max_new_tokens=30)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print("\n================ 大模型的回答 ================")
answer = output.split("ASSISTANT:")[1].strip()
print(f"[第4秒画面描述]：{answer}")
print("=============================================")
