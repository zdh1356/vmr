import os
import json
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import warnings
warnings.filterwarnings("ignore")

# ================= 1. 基础路径配置 =================
test_txt_path = "/mnt/data/VMR/ActivityNet_Dataset/annotations/mini_anet_test.txt"
video_dir = "/mnt/data/VMR/ActivityNet_Dataset/Videos/v1-3/train_val"
output_dir = "/mnt/data/VMR/Results/LLaVA_Descriptions_Anet"
model_path = "/mnt/data/VMR/cache/llava-1.5-7b-hf"

os.makedirs(output_dir, exist_ok=True)

# 🌟 针对长视频的策略调整：每 4 秒抽一帧
SAMPLE_INTERVAL = 2

# ================= 2. 筛选我们要的那 804 个视频 =================
test_video_ids = set()
with open(test_txt_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            test_video_ids.add(line.split(' ')[0])

video_files =[]
for root, dirs, files in os.walk(video_dir):
    for f in files:
        if f.endswith(('.mp4', '.mkv', '.webm')):
            vid_name = f.rsplit('.', 1)[0].replace('v_', '')
            if vid_name in test_video_ids:
                video_files.append(os.path.join(root, f))

print(f"🎯 共匹配到 {len(video_files)} 个教学视频，准备开始视觉扫描！")

# ================= 3. 加载 4-bit 视觉大模型 =================
print(f"🧠 正在从本地加载 LLaVA 大模型...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
processor = AutoProcessor.from_pretrained(model_path)
model = LlavaForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    quantization_config=quantization_config
)
prompt_text = "USER: <image>\nDescribe the action of the person in this image in one short, simple sentence.\nASSISTANT:"

# ================= 4. 批量处理流水线 =================
for video_path in tqdm(video_files, desc="Anet视觉扫描进度"):
    # 统一命名规范，去掉 v_ 前缀，保证和 Whisper 提出来的 json 名字一模一样！
    vid_name = os.path.basename(video_path).rsplit('.', 1)[0].replace('v_', '')
    output_path = os.path.join(output_dir, f"{vid_name}.json")
    
    if os.path.exists(output_path):
        continue

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        continue
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0: fps = 30
    duration = frame_count / fps

    video_descriptions =[]
    
    for sec in range(0, int(duration) + 1, SAMPLE_INTERVAL):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret: break
            
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(model.device)
        generate_ids = model.generate(**inputs, max_new_tokens=25)
        output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        answer = output.split("ASSISTANT:")[1].strip()
        
        video_descriptions.append({"time_sec": sec, "description": answer})
        
    cap.release()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(video_descriptions, f, ensure_ascii=False, indent=2)
        
    torch.cuda.empty_cache()

print("\n🎉 ActivityNet 所有视频的【视觉剧本】提取完毕！")
