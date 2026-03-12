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
test_txt_path = "/mnt/data/VMR/Charades_Dataset/annotations/charades_sta_test.txt"
video_dir = "/mnt/data/VMR/Charades_Dataset/videos/Charades_v1_480"
output_dir = "/mnt/data/VMR/Results/LLaVA_Descriptions_Charades"
model_path = "/mnt/data/VMR/cache/llava-1.5-7b-hf"

os.makedirs(output_dir, exist_ok=True)
SAMPLE_INTERVAL = 1  # 每隔 2 秒抽取一帧进行描述（平衡速度和精度）

# ================= 2. 筛选测试集视频 =================
test_video_ids = set()
with open(test_txt_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            vid = line.split(' ')[0]
            test_video_ids.add(vid)

video_files =[f for f in os.listdir(video_dir) if f.endswith('.mp4') and f.replace('.mp4', '') in test_video_ids]
print(f"🎯 共匹配到 {len(video_files)} 个测试集视频，准备开始视觉扫描！")

# ================= 3. 加载 4-bit 视觉大模型 =================
print(f"🧠 正在加载 LLaVA 大模型，请稍候...")
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
for video_file in tqdm(video_files, desc="视频处理进度"):
    video_path = os.path.join(video_dir, video_file)
    output_filename = video_file.replace('.mp4', '.json')
    output_path = os.path.join(output_dir, output_filename)
    
    # 【断点续传】如果已经处理过，直接跳过
    if os.path.exists(output_path):
        continue

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        continue
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0:
        fps = 30 # 保底默认值
    duration = frame_count / fps

    video_descriptions =[]
    
    # 每隔 SAMPLE_INTERVAL 秒截一帧
    for sec in range(0, int(duration) + 1, SAMPLE_INTERVAL):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            break
            
        # OpenCV 转 PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 喂入模型
        inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(model.device)
        generate_ids = model.generate(**inputs, max_new_tokens=25)
        output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        answer = output.split("ASSISTANT:")[1].strip()
        
        video_descriptions.append({
            "time_sec": sec,
            "description": answer
        })
        
    cap.release()
    
    # 写入 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(video_descriptions, f, ensure_ascii=False, indent=2)
        
    # 每跑完一个视频清一下显存碎片，极致保护
    torch.cuda.empty_cache()

print("\n🎉 恭喜！测试集所有视频的【视觉剧本】提取完毕！")
