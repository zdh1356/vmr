import os
import json
import whisper
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 我们的迷你暗杀名单
test_txt_path = "/mnt/data/VMR/ActivityNet_Dataset/annotations/mini_anet_test.txt"
video_dir = "/mnt/data/VMR/ActivityNet_Dataset/Videos/v1-3/train_val" 
output_dir = "/mnt/data/VMR/Results/Whisper_Transcripts_Anet"
model_cache_dir = "/mnt/data/VMR/cache"

os.makedirs(output_dir, exist_ok=True)

test_video_ids = set()
with open(test_txt_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip(): test_video_ids.add(line.split(' ')[0])

print("正在加载 Whisper Large 模型...")
model = whisper.load_model("large", download_root=model_cache_dir) 

video_files =[]
for root, dirs, files in os.walk(video_dir):
    for f in files:
        if f.endswith(('.mp4', '.mkv', '.webm')):
            vid_name = f.rsplit('.', 1)[0].replace('v_', '')
            if vid_name in test_video_ids:
                video_files.append(os.path.join(root, f))

for video_path in tqdm(video_files, desc="Anet字幕提取"):
    vid_name = os.path.basename(video_path).rsplit('.', 1)[0].replace('v_', '')
    output_path = os.path.join(output_dir, f"{vid_name}.json")
    
    if os.path.exists(output_path): continue

    try:
        result = model.transcribe(video_path, fp16=False)
        segments =[{"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()} for s in result["segments"]]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"\n[错误] 处理视频出错: {e}")

print("\n🎉 ActivityNet 语音提取完毕！")
