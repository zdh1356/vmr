import os
import json
import whisper
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ================= 核心修改：只读取测试集需要的视频 =================
test_txt_path = "/mnt/data/VMR/Charades_Dataset/annotations/charades_sta_test.txt"

# 提取测试集里所有出现过的视频 ID
test_video_ids = set()
with open(test_txt_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Charades-STA标注格式: "视频ID 开始时间 结束时间##查询句子"
        vid = line.split(' ')[0]
        test_video_ids.add(vid)

print(f"🎯 从测试集标注文件中提取到 {len(test_video_ids)} 个唯一的视频目标！")
# =====================================================================

video_dir = "/mnt/data/VMR/Charades_Dataset/videos/Charades_v1_480" 
output_dir = "/mnt/data/VMR/Results/Whisper_Transcripts_Charades"
model_cache_dir = "/mnt/data/VMR/cache"

os.makedirs(output_dir, exist_ok=True)

print("正在加载 Whisper Large 模型...")
model = whisper.load_model("large", download_root=model_cache_dir) 

# 只挑选出在 test_video_ids 里的视频！！
video_files =[f for f in os.listdir(video_dir) if f.endswith('.mp4') and f.replace('.mp4', '') in test_video_ids]
print(f"🚀 实际在文件夹中匹配到需要处理的测试集视频数量：{len(video_files)} 个！")

for video_file in tqdm(video_files, desc="测试集字幕提取"):
    video_path = os.path.join(video_dir, video_file)
    output_filename = video_file.replace('.mp4', '.json')
    output_path = os.path.join(output_dir, output_filename)
    
    if os.path.exists(output_path):
        continue

    try:
        result = model.transcribe(video_path, fp16=False)
        segments = []
        for s in result["segments"]:
            segments.append({
                "start": round(s["start"], 2),
                "end": round(s["end"], 2),
                "text": s["text"].strip()
            })
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"\n[错误] 处理视频 {video_file} 时出错: {e}")

print("\n🎉 测试集全部视频语音提取完毕！可以进入下一阶段了！")
