
import json

# 读取 val_1.json
with open('ActivityNet-Captions/data/val_1.json', 'r') as f:
    data = json.load(f)

# 提取视频ID（键名就是 "v_" + YouTube视频ID）
video_ids = list(data.keys())

# 写入文件
with open('val1_video_ids.txt', 'w') as f:
    for vid in video_ids:
        # 去掉开头的 "v_" 前缀，得到 YouTube ID
        youtube_id = vid.replace('v_', '')
        f.write(youtube_id + '\n')

print(f"验证集共 {len(video_ids)} 个视频")
