import os
import json

# 1. 路径设置
video_root_dir = "/mnt/data/VMR/ActivityNet_Dataset/Videos/v1-3/train_val" # 解压出来的视频通常在这个子目录
val_json_path = "/mnt/data/VMR/ActivityNet_Dataset/annotations/val_1.json"
mini_test_txt = "/mnt/data/VMR/ActivityNet_Dataset/annotations/mini_anet_test.txt"

# 2. 读取解压出来的物理视频文件列表
extracted_vids = set()
for root, dirs, files in os.walk(video_root_dir):
    for f in files:
        if f.endswith(('.mp4', '.mkv', '.webm')):
            # 统一去掉后缀和前缀 v_
            vid_name = f.rsplit('.', 1)[0].replace('v_', '')
            extracted_vids.add(vid_name)

print(f"📦 强行解压出的有效物理视频总数：{len(extracted_vids)} 个")

# 3. 加载官方的验证集 JSON
with open(val_json_path, 'r') as f:
    data = json.load(f)

matched_vids = set()
matched_queries = 0

# 4. 开始碰撞匹配！
with open(mini_test_txt, 'w', encoding='utf-8') as out:
    for vid, info in data.items():
        vid_clean = vid.replace('v_', '')
        
        # 如果这个验证集视频刚好在我们解压出来的这批货里，就收编它！
        if vid_clean in extracted_vids:
            matched_vids.add(vid_clean)
            for i in range(len(info['sentences'])):
                start = info['timestamps'][i][0]
                end = info['timestamps'][i][1]
                sentence = info['sentences'][i].strip()
                out.write(f"{vid_clean} {start} {end}##{sentence}\n")
                matched_queries += 1

print(f"🎯 成功构建 Mini-Testset！")
print(f"✅ 捞到了 {len(matched_vids)} 个视频，共包含 {matched_queries} 条查询任务。")
print(f"📄 测试标注文件已保存在: {mini_test_txt}")

# 5. 极致磁盘瘦身：把没被选中的辣鸡训练集视频全删了！
print("\n🧹 正在删除没用的训练集视频，为您释放磁盘空间...")
deleted_count = 0
for root, dirs, files in os.walk(video_root_dir):
    for f in files:
        if f.endswith(('.mp4', '.mkv', '.webm')):
            vid_clean = f.rsplit('.', 1)[0].replace('v_', '')
            if vid_clean not in matched_vids:
                os.remove(os.path.join(root, f))
                deleted_count += 1
print(f"💥 清理完毕！删除了 {deleted_count} 个无用视频，磁盘安全了！")
