import os
import json
import time
import re
from openai import OpenAI
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ================= 1. 配置 API =================
API_KEY = "sk-c44aa6aa963b454fb6cde9219aa2825e"
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com") # 如果用其他模型，记得改base_url

# ================= 2. 路径配置 =================
test_txt_path = "/mnt/data/VMR/Charades_Dataset/annotations/charades_sta_test.txt"
visual_dir = "/mnt/data/VMR/Results/LLaVA_Descriptions_Charades"
audio_dir = "/mnt/data/VMR/Results/Whisper_Transcripts_Charades"

# 结果保存路径（使用 .jsonl 格式，跑一条存一条，绝对不怕断电！）
output_file = "/mnt/data/VMR/Results/final_predictions.jsonl"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# ================= 3. 解析测试集查询 =================
queries =[]
with open(test_txt_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        # Charades标注格式: "视频ID 开始时间 结束时间##查询句子"
        parts = line.split('##')
        if len(parts) != 2: continue
        
        vid_info, query = parts[0], parts[1]
        vid_info_parts = vid_info.split()
        vid = vid_info_parts[0]
        gt_start = float(vid_info_parts[1])
        gt_end = float(vid_info_parts[2])
        
        queries.append({
            "vid": vid, 
            "query": query, 
            "gt_start": gt_start, 
            "gt_end": gt_end
        })

print(f"🎯 成功加载 {len(queries)} 条查询任务！")

# ================= 4. 读取已完成的任务 (断点续传) =================
processed_keys = set()
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 用 vid + query 作为唯一标识符
            processed_keys.add(data['vid'] + "_" + data['query'])
print(f"✅ 发现已处理 {len(processed_keys)} 条任务，将自动跳过。")

# ================= 5. 大模型提取函数 (带正则和容错) =================
def extract_json(text):
    # 大模型有时候会自作聪明加 ```json ... ```，用正则强行挖出字典
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    return {"start": 0.0, "end": 0.0}

# ================= 6. 开始批量战斗 =================
system_prompt = """你是一个顶级的视频片段检索专家。
你将收到视频的【视觉剧本】和【听觉剧本】。请根据【查询动作】，推理最准确的起止时间。

【核心原则】：
1. 噪声拒绝：如果听觉剧本中的对话与查询动作毫不相干（如背景杂音、无关聊天），请果断忽略听觉，完全信任视觉！
2. 动作连续性：人类的动作通常持续 3 到 8 秒。不要只输出一个点，要结合常识输出一个合理的时间段！
3. 音画异步：如果听觉是教学解说，它通常比画面早发生 1-3 秒，请以画面实际发生为准。

【示范学习 (Few-Shot Example)】
用户输入：
【视觉剧本】: [10s]: 人走向门口; [12s]: 人握住门把手; [14s]: 门开了; [18s]: 人出去了
【听觉剧本】: [08s-11s]: 电视机里播放着广告; [13s-15s]: 咔嚓一声
【查询动作】: a person opens a door
你的内心思考过程：听觉剧本的电视声是噪音，忽略。"咔嚓"声和视觉12s-14s匹配。开门动作一般持续4-6秒。
最终输出：
{"start": 10.0, "end": 16.0}

请严格遵循上述逻辑，对本次查询进行推理，仅输出最终的 JSON，不要输出你的思考过程！
"""


for q in tqdm(queries, desc="LLM推理进度"):
    vid = q['vid']
    query = q['query']
    key = vid + "_" + query
    
    if key in processed_keys:
        continue
        
    # 读取双流剧本
    visual_path = os.path.join(visual_dir, f"{vid}.json")
    audio_path = os.path.join(audio_dir, f"{vid}.json")
    
    # 视觉容错处理
    visual_script = "无"
    if os.path.exists(visual_path):
        with open(visual_path, 'r', encoding='utf-8') as f:
            visual_data = json.load(f)
        visual_script = "\n".join([f"[{item['time_sec']}s]: {item['description']}" for item in visual_data])
        
    # 听觉容错处理
    audio_script = "无"
    if os.path.exists(audio_path):
        with open(audio_path, 'r', encoding='utf-8') as f:
            audio_data = json.load(f)
        if len(audio_data) > 0:
            audio_script = "\n".join([f"[{item['start']}s - {item['end']}s]: {item['text']}" for item in audio_data])
            
    user_prompt = f"【视觉画面剧本】:\n{visual_script}\n\n【听觉语音剧本】:\n{audio_script}\n\n【查询动作】: {query}\n\n请推理时间戳，仅返回 JSON:"
    
    # 失败重试机制（最多重试3次）
    pred_result = {"start": 0.0, "end": 0.0}
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat", # 请替换为你使用的模型名字
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            res_text = response.choices[0].message.content
            pred_result = extract_json(res_text)
            break # 成功就跳出重试循环
        except Exception as e:
            time.sleep(2) # 遇到API限流或报错，冷静2秒再试
            
    # 把结果拼装起来，写入 .jsonl 文件 (追加模式 'a')
    final_output = {
        "vid": vid,
        "query": query,
        "gt_start": q["gt_start"],
        "gt_end": q["gt_end"],
        "pred_start": pred_result.get("start", 0.0),
        "pred_end": pred_result.get("end", 0.0)
    }
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(final_output, ensure_ascii=False) + "\n")
        
    # 防止请求过快被API封杀，睡个短暂的0.5秒
    time.sleep(0.5)

print("\n🎉 大功告成！所有查询的时间戳预测完毕！数据保存在:", output_file)
