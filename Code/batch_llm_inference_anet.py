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
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com") 

# ================= 2. 路径配置 =================
test_txt_path = "/mnt/data/VMR/ActivityNet_Dataset/annotations/mini_anet_test.txt"
visual_dir = "/mnt/data/VMR/Results/LLaVA_Descriptions_Anet"
audio_dir = "/mnt/data/VMR/Results/Whisper_Transcripts_Anet"

output_file = "/mnt/data/VMR/Results/final_predictions_anet.jsonl"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# ================= 3. 解析测试集查询 =================
queries =[]
with open(test_txt_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
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

print(f"成功加载 {len(queries)} 条查询任务！")

# ================= 4. 读取已完成的任务 (断点续传) =================
processed_keys = set()
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            processed_keys.add(data['vid'] + "_" + data['query'])
print(f"发现已处理 {len(processed_keys)} 条任务，将自动跳过。")

# ================= 5. 大模型提取函数 =================
def extract_json(text):
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    return {"start": 0.0, "end": 0.0}

# ================= 6. 开始批量战斗 =================
# 【细节修正】：ActivityNet 我们是每 2 秒一帧，Prompt 里要告诉 LLM 实情！
system_prompt = """你是一个具备高级时序推理能力的视频定位专家。
你将收到一段视频的【视觉剧本】(每2秒一帧)和【听觉剧本】。
请根据用户的【查询动作】，推理出该动作发生的最准确的连续时间段(start到end)。

【高级推理原则】：
1. 跨模态校验：若听觉剧本无意义（噪音），彻底忽略它，仅凭视觉判断。若听觉有明确解说，请注意“解说通常比画面动作早1-3秒”的异步规律。
2. 动作时序连续性（极其重要）：人类动作是一个连续过程，请结合画面描述的上下文，推断动作开始的前摇和结束的余波，给出一个合理的时间区间（绝不能只输出一个点！）。
3. 缺失推断：如果画面描述中没有直接出现查询词，请利用上下文逻辑推断。
4. 查询去偏（Query Debiasing）：用户的【查询动作】中可能包含拼写错误、语法错误或生僻词。请发挥你的语言理解能力，自动纠正这些错误，捕捉其真实意图。

你必须严格输出一个合法的 JSON，必须包含 "reasoning", "start", "end" 三个字段。
格式要求：
{
  "reasoning": "你的简短推理过程",
  "start": 12.0,
  "end": 18.0
}
注意：视频中的语音解说可能早于或晚于画面动作。请利用听觉语音流（Auditory Stream）来准确理解用户的查询意图，但你最终输出的时间戳，必须严格对齐视觉感知流（Visual Stream）中该动作实际发生的起止时间。不要将纯语音解说的时间计入最终的片段中。
"""

for q in tqdm(queries, desc="LLM推理进度"):
    vid = q['vid']
    query = q['query']
    key = vid + "_" + query
    
    if key in processed_keys:
        continue
        
    visual_path = os.path.join(visual_dir, f"{vid}.json")
    audio_path = os.path.join(audio_dir, f"{vid}.json")
    
    visual_script = "无"
    if os.path.exists(visual_path):
        with open(visual_path, 'r', encoding='utf-8') as f:
            visual_data = json.load(f)
        visual_script = "\n".join([f"[{item['time_sec']}s]: {item['description']}" for item in visual_data])
        
    # 【完美修复】：听觉处理逻辑
    audio_script = "无"
    if os.path.exists(audio_path):
        with open(audio_path, 'r', encoding='utf-8') as f:
            audio_data = json.load(f)
        if len(audio_data) > 0:
            temp_script = "\n".join([f"[{item['start']}s - {item['end']}s]: {item['text']}" for item in audio_data])
            # 拿到真实文本后，再判断是不是全是没用的短词或噪音
            if len(temp_script) < 5 or "music" in temp_script.lower():
                audio_script = "本视频无有效语音信息。"
            else:
                audio_script = temp_script
        else:
            audio_script = "本视频无有效语音信息。"
            
    user_prompt = f"【视觉剧本】:\n{visual_script}\n\n【听觉剧本】:\n{audio_script}\n\n【查询动作】: {query}\n\n请先简短推理，然后输出时间戳的 JSON:"
    
    pred_result = {"start": 0.0, "end": 0.0}
    for attempt in range(3):
        try:
            # 【完美修复】：补全了 messages，彻底解决 400 Bad Request
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2, 
                max_tokens=300   
            )
            res_text = response.choices[0].message.content
            pred_result = extract_json(res_text)
            break 
        except Exception as e:
            # 【加上打印日志】：以后API出错了立刻能看到！
            print(f"\n[API 错误] 视频 {vid} 第 {attempt+1} 次请求失败: {e}")
            time.sleep(2) 
            
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
        
    time.sleep(0.5)

print("\n大功告成！所有查询的时间戳预测完毕！数据保存在:", output_file)