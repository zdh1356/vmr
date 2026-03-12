import os
import json
import time
import re
from openai import OpenAI
from tqdm import tqdm
from audio_confidence import filter_audio_segments, calculate_audio_confidence
from visual_confidence import calculate_visual_confidence
import warnings
warnings.filterwarnings("ignore")

# ================= 1. 配置 API =================
API_KEY = "sk-c44aa6aa963b454fb6cde9219aa2825e"
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# ================= 2. 路径配置 =================
# 根据数据集切换（这里以 Charades-STA 为例）
DATASET = "charades"  # 切换为 "activitynet" 时修改路径

if DATASET == "charades":
    test_txt_path = "/mnt/data/VMR/Charades_Dataset/annotations/charades_sta_test.txt"
    visual_dir = "/mnt/data/VMR/Results/LLaVA_Descriptions_Charades"
    audio_dir = "/mnt/data/VMR/Results/Whisper_Transcripts_Charades"
    output_file = "/mnt/data/VMR/Results/final_predictions_charades_v2.jsonl"
elif DATASET == "activitynet":
    test_txt_path = "/mnt/data/VMR/ActivityNet/annotations/val.json"
    visual_dir = "/mnt/data/VMR/Results/LLaVA_Descriptions_ActivityNet"
    audio_dir = "/mnt/data/VMR/Results/Whisper_Transcripts_ActivityNet"
    output_file = "/mnt/data/VMR/Results/final_predictions_activitynet_v2.jsonl"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# ================= 3. 解析测试集查询 =================
queries = []

if DATASET == "charades":
    with open(test_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('##')
            if len(parts) != 2:
                continue
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

elif DATASET == "activitynet":
    with open(test_txt_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    for vid, info in val_data.items():
        # ActivityNet Captions 格式
        timestamps = info.get("timestamps", [])
        sentences = info.get("sentences", [])
        duration = info.get("duration", 0)
        for ts, sent in zip(timestamps, sentences):
            queries.append({
                "vid": vid,
                "query": sent,
                "gt_start": ts[0],
                "gt_end": ts[1],
                "duration": duration
            })

print(f"🎯 成功加载 {len(queries)} 条查询任务！")

# ================= 4. 断点续传 =================
processed_keys = set()
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            processed_keys.add(data['vid'] + "_" + data['query'])
print(f"✅ 已处理 {len(processed_keys)} 条，将跳过。")

# ================= 5. JSON 提取 =================
def extract_json(text):
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            # 验证必须有 start 和 end
            if "start" in result and "end" in result:
                return result
        except:
            pass
    return {"start": 0.0, "end": 0.0}

# ================= 6. 动态 Prompt 生成 =================
def build_system_prompt(audio_confidence, visual_confidence):
    """
    根据双流置信度动态生成 system prompt
    这是整个方法的核心创新点！
    """
    base_prompt = """你是一个顶级的视频片段检索专家。
你将收到一段视频的【视觉画面描述】和【听觉语音转录】。
请根据用户输入的【查询动作】，推理出该动作发生的最准确的起止时间。"""

    # 动态信任度指令
    if audio_confidence < 0.3 and visual_confidence >= 0.5:
        trust_instruction = """
【重要提示】：听觉转录质量极低（可能是静音视频的Whisper幻觉），请完全忽略听觉信息，仅依赖视觉描述进行推理。
你需要仔细分析视觉描述中每个时间点的动作，找到与查询最匹配的时间段。"""
    elif visual_confidence < 0.3 and audio_confidence >= 0.5:
        trust_instruction = """
【重要提示】：视觉描述质量较低（可能是画面模糊或遮挡），请优先依赖听觉转录进行推理。
你需要通过语音内容中的关键词和时间线来定位动作发生的时间段。"""
    elif audio_confidence < 0.3 and visual_confidence < 0.3:
        trust_instruction = """
【重要提示】：视觉和听觉数据质量都较低，请基于有限信息做出最佳猜测。
尝试从任何可用的线索中推断动作的大致时间范围。"""
    else:
        trust_instruction = """
【融合策略】：视觉和听觉信息都较为可靠，请综合两者进行推理。
- 如果听觉中提到的动作时间与视觉观察到的时间有差异，以视觉为主。
- 利用听觉提供的上下文信息来辅助判断动作边界。"""

    format_instruction = """
【推理步骤】：
1. 先通读视觉描述，标记与查询动作相关的时间点
2. 再检查听觉转录中是否有补充信息
3. 综合判断动作的起始和结束时间
4. 输出结果

【输出格式】：只输出一个 JSON，格式为 {"start": 起始秒, "end": 结束秒}
绝对不要输出任何其他文字！"""

    return base_prompt + trust_instruction + format_instruction


def build_user_prompt(visual_script, audio_script, query, audio_conf, visual_conf):
    """构建用户 prompt，附带置信度标注"""
    parts = []
    
    parts.append(f"【视觉画面描述】(置信度: {visual_conf:.1%}):")
    parts.append(visual_script if visual_script != "无" else "（无有效视觉描述）")
    parts.append("")
    
    parts.append(f"【听觉语音转录】(置信度: {audio_conf:.1%}):")
    parts.append(audio_script if audio_script != "无" else "（无有效语音内容，此为静音视频）")
    parts.append("")
    
    parts.append(f"【查询动作】: {query}")
    parts.append("")
    parts.append("请推理该动作的时间戳，仅返回JSON:")
    
    return "\n".join(parts)

# ================= 7. 主推理循环 =================
for q in tqdm(queries, desc="LLM推理进度"):
    vid = q['vid']
    query = q['query']
    key = vid + "_" + query

    if key in processed_keys:
        continue

    # 读取视觉描述
    visual_path = os.path.join(visual_dir, f"{vid}.json")
    visual_data = []
    visual_script = "无"
    if os.path.exists(visual_path):
        with open(visual_path, 'r', encoding='utf-8') as f:
            visual_data = json.load(f)
        if visual_data:
            visual_script = "\n".join(
                [f"[{item['time_sec']}s]: {item['description']}" for item in visual_data]
            )

    # 读取听觉转录
    audio_path = os.path.join(audio_dir, f"{vid}.json")
    raw_audio_data = []
    audio_script = "无"
    if os.path.exists(audio_path):
        with open(audio_path, 'r', encoding='utf-8') as f:
            raw_audio_data = json.load(f)

    # 🔑 核心改进：噪声过滤 + 置信度计算
    filtered_audio, audio_conf = filter_audio_segments(raw_audio_data, min_confidence=0.3)
    visual_conf, visual_reason = calculate_visual_confidence(visual_data)

    if filtered_audio:
        audio_script = "\n".join(
            [f"[{item['start']}s - {item['end']}s]: {item['text']}" for item in filtered_audio]
        )
    else:
        audio_script = "无"

    # 动态构建 prompt
    system_prompt = build_system_prompt(audio_conf, visual_conf)
    user_prompt = build_user_prompt(visual_script, audio_script, query, audio_conf, visual_conf)

    # LLM 调用（带重试）
    pred_result = {"start": 0.0, "end": 0.0}
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.05,  # 降低温度，更确定性
                max_tokens=80      # 增加token上限
            )
            res_text = response.choices[0].message.content
            pred_result = extract_json(res_text)
            
            # 验证结果合理性
            pred_s = pred_result.get("start", 0.0)
            pred_e = pred_result.get("end", 0.0)
            if pred_s is None:
                pred_s = 0.0
            if pred_e is None:
                pred_e = 0.0
            pred_result["start"] = float(pred_s)
            pred_result["end"] = float(pred_e)
            
            break
        except Exception as e:
            time.sleep(3)

    # 保存结果
    final_output = {
        "vid": vid,
        "query": query,
        "gt_start": q["gt_start"],
        "gt_end": q["gt_end"],
        "pred_start": pred_result.get("start", 0.0),
        "pred_end": pred_result.get("end", 0.0),
        "audio_confidence": audio_conf,
        "visual_confidence": visual_conf
    }

    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(final_output, ensure_ascii=False) + "\n")

    time.sleep(0.5)

print(f"\n🎉 推理完毕！结果保存在: {output_file}")
