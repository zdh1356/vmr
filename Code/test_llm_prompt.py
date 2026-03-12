from openai import OpenAI
import json
import os

# ================= 1. 配置你的 API =================
# 这里以 DeepSeek 为例，如果你用其他模型，修改 base_url 即可
API_KEY ="sk-c44aa6aa963b454fb6cde9219aa2825e"
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# ================= 2. 读取我们的双流剧本 =================
# 我们拿测试集里的第一个视频来测试
test_vid = "00607"  # 换成你刚才跑出来的、随便一个 JSON 文件的前缀名
query = "A woman is standing in a room and playing a video game." # 假设用户的搜索词

visual_path = f"/mnt/data/VMR/Results/LLaVA_Descriptions_Charades/{test_vid}.json"
audio_path = f"/mnt/data/VMR/Results/Whisper_Transcripts_Charades/{test_vid}.json"

# 读取视觉剧本
with open(visual_path, 'r', encoding='utf-8') as f:
    visual_data = json.load(f)
visual_script = "\n".join([f"[{item['time_sec']}s]: {item['description']}" for item in visual_data])

# 读取听觉剧本
audio_script = "无"
if os.path.exists(audio_path):
    with open(audio_path, 'r', encoding='utf-8') as f:
        audio_data = json.load(f)
    if len(audio_data) > 0:
        audio_script = "\n".join([f"[{item['start']}s - {item['end']}s]: {item['text']}" for item in audio_data])

# ================= 3. 构造上帝视角的究极 Prompt =================
system_prompt = """你是一个顶级的视频片段检索专家。
你将收到一段教学/日常视频的【视觉画面剧本】和【听觉语音剧本】。
请根据用户输入的【查询动作】，结合两个剧本的内容，推理出该动作发生的最准确的起止时间。

【核心原则】：
1. 容忍异步：如果听觉中提到动作，但视觉还未出现，请以视觉真实发生的时刻为准。
2. 缺失容忍：如果听觉剧本为空（日常视频），请完全依赖视觉剧本进行推理。
3. 输出格式：为了方便后续代码解析，你的最终输出必须是一个合法的 JSON 格式，且只包含 start 和 end 两个字段。绝对不要输出任何其他多余的解释文字！
格式示例：{"start": 12.0, "end": 18.0}
"""

user_prompt = f"""
【视觉画面剧本】：
{visual_script}

【听觉语音剧本】：
{audio_script}

【查询动作】：{query}

请推理时间戳，并仅返回 JSON 格式结果：
"""

# ================= 4. 调用大模型 =================
print("正在召唤大模型进行零样本推理...\n")
response = client.chat.completions.create(
    model="deepseek-chat", # 使用对应平台的模型名
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.1 # 调低温度，让输出更理性、稳定
)

result = response.choices[0].message.content
print("============ 大模型的最终输出 ============")
print(result)
print("==========================================")
