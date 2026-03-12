"""
听觉置信度检测模块
判断Whisper输出是否为幻觉/噪声
"""
import json
import re

# Whisper常见的幻觉模式
HALLUCINATION_PATTERNS = [
    r"thank\s*you",
    r"subscribe",
    r"like\s*and\s*share",
    r"please\s*subscribe",
    r"thanks?\s*for\s*watching",
    r"see\s*you\s*next\s*time",
    r"bye\s*bye",
    r"music",
    r"applause",
    r"laughter",
    r"silence",
    r"\.{3,}",  # 连续省略号
    r"^\s*$",   # 空白
]

def calculate_audio_confidence(segments):
    """
    计算听觉流的置信度分数 (0.0 ~ 1.0)
    
    判断维度:
    1. 是否有实质性内容
    2. 是否包含Whisper幻觉关键词
    3. 片段数量和平均长度是否合理
    4. 是否存在大量重复内容
    """
    if not segments or len(segments) == 0:
        return 0.0, "empty"
    
    total_text = " ".join([s.get("text", "") for s in segments]).strip()
    
    # 检查1: 总文本长度太短
    if len(total_text) < 10:
        return 0.05, "too_short"
    
    # 检查2: 幻觉关键词比例
    hallucination_count = 0
    for seg in segments:
        text = seg.get("text", "").lower().strip()
        for pattern in HALLUCINATION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                hallucination_count += 1
                break
    
    hallucination_ratio = hallucination_count / max(len(segments), 1)
    
    # 检查3: 重复内容检测
    unique_texts = set()
    for seg in segments:
        text = seg.get("text", "").strip().lower()
        if len(text) > 5:
            unique_texts.add(text)
    
    if len(segments) > 3:
        diversity_ratio = len(unique_texts) / len(segments)
    else:
        diversity_ratio = 1.0
    
    # 检查4: 平均片段时长是否合理（正常语音片段2-10秒）
    valid_duration_count = 0
    for seg in segments:
        dur = seg.get("end", 0) - seg.get("start", 0)
        if 0.5 <= dur <= 30.0:
            valid_duration_count += 1
    duration_ratio = valid_duration_count / max(len(segments), 1)
    
    # 综合打分
    score = 1.0
    score -= hallucination_ratio * 0.5   # 幻觉越多，分越低
    score -= (1 - diversity_ratio) * 0.3  # 重复越多，分越低
    score -= (1 - duration_ratio) * 0.2   # 时长异常越多，分越低
    
    score = max(0.0, min(1.0, score))
    
    reason = "normal"
    if hallucination_ratio > 0.5:
        reason = "high_hallucination"
    elif diversity_ratio < 0.3:
        reason = "high_repetition"
    
    return round(score, 3), reason


def filter_audio_segments(segments, min_confidence=0.3):
    """
    过滤掉明显是噪声的片段
    """
    if not segments:
        return [], 0.0
    
    confidence, reason = calculate_audio_confidence(segments)
    
    if confidence < min_confidence:
        return [], confidence  # 置信度太低，直接丢弃整个听觉流
    
    # 逐条过滤幻觉片段
    filtered = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if len(text) < 3:
            continue
        
        is_hallucination = False
        for pattern in HALLUCINATION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                is_hallucination = True
                break
        
        if not is_hallucination:
            filtered.append(seg)
    
    return filtered, confidence
