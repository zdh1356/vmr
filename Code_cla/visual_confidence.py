"""
视觉置信度检测模块
判断LLaVA描述是否有效
"""

# LLaVA常见的低质量描述模式
LOW_QUALITY_PATTERNS = [
    "i cannot",
    "i can't",
    "the image shows",
    "this is an image",
    "blurry",
    "dark image",
    "black screen",
    "no visible",
]

def calculate_visual_confidence(descriptions):
    """
    计算视觉流的置信度分数 (0.0 ~ 1.0)
    """
    if not descriptions or len(descriptions) == 0:
        return 0.0, "empty"
    
    valid_count = 0
    total = len(descriptions)
    
    for desc_item in descriptions:
        text = desc_item.get("description", "").lower().strip()
        
        if len(text) < 5:
            continue
        
        is_low_quality = False
        for pattern in LOW_QUALITY_PATTERNS:
            if pattern in text:
                is_low_quality = True
                break
        
        if not is_low_quality:
            valid_count += 1
    
    # 检查描述多样性（如果所有帧描述都一样，说明视觉流质量差）
    unique_descs = set()
    for desc_item in descriptions:
        text = desc_item.get("description", "").strip().lower()
        if len(text) > 10:
            unique_descs.add(text[:50])  # 取前50字符去重
    
    diversity = len(unique_descs) / max(total, 1)
    validity = valid_count / max(total, 1)
    
    score = validity * 0.6 + diversity * 0.4
    score = max(0.0, min(1.0, score))
    
    reason = "normal"
    if validity < 0.3:
        reason = "low_validity"
    elif diversity < 0.2:
        reason = "low_diversity"
    
    return round(score, 3), reason
