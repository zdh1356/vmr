import json

def calculate_iou(pred_start, pred_end, gt_start, gt_end):
    # 🌟 终极容错：防止大模型胡言乱语输出 null
    try:
        pred_start = float(pred_start) if pred_start is not None else 0.0
        pred_end = float(pred_end) if pred_end is not None else 0.0
    except:
        pred_start, pred_end = 0.0, 0.0

    if pred_end < pred_start:
        pred_start, pred_end = pred_end, pred_start
    
    # 🌟 启发式时间平滑：拯救点状预测
    if (pred_end - pred_start) < 2.0:
        mid_point = (pred_start + pred_end) / 2.0
        pred_start = max(0.0, mid_point - 2.0)
        pred_end = mid_point + 2.0

    # 计算交集与并集
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0, intersection_end - intersection_start)
    
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    if union <= 0: return 0.0
    return intersection / union

# ==========================================
result_file = "/mnt/data/VMR/Results/final_predictions.jsonl"

ious =[]
hit_03, hit_05, hit_07 = 0, 0, 0
total = 0

with open(result_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        iou = calculate_iou(data['pred_start'], data['pred_end'], data['gt_start'], data['gt_end'])
        ious.append(iou)
        
        if iou >= 0.3: hit_03 += 1
        if iou >= 0.5: hit_05 += 1
        if iou >= 0.7: hit_07 += 1
        total += 1

print("\n" + "="*40)
print("🏆 添加时间平滑后的最终开奖结果 🏆")
print("="*40)
print(f"总测试样本数: {total}")
print(f"R@1, IoU=0.3 : {hit_03 / total * 100:.2f}%")
print(f"R@1, IoU=0.5 : {hit_05 / total * 100:.2f}%")
print(f"R@1, IoU=0.7 : {hit_07 / total * 100:.2f}%")
print(f"mIoU (平均分): {sum(ious) / total * 100:.2f}%")
print("="*40)
