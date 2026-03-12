import json
import sys

def calculate_iou(pred_start, pred_end, gt_start, gt_end):
    """纯净的IoU计算，不做启发式平滑"""
    try:
        pred_start = float(pred_start) if pred_start is not None else 0.0
        pred_end = float(pred_end) if pred_end is not None else 0.0
    except:
        pred_start, pred_end = 0.0, 0.0

    # 交换保证 start < end
    if pred_end < pred_start:
        pred_start, pred_end = pred_end, pred_start

    # 计算交并比
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0, intersection_end - intersection_start)

    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def evaluate(result_file):
    ious = []
    hit_03, hit_05, hit_07 = 0, 0, 0
    total = 0

    # 按置信度区间统计
    high_audio_ious = []  # audio_conf >= 0.5
    low_audio_ious = []   # audio_conf < 0.3

    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            iou = calculate_iou(
                data['pred_start'], data['pred_end'],
                data['gt_start'], data['gt_end']
            )
            ious.append(iou)

            if iou >= 0.3: hit_03 += 1
            if iou >= 0.5: hit_05 += 1
            if iou >= 0.7: hit_07 += 1
            total += 1

            # 分组统计（如果有置信度字段）
            audio_conf = data.get("audio_confidence", -1)
            if audio_conf >= 0.5:
                high_audio_ious.append(iou)
            elif 0 <= audio_conf < 0.3:
                low_audio_ious.append(iou)

    print("\n" + "=" * 50)
    print("🏆 评测结果")
    print("=" * 50)
    print(f"总测试样本数: {total}")
    print(f"R@1, IoU=0.3 : {hit_03 / total * 100:.2f}%")
    print(f"R@1, IoU=0.5 : {hit_05 / total * 100:.2f}%")
    print(f"R@1, IoU=0.7 : {hit_07 / total * 100:.2f}%")
    print(f"mIoU         : {sum(ious) / total * 100:.2f}%")

    # 分组结果（用于消融分析）
    if high_audio_ious:
        avg_high = sum(high_audio_ious) / len(high_audio_ious)
        print(f"\n📊 高听觉置信度样本 (n={len(high_audio_ious)}): mIoU = {avg_high*100:.2f}%")
    if low_audio_ious:
        avg_low = sum(low_audio_ious) / len(low_audio_ious)
        print(f"📊 低听觉置信度样本 (n={len(low_audio_ious)}): mIoU = {avg_low*100:.2f}%")

    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result_file = sys.argv[1]
    else:
        result_file = "/mnt/data/VMR/Results/final_predictions_charades_v2.jsonl"
    evaluate(result_file)
