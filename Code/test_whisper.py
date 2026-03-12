import whisper

# 换成你刚才解压出来的、真实存在的任意一个做菜视频的名字（比如 0001.mp4）
video_path = "/mnt/data/VMR/YouCook2_Dataset/YouCookIIVideos/val/zqTXQ-YqrgQ_8.mp4" 

print("正在加载 Whisper 基础模型...")
model = whisper.load_model("base") 

print("正在提取字幕，请稍候...")
result = model.transcribe(video_path)

print("========= 提取成功！结果如下 =========")
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
