import os
from huggingface_hub import snapshot_download

# 1. 在代码里锁死国内加速镜像！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("🚀 开始通过 Python 原生 API 强行拉取 LLaVA 模型...")

# 2. 调用最底层的快照下载函数，直接写死路径
snapshot_download(
    repo_id="llava-hf/llava-1.5-7b-hf",
    local_dir="/mnt/data/VMR/cache/llava-1.5-7b-hf",
    local_dir_use_symlinks=False,  # 核心参数：直接下实体文件，不搞软链接
    resume_download=True,
    max_workers=8  # 开启 8 线程狂暴加速
)

print("🎉 LLaVA 视觉大模型下载彻底完成！")
