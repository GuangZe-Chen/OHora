import os
from modelscope import snapshot_download

# 1. 在代码中直接设置缓存路径，绝对隔离
os.environ['MODELSCOPE_CACHE'] = '/nas_data/xueyue.yang/guangze/ms_cache'

print("开始从魔搭下载 Llama-3-8B...")

# 2. 执行下载
model_dir = snapshot_download(
    model_id='LLM-Research/Meta-Llama-3-8B', 
    local_dir='/nas_data/xueyue.yang/guangze/hf_models/Meta-Llama-3-8B'
)

print(f"下载成功！模型已完整保存在: {model_dir}")