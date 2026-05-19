import os
from modelscope import snapshot_download

# 保持缓存路径与刚才一致，绝对隔离
os.environ['MODELSCOPE_CACHE'] = '/nas_data/xueyue.yang/guangze/ms_cache'

print("开始从魔搭下载 Llama-2-7b...")

# 执行下载，使用 Llama 2 7B 的魔搭镜像 ID
model_dir = snapshot_download(
    model_id='modelscope/Llama-2-7b-ms', 
    local_dir='/nas_data/xueyue.yang/guangze/hf_models/Llama-2-7b-hf'
)

print(f"下载成功！模型已完整保存在: {model_dir}")