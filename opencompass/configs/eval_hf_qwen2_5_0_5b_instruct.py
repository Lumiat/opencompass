from mmengine.config import read_base
with read_base():
    from .datasets.ARC_c.ARC_c_test_gen import ARC_c_datasets

datasets = [*ARC_c_datasets]

from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-0.5b-instruct-hf',
        path='/research-intern02/xjy/ParaGen-Dataset/models/Qwen2.5-0.5B-Instruct',
        peft_path='/research-intern02/xjy/ParaGen-Dataset/saves/common_sense_reasoning/ARC-c/Qwen2.5-0.5B-Instruct_lora-rank_8_finetune/checkpoint-749_adapter_model.safetensors',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
