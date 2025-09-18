from mmengine.config import read_base
with read_base():
    from .datasets.ARC_c.ARC_c_test_gen import ARC_c_datasets_gen
    from .datasets.ARC_c.ARC_c_test_ppl import ARC_c_datasets_ppl

datasets = [*ARC_c_datasets_gen, *ARC_c_datasets_ppl]
# datasets = [*ARC_c_datasets_gen]

# from opencompass.models import HuggingFacewithChatTemplate
from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen2.5-0.5b-instruct-hf-rank8-checkpoint001',
        path='/research-intern02/xjy/ParaGen-Dataset/models/Qwen2.5-0.5B-Instruct',
        peft_path='/research-intern02/xjy/ParaGen-Dataset/saves/common_sense_reasoning/ARC-c/Qwen2.5-0.5B-Instruct_lora-rank_8_finetune/checkpoint-100',
        max_out_len=128, # original: 4096
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    ),
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen2.5-0.5b-instruct-hf-rank8-checkpoint002',
        path='/research-intern02/xjy/ParaGen-Dataset/models/Qwen2.5-0.5B-Instruct',
        peft_path='/research-intern02/xjy/ParaGen-Dataset/saves/common_sense_reasoning/ARC-c/Qwen2.5-0.5B-Instruct_lora-rank_8_finetune/checkpoint-002',
        max_out_len=128, # original: 4096
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    ),
]

work_dir = './outputs/qwen2_5_arc_c_lora_eval'
