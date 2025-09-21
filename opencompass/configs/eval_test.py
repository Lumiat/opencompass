import argparse
import sys
from pathlib import Path

# parse command parameters
parser = argparse.ArgumentParser(description='OpenCompass evaluation with configurable parameters')
parser.add_argument('--dataset', type=str, required=True,
                    help='Dataset name (ARC-c, ARC-e, BoolQ, PIQA, OBQA, HellaSwag, WinoGrande)')
parser.add_argument('--model', type=str, required=True,
                    help='Model name (Qwen2.5-0.5B-Instruct, Mistral-7B-Instruct-v0.3, bert-base-multilingual-cased, gpt2, llama-7b, gemma-3-4b-it)')
parser.add_argument('--rank', type=int, required=True,
                    help='LoRA rank value (2, 4, 8, 16, 64)')

# get parameters from command
args, unknown = parser.parse_known_args()
dataset_name = args.dataset
model_name = args.model
rank = args.rank

output_file = Path(f"./test_configs/{dataset_name}/{model_name}_{rank}_eval.py")
output_file.parent.mkdir(parents=True, exist_ok=True)

# dynamic model configuration
dataset_configs = {
    'ARC-c': {
        'gen': '.datasets.ARC_c.ARC_c_test_gen',
        'ppl': '.datasets.ARC_c.ARC_c_test_ppl',
        'gen_var': 'ARC_c_datasets_gen',
        'ppl_var': 'ARC_c_datasets_ppl'
    },
    'ARC-e': {
        'gen': '.datasets.ARC_e.ARC_e_test_gen',
        'ppl': '.datasets.ARC_e.ARC_e_test_ppl',
        'gen_var': 'ARC_e_datasets_gen',
        'ppl_var': 'ARC_e_datasets_ppl'
    },
    'BoolQ': {
        'gen': '.datasets.BoolQ.BoolQ_test_gen',
        'ppl': '.datasets.BoolQ.BoolQ_test_ppl',
        'gen_var': 'BoolQ_datasets_gen',
        'ppl_var': 'BoolQ_datasets_ppl'
    },
    'PIQA': {
        'gen': '.datasets.piqa.piqa_test_gen',
        'ppl': '.datasets.piqa.piqa_test_ppl',
        'gen_var': 'PIQA_datasets_gen',
        'ppl_var': 'PIQA_datasets_ppl'
    },
    'OBQA': {
        'gen': '.datasets.obqa.obqa_test_gen',
        'ppl': '.datasets.obqa.obqa_test_ppl',
        'gen_var': 'obqa_datasets_gen',
        'ppl_var': 'obqa_datasets_ppl'
    },
    'HellaSwag': {
        'gen': '.datasets.hellaswag.hellaswag_test_gen',
        'ppl': '.datasets.hellaswag.hellaswag_test_ppl',
        'gen_var': 'hellaswag_datasets_gen',
        'ppl_var': 'hellaswag_datasets_ppl'
    },
    'WinoGrande': {
        'gen': 'datasets.winogrande.winogrande_test_gen',
        'ppl': 'datasets.winogrande.winogrande_test_ppl',
        'gen_var': 'winogrande_datasets_gen',
        'ppl_var': 'winogrande_datasets_ppl'
    }
}

# validate dataset configuration
if dataset_name not in dataset_configs:
    print(f"[Error] Unsupported dataset '{dataset_name}'. Supported datasets: {list(dataset_configs.keys())}")
    sys.exit(1)

dataset_cfg = dataset_configs[dataset_name]

# generate configuration files
template = f"""
from mmengine.config import read_base
# dynamic load datasets configuration
with read_base():
    from {dataset_cfg['gen']} import {dataset_cfg['gen_var']}
    from {dataset_cfg['ppl']} import {dataset_cfg['ppl_var']}

# set datasets
datasets = [*{dataset_cfg['gen_var']}, *{dataset_cfg['ppl_var']}]

from opencompass.models import HuggingFaceCausalLM

# dynamic models configuration

models=[
    dict(
        type=HuggingFaceCausalLM,
        abbr='{model_name}_rank{rank}_checkpoint-001',
        path='/research-intern02/xjy/ParaGen-Dataset/models/{model_name}',
        peft_path='/research-intern02/xjy/ParaGen-Dataset/saves/common_sense_reasoning/{dataset_name}/{model_name}_lora-rank_{rank}_finetune/checkpoint-001',
        max_out_len=128,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]

# set work_dir
work_dir = f'./outputs/{model_name}_{dataset_name}_{rank}_eval_test'
"""

output_file.write_text(template, encoding="utf8")
print(f"[INFO] Config file generated at: {output_file}")
print(f"[INFO] Use checkpoint-001 to test")
print(f"[INFO] Run evaluation with {output_file}")
