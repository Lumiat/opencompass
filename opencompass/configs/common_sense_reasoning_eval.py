import argparse
import sys
from mmengine.config import read_base

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
    print(f"Error: Unsupported dataset '{dataset_name}'. Supported datasets: {list(dataset_configs.keys())}")
    sys.exit(1)

# dynamic load datasets configuration
config = dataset_configs[dataset_name]
with read_base():
    exec(f"from {config['gen']} import {config['gen_var']}")
    exec(f"from {config['ppl']} import {config['ppl_var']}")

# set datasets
datasets = eval(f"[*{config['gen_var']}, *{config['ppl_var']}]")

# dynamic models configuration
# model_configs = {
#     'Qwen2.5-0.5B-Instruct': {
#         'path': '/research-intern02/xjy/ParaGen-Dataset/models/Qwen2.5-0.5B-Instruct',
#         'max_out_len': 128,
#         'batch_size': 8,
#     },
#     'Mistral-7B-Instruct-v0.3': {
#         'path': '/research-intern02/xjy/ParaGen-Dataset/models/Mistral-7B-Instruct-v0.3',
#         'max_out_len': 128,
#         'batch_size': 8,
#     },
#     'bert-base-multilingual-cased': {
#         'path': '/research-intern02/xjy/ParaGen-Dataset/models/bert-base-multilingual-cased',
#         'max_out_len': 128,
#         'batch_size': 8,
#     },
#     'gpt2': {
#         'path': '/research-intern02/xjy/ParaGen-Dataset/models/gpt2',
#         'max_out_len': 128,
#         'batch_size': 8,
#     },
#     'llama-7b': {
#         'path': '/research-intern02/xjy/ParaGen-Dataset/models/llama-7b',
#         'max_out_len': 128,
#         'batch_size': 8,
#     },
#     'gemma-3-4b-it': {
#         'path': '/research-intern02/xjy/ParaGen-Dataset/models/gemma-3-4b-it',
#         'max_out_len': 128,
#         'batch_size': 8,
#     },
# }

from opencompass.models import HuggingFaceCausalLM

# dynamic models configuration
models = []
for i in range(1, 101):
    checkpoint_num = f"{i:03d}"  # formulate to 001, 002, ..., 100
    models.append(
        dict(
            type=HuggingFaceCausalLM,
            abbr=f'{model_name}_rank{rank}_checkpoint{checkpoint_num}',
            path=f'/research-intern02/xjy/ParaGen-Dataset/models/{model_name}',
            peft_path=f'/research-intern02/xjy/ParaGen-Dataset/saves/common_sense_reasoning/{dataset_name}/{model_name}_lora-rank_{rank}_finetune/checkpoint-{checkpoint_num}',
            max_out_len=128,
            batch_size=8,
            run_cfg=dict(num_gpus=1),
        )
    )

# set work_dir
work_dir = f'./outputs/{model_name}_{dataset_name}_{rank}_eval'
