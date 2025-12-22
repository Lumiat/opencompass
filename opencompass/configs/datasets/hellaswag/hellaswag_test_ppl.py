from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HellaswagDataset

hellaswag_reader_cfg = dict(
    input_columns=['ctx', 'A', 'B', 'C', 'D'],
    output_column='label')

system_prompt='You are an expert in commonsense reasoning. Select the most plausible continuation for the given context from options A, B, C, and D. Answer the capital character of the choice directly.'

hellaswag_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'A':
            dict(
                begin=[dict(role='SYSTEM', prompt=system_prompt)],
                round=[
                    dict(role='HUMAN', prompt='{ctx}'),
                    dict(role='BOT', prompt='{A}')
                ], ),
            'B':
            dict(
                begin=[dict(role='SYSTEM', prompt=system_prompt)],
                round=[
                    dict(role='HUMAN', prompt='{ctx}'),
                    dict(role='BOT', prompt='{B}')
                ], ),
            'C':
            dict(
                begin=[dict(role='SYSTEM', prompt=system_prompt)],
                round=[
                    dict(role='HUMAN', prompt='{ctx}'),
                    dict(role='BOT', prompt='{C}')
                ], ),
            'D':
            dict(
                begin=[dict(role='SYSTEM', prompt=system_prompt)],
                round=[
                    dict(role='HUMAN', prompt='{ctx}'),
                    dict(role='BOT', prompt='{D}')
                ], ),
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

hellaswag_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

hellaswag_datasets_ppl = [
    dict(
        abbr='hellaswag-test-ppl',
        type=HellaswagDataset,
        path='opencompass/hellaswag',
        reader_cfg=hellaswag_reader_cfg,
        infer_cfg=hellaswag_infer_cfg,
        eval_cfg=hellaswag_eval_cfg)
]
