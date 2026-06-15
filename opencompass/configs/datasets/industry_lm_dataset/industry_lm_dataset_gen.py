from opencompass.datasets import IndustryLMDataset
from opencompass.openicl.icl_evaluator import JiebaRougeEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

industry_lm_dataset_reader_cfg = dict(
    input_columns=['prompt'], output_column='output')

industry_lm_dataset_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='{prompt}'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1024),
)

industry_lm_dataset_eval_cfg = dict(
    evaluator=dict(type=JiebaRougeEvaluator))

industry_lm_dataset_datasets = [
    dict(
        type=IndustryLMDataset,
        abbr='industry_lm_dataset',
        path='./data/industry_lm_dataset/sample.json',
        reader_cfg=industry_lm_dataset_reader_cfg,
        infer_cfg=industry_lm_dataset_infer_cfg,
        eval_cfg=industry_lm_dataset_eval_cfg,
    )
]
