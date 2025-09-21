from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import PIQADatasetV2
from opencompass.utils.text_postprocessors import first_option_postprocess

piqa_reader_cfg = dict(
    input_columns=['goal', 'sol1', 'sol2'],
    output_column='answer',
    test_split='validation')

piqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'A':
            dict(
                round=[
                    dict(role='HUMAN', prompt='{goal}'),
                    dict(role='BOT', prompt='{sol1}')
                ], ),
            'B':
            dict(
                round=[
                    dict(role='HUMAN', prompt='{goal}'),
                    dict(role='BOT', prompt='{sol2}')
                ], ),
        }
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

piqa_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess),
)

piqa_datasets_ppl = [
    dict(
        abbr='piqa-test-ppl',
        type=PIQADatasetV2,
        path='opencompass/piqa',
        reader_cfg=piqa_reader_cfg,
        infer_cfg=piqa_infer_cfg,
        eval_cfg=piqa_eval_cfg)
]
