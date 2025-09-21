from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import BoolQDatasetV4

BoolQ_reader_cfg = dict(
    input_columns='question',
    output_column='answer',
    test_split='validation')

BoolQ_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'False':
            dict(round=[
                dict(role='HUMAN', prompt='{question}?'),
                dict(role='BOT', prompt='No'),
            ]),
            'True':
            dict(round=[
                dict(role='HUMAN', prompt='{question}?'),
                dict(role='BOT', prompt='Yes'),
            ]),
        },
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)

BoolQ_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

BoolQ_datasets_ppl = [
    dict(
        type=BoolQDatasetV4,
        abbr='BoolQ-test-ppl',
        path='opencompass/boolq',
        reader_cfg=BoolQ_reader_cfg,
        infer_cfg=BoolQ_infer_cfg,
        eval_cfg=BoolQ_eval_cfg,
    )
]
