from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import BoolQDatasetV4
from opencompass.utils.text_postprocessors import first_capital_postprocess

BoolQ_reader_cfg = dict(
    input_columns='question',
    output_column='answer',
    test_split='validation'
)

BoolQ_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='you are a helpful AI assistant, and you are going to answer the question of the user by determining the statement is true or false. Answer true or false directly. You\'ll only need to answer by a single [ans] (ans is A,B,C,D or True/False)\n{question}?'),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

BoolQ_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_capital_postprocess),
)

BoolQ_datasets_gen = [
    dict(
        abbr='BoolQ-test-gen',
        type=BoolQDatasetV4,
        path='opencompass/boolq',
        reader_cfg=BoolQ_reader_cfg,
        infer_cfg=BoolQ_infer_cfg,
        eval_cfg=BoolQ_eval_cfg,
    )
]
d