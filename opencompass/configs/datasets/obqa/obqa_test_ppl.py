from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import OBQADataset

obqa_reader_cfg = dict(
    input_columns=['question_stem', 'A', 'B', 'C', 'D'], output_column='answerKey'
)

obqa_infer_cfg = dict(
    prompt_template=dict(
    type=PromptTemplate,
    template={
        'A':
            dict(
                round=[
                    dict(role='HUMAN', prompt='Question: {question_stem}\nAnswer: '),
                    dict(role='BOT', prompt='{A}')
                ], ),
            'B':
            dict(
                round=[
                    dict(role='HUMAN', prompt='Question: {question_stem}\nAnswer: '),
                    dict(role='BOT', prompt='{B}')
                ], ),
            'C':
            dict(
                round=[
                    dict(role='HUMAN', prompt='Question: {question_stem}\nAnswer: '),
                    dict(role='BOT', prompt='{C}')
                ], ),
            'D':
            dict(
                round=[
                    dict(role='HUMAN', prompt='Question: {question_stem}\nAnswer: '),
                    dict(role='BOT', prompt='{D}')
                ], ),
    },),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)

obqa_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=AccEvaluator),
)

obqa_datasets_ppl = [
    dict(
        abbr='openbookqa-test-ppl',
        type=OBQADataset,
        path='opencompass/openbookqa_test',
        name='main',
    ),
]