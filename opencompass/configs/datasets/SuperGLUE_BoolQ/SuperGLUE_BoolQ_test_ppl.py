from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import BoolQDatasetV5

BoolQ_reader_cfg = dict(
    input_columns=['question', 'passage'],
    output_column='label',
    test_split='validation')

system_prompt = 'You are a helpful assistant. Answer the question based only on the information provided in the passage, and select the corresponding option. Answer the capital character of the option directly.'

BoolQ_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'A':
            dict(
                begin=[dict(role='SYSTEM', prompt=system_prompt)],
                round=[
                dict(role='HUMAN', prompt='{passage}\nQuestion: {question}?\nA. Yes\nB. No\nAnswer:'),
                dict(role='BOT', prompt='Yes'),
            ]),
            'B':
            dict(
                begin=[dict(role='SYSTEM', prompt=system_prompt)],
                round=[
                dict(role='HUMAN', prompt='{passage}\nQuestion: {question}?\nA. Yes\nB. No\nAnswer:'),
                dict(role='BOT', prompt='No'),
            ]),
        },
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)

BoolQ_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

boolq_datasets_ppl = [
    dict(
        type=BoolQDatasetV5,
        abbr='BoolQ-test-ppl',
        path='opencompass/boolq',
        reader_cfg=BoolQ_reader_cfg,
        infer_cfg=BoolQ_infer_cfg,
        eval_cfg=BoolQ_eval_cfg,
    )
]
