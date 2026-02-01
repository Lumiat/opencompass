from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import OBQADataset

obqa_reader_cfg = dict(
    input_columns=['question_stem', 'A', 'B', 'C', 'D', 'fact1'], output_column='answerKey'
)

system_prompt='you are a helpful AI assistant, and you are going to answer the question of the user by picking one answer among the given choices. Answer the capital character of the choice directly.'

obqa_infer_cfg = dict(
    prompt_template=dict(
    type=PromptTemplate,
    template={
            'A':dict(
                begin=[dict(role='SYSTEM', prompt=system_prompt)],
                round=[
                    dict(role='HUMAN', prompt='Given the fact: {fact1}\nQuestion: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:'),
                    dict(role='BOT', prompt='A')
                ], ),
            'B':dict(
                begin=[dict(role='SYSTEM', prompt=system_prompt)],
                round=[
                    dict(role='HUMAN', prompt='Given the fact: {fact1}\nQuestion: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:'),
                    dict(role='BOT', prompt='B')
                ], ),
            'C':dict(
                begin=[dict(role='SYSTEM', prompt=system_prompt)],
                round=[
                    dict(role='HUMAN', prompt='Given the fact: {fact1}\nQuestion: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:'),
                    dict(role='BOT', prompt='C')
                ], ),
            'D':dict(
                begin=[dict(role='SYSTEM', prompt=system_prompt)],
                round=[
                    dict(role='HUMAN', prompt='Given the fact: {fact1}\nQuestion: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:'),
                    dict(role='BOT', prompt='D')
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
        abbr='obqa-test-ppl',
        type=OBQADataset,
        path='opencompass/openbookqa_test',
        name='additional',
        reader_cfg=obqa_reader_cfg,
        infer_cfg=obqa_infer_cfg,
        eval_cfg=obqa_eval_cfg,
    ),
]