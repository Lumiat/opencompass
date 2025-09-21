from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import OBQADataset
from opencompass.utils.text_postprocessors import first_option_postprocess

obqa_reader_cfg = dict(
    input_columns=['question_stem', 'A', 'B', 'C', 'D'], output_column='answerKey'
)

obqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                    dict(
                        role='HUMAN',
                        prompt=
                        'you are a helpful AI assistant, and you are going to answer the question of the user by picking one answer among the given choices. Answer the capital character of the choice directly. You\'ll only need to answer by a single [ans] (ans is A,B,C,D or True/False)\n{question_stem}\nA: {A}\nB: {B}\nC: {C}\nD: {D}\nPlease answer the question choosing from [A]/[B]/[C]/[D].'
                    ),
                ],
        )
     ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

obqa_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)

obqa_datasets_gen = [
    dict(
        abbr='openbookqa-test-gen',
        type=OBQADataset,
        path='opencompass/openbookqa_test',
        name='main',
    ),
]