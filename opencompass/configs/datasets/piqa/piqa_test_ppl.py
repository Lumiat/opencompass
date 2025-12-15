from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer 
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import PIQADatasetV2
from opencompass.utils.text_postprocessors import first_option_postprocess
from piqa_utils import piqa_label_to_AB 


piqa_reader_cfg = dict(
    input_columns=['goal', 'sol1', 'sol2'],
    output_column='answer', 
    test_split='validation')

system_prompt_text = "you are a helpful AI assistant, and you are going to find the better solution to a specific problem from the given 2 solutions (A and B). Answer the capital character of the better solution directly."

piqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'A': dict(
                begin=[dict(role='SYSTEM', prompt=system_prompt_text)],
                round=[
                    dict(role='HUMAN', prompt='{goal}\nA. {sol1}\nB. {sol2}\nAnswer:'),
                    dict(role='BOT', prompt='A')
                ]
            ),
            'B': dict(
                begin=[dict(role='SYSTEM', prompt=system_prompt_text)],
                round=[
                    dict(role='HUMAN', prompt='{goal}\nA. {sol1}\nB. {sol2}\nAnswer:'),
                    dict(role='BOT', prompt='B')
                ]
            ),
        }
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)

piqa_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
)

piqa_datasets_ppl = [
    dict(
        abbr='piqa_ppl',
        type=PIQADatasetV2,
        path='opencompass/piqa',
        reader_cfg=piqa_reader_cfg,
        infer_cfg=piqa_infer_cfg,
        eval_cfg=piqa_eval_cfg,
    )
]
