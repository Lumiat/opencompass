from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import LLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import WinograndeDatasetV2

winogrande_reader_cfg = dict(
    input_columns=['only_option1', 'only_option2'],
    output_column='answer',
)

system_prompt='You are an AI assistant capable of commonsense reasoning. Read the sentence with the missing part and select the option that most logically fills the blank. Output only the letter of the correct answer (A or B).'


winogrande_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'A': dict(
                begin=[dict(role='SYSTEM', prompt=system_prompt)],
                round=[
                    dict(role='HUMAN', prompt='{prompt}\nA. {only_option1}\nB. {only_option2}\nAnswer:'),
                    dict(role='BOT', prompt='A')
                ]
            ),
            'B': dict(
                begin=[dict(role='SYSTEM', prompt=system_prompt)],
                round=[
                    dict(role='HUMAN', prompt='{prompt}\nA. {only_option1}\nB. {only_option2}\nAnswer:'),
                    dict(role='BOT', prompt='B')
                ]
            ),
        }
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=LLInferencer))

winogrande_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

winogrande_datasets_ppl = [
    dict(
        abbr='winogrande-test-ppl',
        type=WinograndeDatasetV2,
        path='opencompass/winogrande',
        reader_cfg=winogrande_reader_cfg,
        infer_cfg=winogrande_infer_cfg,
        eval_cfg=winogrande_eval_cfg)
]
