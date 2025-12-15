from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import WinograndeDatasetV2
from opencompass.utils.text_postprocessors import first_option_postprocess

winogrande_reader_cfg = dict(
    input_columns=['prompt', 'only_option1', 'only_option2'],
    output_column='answer',
)

system_prompt='You are an AI assistant capable of commonsense reasoning. Read the sentence with the missing part and select the option that most logically fills the blank. Output only the letter of the correct answer (A or B).'

winogrande_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role='SYSTEM',
                    prompt=system_prompt
                )
            ],
            round=[
                dict(role='HUMAN', 
                     prompt='Question: {prompt}\nA. {only_option1}\nB. {only_option2}\nAnswer:'),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

winogrande_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='AB'),
)

winogrande_datasets_gen = [
    dict(
        abbr='winogrande-test-gen',
        type=WinograndeDatasetV2,
        path='opencompass/winogrande',
        reader_cfg=winogrande_reader_cfg,
        infer_cfg=winogrande_infer_cfg,
        eval_cfg=winogrande_eval_cfg,
    )
]
