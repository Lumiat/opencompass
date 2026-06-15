import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class IndustryLMDataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise TypeError('IndustryLMDataset expects a JSON array.')

        for item in data:
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            item['prompt'] = (f'{instruction}\n\n{input_text}'
                              if input_text else instruction)

        return Dataset.from_list(data)
