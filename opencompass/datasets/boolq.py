import json

from datasets import Dataset, load_dataset
from datasets import DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset
from os import environ
import os

@LOAD_DATASET.register_module()
class BoolQDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            if example['label'] == 'true':
                example['answer'] = 1
            else:
                example['answer'] = 0
            return example

        dataset = dataset.map(preprocess)
        return dataset


@LOAD_DATASET.register_module()
class BoolQDatasetV2(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path)
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                line['label'] = {'true': 'A', 'false': 'B'}[line['label']]
                dataset.append(line)
        return Dataset.from_list(dataset)


@LOAD_DATASET.register_module()
class BoolQDatasetV3(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                line['passage'] = ' -- '.join(
                    line['passage'].split(' -- ')[1:])
                line['question'] = line['question'][0].upper(
                ) + line['question'][1:]
                dataset.append(line)
        return Dataset.from_list(dataset)


@LOAD_DATASET.register_module()
class BoolQDatasetV4(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)
        
        def preprocess(example):
            if example['label'] == 'true':
                example['answer'] = 'True'
            else:
                example['answer'] = 'False'
            return example

        dataset = dataset.map(preprocess)
        return dataset
    
@LOAD_DATASET.register_module()
class BoolQDatasetV5(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            ms_dataset = MsDataset.load(path, split='validation')
            dataset_list = []
            for line in ms_dataset:
                data_item = {
                    'passage': line['passage'],
                    'question': line['question'],
                    'label': 'A' if line['answer'] == 'true' else 'B',
                }
                dataset_list.append(data_item)
            dataset_list = Dataset.from_list(dataset_list)
        else:
            dataset_list = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)
                    data_item = {
                        'passage': line['passage'],
                        'question': line['question'],
                        'label': 'A' if line['answer'] == 'true' else 'B',
                    }
                    dataset_list.append(data_item)
            dataset_list = Dataset.from_list(dataset_list)

        return DatasetDict({'validation': dataset_list})
