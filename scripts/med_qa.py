# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
In this work, we present the first free-form multiple-choice OpenQA dataset for solving medical problems, MedQA,
collected from the professional medical board exams. It covers three languages: English, simplified Chinese, and
traditional Chinese, and contains 12,723, 34,251, and 14,123 questions for the three languages, respectively. Together
with the question data, we also collect and release a large-scale corpus from medical textbooks from which the reading
comprehension models can obtain necessary knowledge for answering the questions.
"""

import os
import random
import json
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from bigbiohub import qa_features
from bigbiohub import BigBioConfig
from bigbiohub import Tasks

_LANGUAGES = ['English', "Chinese (Simplified)", "Chinese (Traditional, Taiwan)"]
_PUBMED = False
_LOCAL = False

# TODO: Add BibTeX citation
_CITATION = """\
@article{jin2021disease,
  title={What disease does this patient have? a large-scale open domain question answering dataset from medical exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={Applied Sciences},
  volume={11},
  number={14},
  pages={6421},
  year={2021},
  publisher={MDPI}
}
"""

_DATASETNAME = "med_qa"
_DISPLAYNAME = "MedQA"

_DESCRIPTION = """\
In this work, we present the first free-form multiple-choice OpenQA dataset for solving medical problems, MedQA,
collected from the professional medical board exams. It covers three languages: English, simplified Chinese, and
traditional Chinese, and contains 12,723, 34,251, and 14,123 questions for the three languages, respectively. Together
with the question data, we also collect and release a large-scale corpus from medical textbooks from which the reading
comprehension models can obtain necessary knowledge for answering the questions.
"""

_HOMEPAGE = "https://github.com/jind11/MedQA"

_LICENSE = 'UNKNOWN'

_URLS = {
    _DATASETNAME: "data_clean.zip",
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

_SUBSET2NAME = {
    "en": "English",
    "zh": "Chinese (Simplified)",
    "tw": "Chinese (Traditional, Taiwan)",
    "tw_en": "Chinese (Traditional, Taiwan) translated to English",
    "tw_zh": "Chinese (Traditional, Taiwan) translated to Chinese (Simplified)",
}


class MedQADataset(datasets.GeneratorBasedBuilder):
    """Free-form multiple-choice OpenQA dataset covering three languages."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = []

    for subset in ["en", "zh", "tw", "tw_en", "tw_zh"]:
        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"med_qa_{subset}_source",
                version=SOURCE_VERSION,
                description=f"MedQA {_SUBSET2NAME.get(subset)} source schema",
                schema="source",
                subset_id=f"med_qa_{subset}",
            )
        )
        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"med_qa_{subset}_bigbio_qa",
                version=BIGBIO_VERSION,
                description=f"MedQA {_SUBSET2NAME.get(subset)} BigBio schema",
                schema="bigbio_qa",
                subset_id=f"med_qa_{subset}",
            )
        )
        if subset == "en" or subset == "zh":
            BUILDER_CONFIGS.append(
                BigBioConfig(
                    name=f"med_qa_{subset}_4options_source",
                    version=SOURCE_VERSION,
                    description=f"MedQA {_SUBSET2NAME.get(subset)} source schema (4 options)",
                    schema="source",
                    subset_id=f"med_qa_{subset}_4options",
                )
            )
            BUILDER_CONFIGS.append(
                BigBioConfig(
                    name=f"med_qa_{subset}_4options_bigbio_qa",
                    version=BIGBIO_VERSION,
                    description=f"MedQA {_SUBSET2NAME.get(subset)} BigBio schema (4 options)",
                    schema="bigbio_qa",
                    subset_id=f"med_qa_{subset}_4options",
                )
            )

    DEFAULT_CONFIG_NAME = "med_qa_en_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.name == "med_qa_en_4options_source":
            features = datasets.Features(
                {
                    "meta_info": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer_idx": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "options": [
                        {
                            "key": datasets.Value("string"),
                            "value": datasets.Value("string"),
                        }
                    ],
                    "metamap_phrases": datasets.Sequence(datasets.Value("string")),
                }
            )
        elif self.config.schema == "source":
            features = datasets.Features(
                {
                    "meta_info": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer_idx": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "options": [
                        {
                            "key": datasets.Value("string"),
                            "value": datasets.Value("string"),
                        }
                    ],
                }
            )
        elif self.config.schema == "bigbio_qa":
            features = qa_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)
        lang_dict = {"en": "US", "zh": "Mainland", "tw": "Taiwan"}
        base_dir = os.path.join(data_dir, "data_clean", "questions")
        if self.config.subset_id in ["med_qa_en", "med_qa_zh", "med_qa_tw"]:
            lang_path = lang_dict.get(self.config.subset_id.rsplit("_", 1)[1])
            paths = {
                "train": os.path.join(base_dir, lang_path, "train.jsonl"),
                "test": os.path.join(base_dir, lang_path, "test.jsonl"),
                "valid": os.path.join(base_dir, lang_path, "dev.jsonl"),
            }
        elif self.config.subset_id == "med_qa_tw_en":
            paths = {
                "train": os.path.join(
                    base_dir, "Taiwan", "tw_translated_jsonl", "en", "train-2en.jsonl"
                ),
                "test": os.path.join(
                    base_dir, "Taiwan", "tw_translated_jsonl", "en", "test-2en.jsonl"
                ),
                "valid": os.path.join(
                    base_dir, "Taiwan", "tw_translated_jsonl", "en", "dev-2en.jsonl"
                ),
            }
        elif self.config.subset_id == "med_qa_tw_zh":
            paths = {
                "train": os.path.join(
                    base_dir, "Taiwan", "tw_translated_jsonl", "zh", "train-2zh.jsonl"
                ),
                "test": os.path.join(
                    base_dir, "Taiwan", "tw_translated_jsonl", "zh", "test-2zh.jsonl"
                ),
                "valid": os.path.join(
                    base_dir, "Taiwan", "tw_translated_jsonl", "zh", "dev-2zh.jsonl"
                ),
            }
        elif self.config.subset_id == "med_qa_en_4options":
            paths = {
                "train": os.path.join(
                    base_dir, "US", "4_options", "phrases_no_exclude_train.jsonl"
                ),
                "test": os.path.join(
                    base_dir, "US", "4_options", "phrases_no_exclude_test.jsonl"
                ),
                "valid": os.path.join(
                    base_dir, "US", "4_options", "phrases_no_exclude_dev.jsonl"
                ),
            }
        elif self.config.subset_id == "med_qa_zh_4options":
            paths = {
                "train": os.path.join(
                    base_dir, "Mainland", "4_options", "train.jsonl"
                ),
                "test": os.path.join(
                    base_dir, "Mainland", "4_options", "test.jsonl"
                ),
                "valid": os.path.join(
                    base_dir, "Mainland", "4_options", "dev.jsonl"
                ),
            }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": paths["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": paths["test"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": paths["valid"],
                },
            ),
        ]

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        print(filepath)
        data = pd.read_json(filepath, lines=True)

        if self.config.schema == "source":
            for key, example in data.iterrows():
                example = example.to_dict()
                example["options"] = [
                    {"key": key, "value": value}
                    for key, value in example["options"].items()
                ]
                yield key, example

        elif self.config.schema == "bigbio_qa":
            for key, example in data.iterrows():
                example = example.to_dict()
                example_ = {}
                example_["id"] = key
                example_["question_id"] = key
                example_["document_id"] = key
                example_["question"] = example["question"]
                example_["type"] = "multiple_choice"
                example_["choices"] = [value for value in example["options"].values()]
                example_["context"] = ""
                example_["answer"] = [example["answer"]]
                yield key, example_


class MedQAExamplarSampler:
    def __init__(self, seed=42):
        self.seed = seed
        random.seed(seed)
        
    def load_dataset(self):
        """Load MedQA dataset using existing MedQADataset class"""
        # Initialize the dataset builder with default config
        builder = MedQADataset(
            datasets.builder.BuilderConfig(
                name="med_qa_en_bigbio_qa",
                version=datasets.Version("1.0.0")
            )
        )
        data_dir = "../data/med_qa/data_clean/questions"
        
        # Get the training split
        train_examples = []
        for key, example in builder._generate_examples(
            filepath=os.path.join(data_dir, "US", "train.jsonl")
        ):
            train_examples.append(example)
        
        test_examples = []
        for key, example in builder._generate_examples(
            filepath=os.path.join(data_dir, "US", "test.jsonl")
        ):
            test_examples.append(example)
            
        return {"train": train_examples, "test": test_examples}
    
    def sample_examples(self, examples, train_num=1000, test_num=100):
        """
        Sample n examples from dataset
        Args:
            examples: List of dataset examples
            n_samples: Number of examples to sample
        Returns:
            list: Sampled examples
        """
        # Sample examples
        train_set = random.sample(examples["train"], train_num)
        test_set = random.sample(examples["test"], test_num)
            
        return train_set, test_set
    
    def add_cot_template(self, examples):
        """
        Add Chain-of-Thought template to examples
        Args:
            examples: List of sampled examples
        Returns:
            list: Examples with CoT template
        """
        cot_examples = []
        idx = 0
        for example in examples:
            # Create CoT template
            cot_template = {
                'id': idx,
                'messages': [
                    {
                        'role': 'user',
                        'content': self._format_question(example)
                    },
                    {
                        'role': 'assistant',
                        'content': self._format_cot_answer(example)
                    }
                ]
            }
            cot_examples.append(cot_template)
            idx += 1
        return cot_examples
    
    def _format_question(self, example):
        """Format question with choices"""
        question = example['question']
        choices = example['options']
        
        # Format choices as A, B, C, etc.
        choice_text = "\n".join([
            f"{chr(65+i)}. {choice}" 
            for i, choice in enumerate(choices)
        ])
        
        return (
            f"Question: {question}\n\n"
            f"Choices:\n{choice_text}\n\n"
            f"Please solve this step by step, explaining your medical reasoning."
        )
    
    def save_examples(self, examples, output_file):
        """Save examples to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(examples)} examples to {output_file}")
