---
annotations_creators:
- crowdsourced
language_creators:
- crowdsourced
language:
- en
license:
- cc-by-sa-4.0
multilinguality:
- monolingual
size_categories:
- 100K<n<1M
source_datasets:
- original
task_categories:
- question-answering
task_ids:
- open-domain-qa
- extractive-qa
paperswithcode_id: squad
pretty_name: SQuAD2.0
dataset_info:
  config_name: squad_v2
  features:
  - name: id
    dtype: string
  - name: title
    dtype: string
  - name: context
    dtype: string
  - name: question
    dtype: string
  - name: answers
    sequence:
    - name: text
      dtype: string
    - name: answer_start
      dtype: int32
  splits:
  - name: train
    num_bytes: 116732025
    num_examples: 130319
  - name: validation
    num_bytes: 11661091
    num_examples: 11873
  download_size: 17720493
  dataset_size: 128393116
configs:
- config_name: squad_v2
  data_files:
  - split: train
    path: squad_v2/train-*
  - split: validation
    path: squad_v2/validation-*
  default: true
train-eval-index:
- config: squad_v2
  task: question-answering
  task_id: extractive_question_answering
  splits:
    train_split: train
    eval_split: validation
  col_mapping:
    question: question
    context: context
    answers:
      text: text
      answer_start: answer_start
  metrics:
  - type: squad_v2
    name: SQuAD v2
---

# Dataset Card for SQuAD 2.0

## Table of Contents
- [Dataset Card for "squad_v2"](#dataset-card-for-squad_v2)
  - [Table of Contents](#table-of-contents)
  - [Dataset Description](#dataset-description)
    - [Dataset Summary](#dataset-summary)
    - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
    - [Languages](#languages)
  - [Dataset Structure](#dataset-structure)
    - [Data Instances](#data-instances)
      - [squad_v2](#squad_v2)
    - [Data Fields](#data-fields)
      - [squad_v2](#squad_v2-1)
    - [Data Splits](#data-splits)
  - [Dataset Creation](#dataset-creation)
    - [Curation Rationale](#curation-rationale)
    - [Source Data](#source-data)
      - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
      - [Who are the source language producers?](#who-are-the-source-language-producers)
    - [Annotations](#annotations)
      - [Annotation process](#annotation-process)
      - [Who are the annotators?](#who-are-the-annotators)
    - [Personal and Sensitive Information](#personal-and-sensitive-information)
  - [Considerations for Using the Data](#considerations-for-using-the-data)
    - [Social Impact of Dataset](#social-impact-of-dataset)
    - [Discussion of Biases](#discussion-of-biases)
    - [Other Known Limitations](#other-known-limitations)
  - [Additional Information](#additional-information)
    - [Dataset Curators](#dataset-curators)
    - [Licensing Information](#licensing-information)
    - [Citation Information](#citation-information)
    - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://rajpurkar.github.io/SQuAD-explorer/
- **Repository:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
- **Paper:** https://arxiv.org/abs/1806.03822
- **Point of Contact:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Dataset Summary

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

SQuAD 2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers
to look similar to answerable ones. To do well on SQuAD2.0, systems must not only answer questions when possible, but
also determine when no answer is supported by the paragraph and abstain from answering.

### Supported Tasks and Leaderboards

Question Answering.

### Languages

English (`en`).

## Dataset Structure

### Data Instances

#### squad_v2

- **Size of downloaded dataset files:** 46.49 MB
- **Size of the generated dataset:** 128.52 MB
- **Total amount of disk used:** 175.02 MB

An example of 'validation' looks as follows.
```
This example was too long and was cropped:

{
    "answers": {
        "answer_start": [94, 87, 94, 94],
        "text": ["10th and 11th centuries", "in the 10th and 11th centuries", "10th and 11th centuries", "10th and 11th centuries"]
    },
    "context": "\"The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave thei...",
    "id": "56ddde6b9a695914005b9629",
    "question": "When were the Normans in Normandy?",
    "title": "Normans"
}
```

### Data Fields

The data fields are the same among all splits.

#### squad_v2
- `id`: a `string` feature.
- `title`: a `string` feature.
- `context`: a `string` feature.
- `question`: a `string` feature.
- `answers`: a dictionary feature containing:
  - `text`: a `string` feature.
  - `answer_start`: a `int32` feature.

### Data Splits

| name     |  train | validation |
| -------- | -----: | ---------: |
| squad_v2 | 130319 |      11873 |

## Dataset Creation

### Curation Rationale

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Source Data

#### Initial Data Collection and Normalization

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

#### Who are the source language producers?

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Annotations

#### Annotation process

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

#### Who are the annotators?

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Personal and Sensitive Information

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Discussion of Biases

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Other Known Limitations

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Additional Information

### Dataset Curators

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Licensing Information

The dataset is distributed under the CC BY-SA 4.0 license.

### Citation Information

```
@inproceedings{rajpurkar-etal-2018-know,
    title = "Know What You Don{'}t Know: Unanswerable Questions for {SQ}u{AD}",
    author = "Rajpurkar, Pranav  and
      Jia, Robin  and
      Liang, Percy",
    editor = "Gurevych, Iryna  and
      Miyao, Yusuke",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P18-2124",
    doi = "10.18653/v1/P18-2124",
    pages = "784--789",
    eprint={1806.03822},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
@inproceedings{rajpurkar-etal-2016-squad,
    title = "{SQ}u{AD}: 100,000+ Questions for Machine Comprehension of Text",
    author = "Rajpurkar, Pranav  and
      Zhang, Jian  and
      Lopyrev, Konstantin  and
      Liang, Percy",
    editor = "Su, Jian  and
      Duh, Kevin  and
      Carreras, Xavier",
    booktitle = "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2016",
    address = "Austin, Texas",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D16-1264",
    doi = "10.18653/v1/D16-1264",
    pages = "2383--2392",
    eprint={1606.05250},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
}
```


### Contributions

Thanks to [@lewtun](https://github.com/lewtun), [@albertvillanova](https://github.com/albertvillanova), [@patrickvonplaten](https://github.com/patrickvonplaten), [@thomwolf](https://github.com/thomwolf) for adding this dataset.