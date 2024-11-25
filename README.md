
# HIYO-Encoder
This is the repository for the paper HIYO-Encoder: A Dual-Retrieval Model Based on Expanded Question Generation for Addressing Hallucinations in Question Answering
## Abstract

Language model performance has show promise using retrieval-augmented generation (RAG) techniques. But conventional RAG methods can rely mostly on direct user searches, so neglecting complex semantic links between the query and retrieval text, so reducing retrieval accuracy and raising hallucination risk. To tackle this, we propose **HIYO-Encoder** , a plug-and-play adaptive encoding method leveraging higher query generating to improve retrieval accuracy. By means of large language models (LLMs), HIYO-Encoder generates expanded searches and develops semantic centers for both user queries and retrieval databases, hence enhancing the alignment between the retrieved content and the original query. This twin-search approach lowers hallucinations and maximizes retrieval performance. We evaluate HIYO-Encoder on question-answering challenges to demonstrate its efficacy in significantly reducing hallucination rates and improving retrieval quality. We also conduct ablation studies.

## Keywords

Dual-Retrieval Model, Question Answering Systems, 5W1H Framework, Hallucination Mitigation, RAG Optimization

## 1. Introduction

Question answering (QA) systems play a pivotal role in the field of natural language processing, offering solutions across various applications. However, existing systems often grapple with inaccurate information retrieval and the generation of hallucinated content. To address these challenges, we introduce **HIYO-Encoder**, a dual-retrieval model based on expanded question generation. By leveraging the 5W1H framework, HIYO-Encoder broadens the retrieval scope, thereby improving retrieval accuracy and reducing hallucinations.

## 2. Architecture

The HIYO-Encoder project is organized into several key components:

1. **Data Preprocessing (`preprocessing.py`)**: Processes the raw dataset to generate augmented data with 5W1H sub-queries.
2. **Dual Retrieval (`retrival.py`)**: Manages embedding generation and dual retrieval through cosine similarity for efficient retrieval of relevant contexts.
3. **Evaluation (`evaluate.py`)**: Computes F1 and Exact Match scores to assess the accuracy of generated answers.
4. **Main Execution (`main.py`)**: Integrates retrieval and evaluation processes to generate and evaluate answers.

## 3. Installation

### 3.1 Prerequisites

- Python 3.7 or higher
- Git

### 3.2 Steps

1. **Clone the Repository**

    ```bash
    git clone https://github.com/your-repo/HIYO-Encoder.git
    cd HIYO-Encoder
    ```

2. **Create and Activate a Virtual Environment**

    ```bash
    python -m venv venv

    # Windows
    venv\Scripts\activate

    # macOS/Linux
    source venv/bin/activate
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

## 4. Configuration

### 4.1 OpenAI API Key

Replace the placeholder API key with your actual OpenAI API key in both `preprocessing.py` and `retrival.py`.

```python
# retrival.py example
self.client = OpenAI(
    api_key="your_actual_api_key",
    base_url="https://api.ephone.chat/v1"
)
```

### 4.2 Dataset Path

Ensure the `dataset_path` parameter points to your dataset file. Modify this in `main.py` and other relevant scripts as needed.

```python
retriever = Retriever(
    api_key="your_api_key",
    base_url="https://api.openai.chat/",
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    dataset_path="path/to/your_dataset.parquet"
)
```

## 5. Usage

### 5.1 Data Preprocessing

Preprocess the dataset to generate augmented data before performing retrieval and evaluation.

```bash
python preprocessing.py
```

### 5.2 Retrieval and Evaluation

Execute the main script to perform context retrieval, answer generation, and evaluation.

```bash
python main.py
```

## 6. Project Structure

```
HIYO-Encoder/HIYOHIYO/HIYOEncoder-beta/
├── main.py
├── retrival.py
├── evaluate.py
└── preprocessing.py
HIYO-Encoder/HIYOHIYO/
└── requirements.txt
HIYO-Encoder/
└── README.md
```

- **main.py**: Integrates retrieval and evaluation processes.
- **retrival.py**: Contains the `Retriever` class for dual retrieval.
- **evaluate.py**: Contains the `Evaluator` class for performance evaluation.
- **preprocessing.py**: Handles data augmentation and preprocessing (5W1H generation).
- **requirements.txt**: Lists all Python dependencies.
- **README.md**: Project documentation.

## 7. Dependencies

The HIYO-Encoder project relies on the following Python libraries:

- `openai`
- `sentence-transformers`
- `faiss-cpu`
- `datasets`
- `pandas`
- `tqdm`

Ensure all dependencies are installed via `requirements.txt`.

```plaintext
openai
sentence-transformers
faiss-cpu
datasets
pandas
tqdm
```

## 8. Example

### 8.1 Sample Output

```
Query: Explain the advantages of AI in medicine.
Prediction: AI in medicine offers improved diagnostic accuracy, optimized treatment plans, and enhanced patient care.
Ground Truth: The advantages of AI medicine.
F1 Score: 0.8
Exact Match Score: 0
--------------------------------------------------
Average F1 Score: 0.75
Average Exact Match Score: 0.2
```

### 8.2 Code Snippet

```python
# main.py snippet
print(f"Average F1 Score: {total_f1 / count}")
print(f"Average Exact Match Score: {total_em / count}")
```

## 9. Evaluation Metrics

### 9.1 F1 Score

The F1 score is the harmonic mean of precision and recall, providing a measure of test accuracy. It ranges from 0 to 1, with 1 indicating perfect precision and recall.

```python
f1 = evaluator.f1_score(prediction, ground_truth)
```

### 9.2 Exact Match (EM) Score

The Exact Match score measures the percentage of predictions that exactly match any of the ground truth answers.

```python
em = evaluator.exact_match_score(prediction, ground_truth)
```

## 10. Contact

For any inquiries or issues, please contact:

- **Email**: 20241008398@stu.shzu.edu.cn
- **GitHub**: [nanlingyin](https://github.com/nanlingyin)
- **Personal website**: [https://lynngnan.xyz](https://www.lynngnan.xyz/)

---
