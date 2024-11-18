# HIYO-Encoder
This is the English edition of readme
## Table of Contents
- Introduction
- Architecture
- Installation
- Configuration
- Usage
  - Data Preprocessing
  - Retrieval and Evaluation
- Project Structure
- Dependencies
- Example
- Evaluation Metrics
- Contact

## Introduction

The **HIYO-Encoder** is a dual-retrieval model designed to increase retrieval accuracy and lower hallucinations. HIYO-Encoder increases retrieval scope by extending searches using a 5W1H framework (who, what, where, when, why, and how), thereby better aligning retrieved information with user intent. In high-precision, low-hallucinity QA settings, experimental data show that HIYO-Encoder outperforms conventional RAG models in F1 and Exact Match measures. HIYO-Encoder is presented in this work as a strong solution for RAG system optimization in challenging environments.


## Architecture

The HIYO-Encoder project is structured into several key components:

1. **Data Preprocessing (`preprocessing.py`):** Processes the raw dataset to generate augmented data with generating 5W1H sub-queries.
2. **Dual Retrieval  (`retrival.py`):** Handles embedding generation and FAISS indexing for efficient retrieval of relevant contexts.
3. **Evaluation (`evaluate.py`):** Computes F1 and Exact Match scores to evaluate the accuracy of generated answers.
4. **Main Execution (`main.py`):** Integrates retrieval and evaluation processes to generate and assess answers.

## Installation

### Prerequisites

- Python 3.7 or higher
- Git

### Steps

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

## Configuration

1. **OpenAI API Key**

   Replace the placeholder API key in `preprocessing.py` and `retrival.py` with your actual OpenAI API key.

   ```python
   # Example in retrival.py
   self.client = OpenAI(
       api_key="your_actual_api_key",
       base_url="https://api.ephone.chat/v1"
   )
   ```

2. **Dataset Path**

   Ensure the `dataset_path` parameter points to your dataset file. Modify it in `main.py` and other relevant scripts as needed.

   ```python
   retriever = Retriever(
       api_key="your_api_key",
       base_url="https://api.ephone.chat/v1",
       model_name='sentence-transformers/all-MiniLM-L6-v2',
       dataset_path="path/to/your_dataset.parquet"
   )
   ```

## Usage

### Data Preprocessing

Before performing retrieval and evaluation, preprocess your dataset to generate augmented data.

```bash
python preprocessing.py
```


### Retrieval and Evaluation

Execute the main script to perform context retrieval, answer generation, and evaluation.

```bash
python main.py
```


## Project Structure

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

- **main.py:** Integrates retrieval and evaluation processes.
- **retrival.py:** Contains the `Retriever` class for dual retrieval.
- **evaluate.py:** Contains the `Evaluator` class for performance evaluation.
- **preprocessing.py:** Handles data augmentation and preprocessing (5W1H-generation).
- **requirements.txt:** Lists all Python dependencies.
- **README.md:** Project documentation.

## Dependencies

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

## Example

### Sample Output

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

### Code Snippet

```python
# main.py snippet
print(f"Average F1 Score: {total_f1 / count}")
print(f"Average Exact Match Score: {total_em / count}")
```

## Evaluation Metrics

### F1 Score

The F1 Score is the harmonic mean of precision and recall, providing a measure of a test's accuracy. It ranges from 0 to 1, where 1 signifies perfect precision and recall.

```python
f1 = evaluator.f1_score(prediction, ground_truth)
```

### Exact Match (EM) Score

The Exact Match score measures the percentage of predictions that match any one of the ground truth answers exactly.

```python
em = evaluator.exact_match_score(prediction, ground_truth)
```

## Contact

For any inquiries or issues, please contact:

- **Email:** 20241008398@stu.shzu.edu.cn
- **GitHub:** [nanlingyin](https://github.com/nanlingyin)

---
