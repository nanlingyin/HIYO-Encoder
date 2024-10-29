# HIYO-Encoder

## Overview

This project aims to create a system that generates questions based on a given context, extends the context with relevant questions, and retrieves the most relevant context to answer a query. It uses a combination of natural language processing (NLP) models and similarity search techniques to achieve this.

## Requirements

Ensure you have the following dependencies installed:

- `re`
- `openai`
- `sentence-transformers`
- `numpy`
- `faiss`
- `datasets`
- `torch`

You can install the necessary Python packages using pip:

```bash
pip install openai sentence-transformers numpy faiss-cpu datasets torch
```

## Components

### 1. **Preprocessing Functions**

- **normalize_answer(s):** Normalizes text by converting it to lowercase, removing punctuation, articles, and extra whitespace.
- **f1_score(prediction, ground_truth):** Computes the F1 score between the predicted answer and the ground truth.

### 2. **OpenAI Client Setup**

The OpenAI client is set up with the provided API key and base URL:

```python
client = OpenAI(
    api_key="your_openai_api_key",
    base_url="your_url"
)
```

### 3. **Context Extension and Question Generation**

- **context_extension(context):** Generates a set of six questions based on the given context using the OpenAI API.
- **extension(topic):** Extends a given topic by generating six questions using the OpenAI API.
- **standardization(query):** Extracts a topic sentence from the given query using the OpenAI API.

### 4. **Dataset Loading**

The SQuAD v2 dataset is loaded and processed to extract queries, answers, and contexts:

```python
dataset = load_dataset("squad_v2", "default")
```

### 5. **Embedding and Indexing**

- **SentenceTransformer:** The `all-MiniLM-L6-v2` model is used to encode the contexts and queries into embeddings.
- **FAISS:** A FAISS index is created to store and search for the context embeddings.

### 6. **Retrieval and Answering**

For each query:
- The topic is extracted.
- Questions are generated based on the topic.
- The generated questions are encoded, and their embeddings are pooled.
- The pooled embedding is used to retrieve the most relevant contexts from the FAISS index.
- The most similar context is used to answer the query using the OpenAI API.

### 7. **Evaluation**

The system evaluates the generated answers by computing the F1 score against the ground truth answers.

## Usage

1. **Set Up OpenAI API Key:**
   Replace `"your_openai_api_key"` with your actual OpenAI API key in the `OpenAI` client setup.

2. **Run the Script:**
   Ensure all dependencies are installed and run the script. It will process the dataset, generate questions, retrieve relevant contexts, and compute F1 scores for the answers.

```bash
python your_script_name.py
```

3. **Output:**
   The script will print:
   - The original query.
   - The extracted topic.
   - The generated questions.
   - The most similar context.
   - The generated answer.
   - The F1 score for each query.
   - The average F1 score.

## Example Output

```plaintext
Query: What is the capital of France?
Topic: The capital of France.
Generated Questions: ["What is the capital of France?", "Where is the capital of France located?", ...]
Most Similar Context: Paris is the capital of France.
Generated Answer: Paris
F1 Score: 1.0
Average F1 Score: 0.95
```

## Notes

- Ensure you have a valid OpenAI API key.
- The script currently processes the entire SQuAD v2 training dataset. You may want to limit the number of examples for quicker testing.
- The F1 score is used to evaluate the accuracy of the generated answers.

