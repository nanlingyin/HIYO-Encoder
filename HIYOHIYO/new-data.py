from openai import OpenAI
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import json
import re

# Set OpenAI API key and base URL
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY)
    api_key="sk-kwW1BsPpbbLfLDFBNmByeYYjIBqnACvv6TyuL8IQN7FRnUID",
    base_url="https://api.ephone.chat/v1"
)

# Load SQuAD v2 dataset
dataset = load_dataset("squad_v2")

# Initialize embedding models
# You can choose appropriate models for context and question embeddings
context_model = SentenceTransformer('all-MiniLM-L6-v2')
question_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function: Use LLM to generate questions
def context_extension(context):
    prompt = f"""
You are a Professor proficient in asking questions.

**Task**: Generate 6 questions for students based on the provided context.

**Context**:
{context}

**Guidelines**:
1. Questions should be diverse and cover various aspects of the context.
2. Include "who," "where," "what," "when," "why," and "how" questions.
3. Restrict questions to the provided information; avoid ambiguous references.

**Output Format**:
Provide exactly 6 questions in a numbered list, one question per line:
1. Question 1
2. Question 2
3. Question 3
4. Question 4
5. Question 5
6. Question 6
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that generates questions based on context."},
            {"role": "user", "content": prompt},
        ]
    )
    output = response.choices[0].message.content.strip()
    # Extract questions from the numbered list
    questions = re.findall(r'^\d+\.\s*(.*)', output, re.MULTILINE)
    if len(questions) < 6:
        # If less than 6 questions are found, attempt alternative parsing
        questions = [line.strip('-* ') for line in output.strip().split('\n') if line.strip()]
    return questions

# Function: Process each sample, generate new questions, and extend dataset
def process_sample(sample):
    context = sample['context']
    generated_questions = context_extension(context)

    new_samples = []

    # Embed the context
    context_embedding = context_model.encode(context)

    # Add original question and embeddings
    question_embedding = question_model.encode(sample['question'])
    new_samples.append({
        'context': context,
        'context_embedding': context_embedding,
        'question': sample['question'],
        'question_embedding': question_embedding,
        'answers': sample['answers'],
        'id': sample['id']
    })

    # Add generated questions and embeddings
    for i, q in enumerate(generated_questions):
        question_embedding = question_model.encode(q)
        new_samples.append({
            'context': context,
            'context_embedding': context_embedding,
            'question': q,
            'question_embedding': question_embedding,
            'answers': {'text': [], 'answer_start': []},
            'id': f"{sample['id']}_gen_{i}"
        })

    return new_samples

# Process the entire dataset
new_data = []
for sample in tqdm(dataset['train']):
    new_samples = process_sample(sample)
    new_data.extend(new_samples)

# Create new Dataset object
df = pd.DataFrame(new_data)
new_dataset = Dataset.from_pandas(df)

# Save path
save_path = 'C:/Users/admin/Desktop/HIYOHIYO/new_dataset.parquet'

# Save as Parquet file
new_dataset.to_parquet(save_path)

print(f"SQuAD extension database has been created and saved as '{save_path}'")