import re
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import util
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY)
    api_key="sk-kwW1BsPpbbLfLDFBNmByeYYjIBqnACvv6TyuL8IQN7FRnUID",
    base_url="https://api.ephone.chat/v1"
)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):import re
import string
from collections import Counter

# 数据预处理函数
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punctuation(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

# F1 Score 计算函数
def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

dataset = load_dataset("squad_v2", "default")
queries = []
answers = []
contexts = []
for idx, example in enumerate(dataset["train"]):  # 遍历训练集中的所有样本
    context = example.get("question", "").strip()  # 提取 'query' 作为上下文
    context3 = example.get("context", "").strip()
    answerx = example.get("answers", "")
    answer = answerx.get("text", [])
    if context:  # 确保不为空
        queries.append(context)
        answers.append(answer)
        contexts.append(context3)
count = 0
f1sc = []
total = 0
context_embeddings = model.encode(contexts)

# 构建 FAISS 索引
dimension = context_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(context_embeddings))

for i in range(len(queries)):
    query = queries[i]
    answer = answers[i][0]
    print(query)
    query_embedding = model.encode(query)
    k = 1
    distances, indices = index.search(np.array([query_embedding]), k)
    retrieved_texts = [contexts[i] for i in indices[0]]
    print("最相似的上下文: ", retrieved_texts, "\n")
    print("最相似的answer: ", answer, "\n")
    input_text = f"Context: {retrieved_texts}\nQuestion: {query}\nprompt:Your answer only needs to contain the answer to my question, and you don't need to include the reason or redundant information in the context that has nothing to do with the question. Try to be as concise and accurate as possible."
    re1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": input_text},
        ]
    )
    ans = re1.choices[0].message.content
    print(ans, "\n")
    predictions = ans
    ground_truths = answer
    results = f1_score(predictions, ground_truths)
    total += results
    count += 1
    print(f"F1 Score: {results}\n")
    print(f"average F1 Score: {total / count}\n")