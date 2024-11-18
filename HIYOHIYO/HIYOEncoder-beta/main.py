from retrival import Retriever
from evaluate import Evaluator
import random
import string
import re
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import util
import random

def main():
    retriever = Retriever(
        api_key="sk-kwW1BsPpbbLfLDFBNmByeYYjIBqnACvv6TyuL8IQN7FRnUID",
        base_url="https://api.ephone.chat/v1",
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        dataset_path="new_dataset.parquet" # 通过改变这里改变检索的数据集
    )
    evaluator = Evaluator()
    print(f"Number of queries: {len(retriever.queries)}")
    print(f"Number of contexts: {len(retriever.contexts)}")
    print(f"Number of quesgroups: {len(retriever.quesgroups)}")
    print(f"Number of datagroup: {len(retriever.datagroup)}")
    print(f"Number of context embeddings: {len(retriever.context_embeddings)}")
    assert len(retriever.queries) == len(retriever.contexts) == len(retriever.quesgroups) == len(retriever.datagroup) == len(retriever.context_embeddings), "Lengths of queries, contexts, quesgroups, datagroup, and context_embeddings must be equal."

    queries = retriever.queries
    answers = retriever.answers

    total_f1 = 0
    total_em = 0
    count = 0

    # 设置测试次数
    num_tests = 120

    for _ in range(num_tests):
        # 随机选择一个索引
        idx = random.randint(0, len(queries) - 5)
        query = queries[idx]
        answer_list = answers[idx]
        if answer_list:
            ground_truth = answer_list[0]
        else:
            ground_truth = "none"

        # 获取检索到的上下文
        top_contexts_str = retriever.get_top_contexts(query)
        # 构建输入文本
        input_text = f'''
        The question you get is:
        {query}
        Here are some documents that may help you answer this question. You can choose one of them to answer.
        Documents:
        {top_contexts_str}
        The most important is:
        Your answer does not need to contain a complete grammatical structure; it just needs to answer my question accurately. The answer must be one word or a phrase.
        If you can't get an answer to the question from the documents, only in this time you output "none"
        '''

        response = retriever.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": input_text},
            ],
            temperature=0.9
        )
        prediction = response.choices[0].message.content.strip()

        # 计算F1分数和EM分数
        f1 = evaluator.f1_score(prediction, ground_truth)
        em = evaluator.exact_match_score(prediction, ground_truth)

        total_f1 += f1
        total_em += em
        count += 1

        print(f"Query: {query}")
        print(f"Prediction: {prediction}")
        print(f"Ground Truth: {ground_truth}")
        print(f"F1 Score: {f1}")
        print(f"Exact Match Score: {em}")
        print("-" * 50)

    # 输出平均分数
    print(f"Average F1 Score: {total_f1 / count}")
    print(f"Average Exact Match Score: {total_em / count}")

if __name__ == "__main__":
    main()