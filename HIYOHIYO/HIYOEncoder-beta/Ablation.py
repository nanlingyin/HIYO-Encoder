import logging
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
    # 配置日志记录（可选）
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    retriever = Retriever(
        api_key="sk-wF4dimBG8fJ7XrLqdrqMtqi7dVP94W1y2jqHBFF50mgH6bJ6",  # 请替换为您的实际API密钥
        base_url="https://api.ephone.chat/v1",
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        dataset_path="new_dataset-TQA.parquet"  # 请确保数据集路径正确
    )
    evaluator = Evaluator()
    print(f"Number of queries: {len(retriever.queries)}")
    
    queries = retriever.queries
    answers = retriever.answers

    total_f1_query = 0
    total_em_query = 0
    total_f1_group = 0
    total_em_group = 0
    count = 0

    num_tests = 100  # 设置测试次数

    for _ in range(num_tests):
        idx = random.randint(0, len(queries) - 1)
        query = queries[idx]
        answer_list = answers[idx]

        # 提取 ground_truth
        ground_truth = "none"  # 默认值
        if answer_list and isinstance(answer_list, list) and len(answer_list) > 0:
            first_answer = answer_list[0]
            if isinstance(first_answer, dict):
                ground_truth = first_answer.get('text', 'none').strip()
            elif isinstance(first_answer, str):
                ground_truth = first_answer.strip()
            else:
                logging.warning(f"样本索引 {idx} 的 'answers' 格式不正确：{first_answer}")

        # **实验1**：使用 query_embedding
        top_contexts_query = retriever.get_top_contexts_using_query_embedding(query)
        input_text_query = f'''
        The question you get is:
        {query}
        Here are some documents that may help you answer this question. You can choose one of them to answer.
        Documents:
        {top_contexts_query}
        The most important is:
        Your answer does not need to contain a complete grammatical structure, it just needs to answer my question accurately. The answer must be one word or a phrase.
        You'd better think carefully to generate your answer as correctly as possible
        '''
        try:
            response_query = retriever.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": input_text_query},
                ],
                temperature=0.1
            )
            prediction_query = response_query.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"调用 OpenAI API 时出错（query_embedding）：{e}")
            continue

        # 计算得分
        f1_query = evaluator.f1_score(prediction_query, ground_truth)
        em_query = evaluator.exact_match_score(prediction_query, ground_truth)
        total_f1_query += f1_query
        total_em_query += em_query

        # **实验2**：使用 pooled_group_embedding
        top_contexts_group = retriever.get_top_contexts_using_group_embedding(query)
        input_text_group = f'''
        The question you get is:
        {query}
        Here are some documents that may help you answer this question. You can choose one of them to answer.
        Documents:
        {top_contexts_group}
        The most important is:
        Your answer does not need to contain a complete grammatical structure, it just needs to answer my question accurately. The answer must be one word or a phrase.
        You'd better think carefully to generate your answer as correctly as possible
        '''
        try:
            response_group = retriever.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": input_text_group},
                ],
                temperature=0.1
            )
            prediction_group = response_group.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"调用 OpenAI API 时出错（group_embedding）：{e}")
            continue

        # 计算得分
        f1_group = evaluator.f1_score(prediction_group, ground_truth)
        em_group = evaluator.exact_match_score(prediction_group, ground_truth)
        total_f1_group += f1_group
        total_em_group += em_group

        count += 1

        # 输出测试结果
        print(f"Test #{count}")
        print(f"Query: {query}")
        print("Using query_embedding:")
        print(f"Prediction: {prediction_query}")
        print(f"F1 Score: {f1_query}")
        print(f"Exact Match Score: {em_query}")
        print("Using pooled_group_embedding:")
        print(f"Prediction: {prediction_group}")
        print(f"F1 Score: {f1_group}")
        print(f"Exact Match Score: {em_group}")
        print(f"Ground Truth: {ground_truth}")
        print("-" * 50)
        print(f"Average F1 Score (query_embedding): {total_f1_query / count:.4f}")
        print(f"Average EM Score (query_embedding): {total_em_query / count:.4f}")
        print(f"Average F1 Score (group_embedding): {total_f1_group / count:.4f}")
        print(f"Average EM Score (group_embedding): {total_em_group / count:.4f}")
        print("=" * 50)

    # 输出最终平均得分
    print(f"Final Average F1 Score (query_embedding): {total_f1_query / count:.4f}")
    print(f"Final Average EM Score (query_embedding): {total_em_query / count:.4f}")
    print(f"Final Average F1 Score (group_embedding): {total_f1_group / count:.4f}")
    print(f"Final Average EM Score (group_embedding): {total_em_group / count:.4f}")

if __name__ == "__main__":
    main()