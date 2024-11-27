import re
from openai import OpenAI
from sentence_transformers  import SentenceTransformer, util
import numpy as np
from datasets import load_dataset
import random
import string
from collections import Counter
import torch
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, api_key, base_url, model_name='sentence-transformers/all-MiniLM-L6-v2', dataset_path=r"C:\Users\admin\Desktop\HIYO-Encoder\new_dataset-squad.parquet",split='train'):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = SentenceTransformer(model_name)
        self.dataset = load_dataset("parquet", data_files=dataset_path)
        self.queries, self.answers, self.contexts, self.quesgroups = self._process_dataset()
        self.datagroup, self.context_embeddings = self._encode_embeddings()

    def _process_dataset(self):
        queries, answers, contexts, quesgroups = [], [], [], []
        for i, example in enumerate(self.dataset["train"]):
            query = example.get("question", "").strip() 
            context = example.get("context", "").strip() 
            #answerx = example.get("answers", "").strip()
            #answers_list = answerx.get("text", [])
            answerx = example.get("answers", "")
            answers_list = answerx.get("text", [])
            quesgroup = example.get("textqueries", "")
            
            if query:
                queries.append(query)
                answers.append(answers_list)
                quesgroups.append(quesgroup)
            
            
            # 始终添加 context，避免长度不一致
            if context:
                contexts.append(context)
        
        return queries, answers, contexts, quesgroups

    def _encode_embeddings(self):
        datagroup = []
        for group in self.quesgroups:
            if group:
                embeddings = self.model.encode(group)
                pooled_embedding = np.mean(embeddings, axis=0)
            else:
            # 如果没有问题组，使用零向量
                pooled_embedding = np.zeros(self.model.get_sentence_embedding_dimension())
            datagroup.append(pooled_embedding)
    
    # 确保编码所有 contexts，并转换为 float32
        context_embeddings = self.model.encode(self.contexts)
        return np.array(datagroup, dtype=np.float32), np.array(context_embeddings, dtype=np.float32)

    def standardization(self, query):
        prompt = f"""
        The context you get is below
        -------------------------------------------------------
        {query}
        -------------------------------------------------------
        You are a Professor of Semantic recognition. Your task is to extract the topic of the context and what you need to do is below.
        1.You need to extract a topic sentence from the context, remove redundant modifiers and try to retain key elements in the context.
        2.The topic sentence must be a declarative sentence and summarize the context.
        3.The topic sentence should be short without any redundant content.
        4.Avoid the topic sentence being irrelevant to the context or as same as the context.
        the example is below.
        Context: Please talk about the advantages of AI medicine.
        Your output: The advantages of AI medicine.
    
        Output Format:
        [
        "topic sentence"
        ]
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
            ]
        )
        return response.choices[0].message.content

    def extension(self, topic):
        prompt = f"""
        Context information is below.
        ---------------------
        {topic}
        ---------------------
        Given the context information and not prior knowledge, generate only questions based on the below query.
        You are a Professor who is proficient in asking questions. Your task is to setup 6 questions for students to answer.
        1.The questions should be diverse in nature across the document. At least it should include "who, where, what, when, why, how".
        2.Restrict the questions to the information provided, and avoid ambiguous references.
        Output Format:
        [
        "question",
        "question",
        ]
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
            ]
        )
        return response.choices[0].message.content

    def get_top_contexts(self, query):
        query_embedding = self.model.encode(query)
        # 提取主题
        topic = self.standardization(query)
    
        # 生成问题组
        response_text = self.extension(topic)
        querygroup = eval(response_text)
 
    
        # 编码问题组
        group_embeddings = self.model.encode(querygroup)
        pooled_group_embedding = np.mean(group_embeddings, axis=0)
    
        # 计算 pooled_group_embedding 与所有 quesgroups 编码的余弦相似度
        similarities = util.cos_sim(pooled_group_embedding, self.datagroup)[0]
    
        # 获取相似度最高的 20 个索引
        top_k = min(20, len(similarities))
        top_k_values, top_k_indices = torch.topk(similarities, k=top_k)
        top_k_indices = top_k_indices.numpy()
    
        # 获取对应的上下文
        top_k_contexts = [self.contexts[idx] for idx in top_k_indices]
    
        # 对前 20 个上下文进行去重，同时保持索引对应
        unique_contexts = []
        unique_indices = []
        seen_contexts = set()
        for idx, context in zip(top_k_indices, top_k_contexts):
            if context not in seen_contexts:
                seen_contexts.add(context)
                unique_contexts.append(context)
                unique_indices.append(idx)
    
         # 使用去重后的索引获取对应的上下文嵌入
        unique_context_embeddings = self.context_embeddings[unique_indices]
    
        # 计算 query_embedding 与去重后的 context_embeddings 的余弦相似度
        context_similarities = util.cos_sim(query_embedding, unique_context_embeddings)[0]
    
        # 获取相似度最高的 3 个索引
        top_n = min(3, len(context_similarities))
        top_n_values, top_n_indices_in_unique = torch.topk(context_similarities, k=top_n)
        # 映射回原始索引
        top_n_indices = [unique_indices[idx] for idx in top_n_indices_in_unique.numpy()]
    
        # 获取最终的上下文
        top_contexts = [self.contexts[idx] for idx in top_n_indices]
        top_contexts_str = "\nDocument: ".join(top_contexts)
        return top_contexts_str

    def get_top_contexts_using_query_embedding(self, query):
        query_embedding = self.model.encode(query)
        # 计算与所有context_embeddings的余弦相似度
        similarities = util.cos_sim(query_embedding, self.context_embeddings)[0]
        # 获取相似度最高的3个索引
        top_n = min(3, len(similarities))
        top_n_values, top_n_indices = torch.topk(similarities, k=top_n)
        top_n_indices = top_n_indices.numpy()
        # 获取对应的上下文
        top_contexts = [self.contexts[idx] for idx in top_n_indices]
        # 将上下文拼接成字符串
        top_contexts_str = "\nDocument: ".join(top_contexts)
        return top_contexts_str

    def get_top_contexts_using_group_embedding(self, query):
        # 提取主题
        topic = self.standardization(query)
        # 生成问题组
        response_text = self.extension(topic)
        try:
            querygroup = eval(response_text)
        except Exception as e:
            logging.error(f"Failed to parse querygroup: {e}")
            querygroup = []
        # 编码问题组
        if querygroup:
            group_embeddings = self.model.encode(querygroup)
            pooled_group_embedding = np.mean(group_embeddings, axis=0)
        else:
            # 如果生成的问题组为空，使用零向量
            pooled_group_embedding = np.zeros(self.model.get_sentence_embedding_dimension())
        # 计算与所有context_embeddings的余弦相似度
        similarities = util.cos_sim(pooled_group_embedding, self.context_embeddings)[0]
        # 获取相似度最高的3个索引
        top_n = min(3, len(similarities))
        top_n_values, top_n_indices = torch.topk(similarities, k=top_n)
        top_n_indices = top_n_indices.numpy()
        # 获取对应的上下文
        top_contexts = [self.contexts[idx] for idx in top_n_indices]
        # 将上下文拼接成字符串
        top_contexts_str = "\nDocument: ".join(top_contexts)
        return top_contexts_str