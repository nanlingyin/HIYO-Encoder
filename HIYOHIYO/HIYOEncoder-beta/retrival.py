import re
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss
from datasets import load_dataset
import random
import string
from collections import Counter

class Retriever:
    def __init__(self, api_key, base_url, model_name='sentence-transformers/all-MiniLM-L6-v2', dataset_path="new_dataset.parquet"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = SentenceTransformer(model_name)
        self.dataset = load_dataset("parquet", data_files=dataset_path) #通过改变这里改变检索的数据集
        self.queries, self.answers, self.contexts, self.quesgroups = self._process_dataset()
        self.datagroup, self.context_embeddings = self._encode_embeddings()
        self.index = self._build_faiss_index()
    
    def _process_dataset(self):#对数据集进行预处理
        queries, answers, contexts, quesgroups = [], [], [], []
        seen_contexts = set()
        for example in self.dataset["train"]:
            context = example.get("question", "").strip()
            context3 = example.get("context", "").strip()
            answerx = example.get("answers", "")
            answer = answerx.get("text", [])
            quesgroup = example.get("textqueries", "")
            if context:
                queries.append(context)
                answers.append(answer)
                quesgroups.append(quesgroup)
            if context3 and context3 not in seen_contexts:
                contexts.append(context3)
                seen_contexts.add(context3)
        return queries, answers, contexts, quesgroups
    
    def _encode_embeddings(self):
        datagroup = []
        for group in self.quesgroups:
            embeddings = self.model.encode(group)
            pooled_embedding = np.mean(embeddings, axis=0)
            datagroup.append(pooled_embedding)
        context_embeddings = self.model.encode(self.contexts)
        return np.array(datagroup), np.array(context_embeddings)
    
    def _build_faiss_index(self):
        dimension = self.datagroup.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(self.datagroup)
        return index
    
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
        # 提取主题
        topic = self.standardization(query)
        
        # 生成问题组
        response_text = eval(self.extension(topic))
        querygroup = response_text
        
        # 编码问题组
        group_embeddings = self.model.encode(querygroup)
        pooled_group_embedding = np.mean(group_embeddings, axis=0)
        
        # 检索最相似的上下文
        k = 20
        distances, indices = self.index.search(np.array([pooled_group_embedding]), k)
        
        retrieved_texts = [self.contexts[i] for i in indices[0]]
        retrieved_embeddings = [self.context_embeddings[i] for i in indices[0]]
        
        # 选择前三个唯一的上下文，可以通过改变下面的数字来改变返回的上下文数量
        top_contexts = []
        seen_contexts = set()
        for idx in indices[0]:
            context = self.contexts[idx]
            if context not in seen_contexts:
                top_contexts.append(context)
                seen_contexts.add(context)
            if len(top_contexts) == 3:
                break
        
        top_contexts_str = "\nDocument: ".join(top_contexts)
        return top_contexts_str