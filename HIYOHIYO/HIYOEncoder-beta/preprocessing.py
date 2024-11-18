import openai
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# 设置 OpenAI API 客户端
client = OpenAI(
    api_key="sk-kwW1BsPpbbLfLDFBNmByeYYjIBqnACvv6TyuL8IQN7FRnUID",  # 请将此处替换为您的实际 API 密钥
    base_url="https://api.ephone.chat/v1"
)

# 此处选择想加载的数据集
dataset = load_dataset("squad_v2")

# 函数：使用 GPT-4 根据主题生成问题 
def context_extension(context,topic):
    prompt = f"""
    Context information is below.
    ---------------------
    topic : {topic}
    context : {context}
    ---------------------
    You are a Professor who is proficient in asking questions. Your task is to set up 6 questions of my topic for students to answer based on the provided context .
    1. The questions should be diverse in nature and cover various aspects of the document. Ensure that they include the words "who," "where," "what," "when," "why," and "how."
    2. Restrict the questions to the information provided, and avoid ambiguous references.
    Output Format:
    [
    "question",
    "question",
    ]
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
        ]
    )
    questions = response.choices[0].message.content.strip()
    # 将生成的文本转换为问题列表
    questions = questions.strip('[]').replace('"', '').split(',')
    questions = [q.strip() for q in questions if q.strip()]
    return questions

def standardization(query):#主题提取
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
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": prompt},
    ]
)
    return response.choices[0].message.content
# 函数：处理每个样本，生成新问题并扩展数据集
def process_sample(sample):
    context = sample['context']
    topic = standardization(sample['question'])
    generated_questions = context_extension(context,topic)

    new_samples = []
    '''
    # 添加原始问题,该部分仅为测试作用
    new_samples.append({
        'context': context,
        'question': sample['question'],
        'answers': sample['answers'],
        'id': sample['id']
    })
    '''
    # 添加生成的问题
    new_samples.append({
        'context': context,
        'question': sample['question'],
        'textqueries': generated_questions,#将queries作为依据文本生成的问题组
        'questionqueries': "",#暂时不添加根据问题生成的问题组
        'answers': sample['answers'],
        'id': sample['id']
    })

    return new_samples

# 处理前100个样本，可以通过修改此处的数值来处理更多样本
new_data = []
for idx, sample in enumerate(tqdm(dataset['train'])):
    if idx >= 100:
        break
    new_data.extend(process_sample(sample))

# 创建新的 Dataset 对象
new_dataset = Dataset.from_pandas(pd.DataFrame(new_data))

# 指定保存路径
save_path = 'C:/Users/admin/Desktop/new_dataset.parquet'

# 保存为 Parquet 文件
new_dataset.to_parquet(save_path)

print(f"The extension database has been created and saved as '{save_path}'")
