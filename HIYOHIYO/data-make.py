import openai
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# 设置 OpenAI API 密钥
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY)
    api_key="sk-kwW1BsPpbbLfLDFBNmByeYYjIBqnACvv6TyuL8IQN7FRnUID",
    base_url="https://api.ephone.chat/v1"
)

# 加载 SQuAD v2 数据集
dataset = load_dataset("squad_v2", "default")


# 函数：使用 GPT-4 生成问题
def context_extension(context):
    prompt = f"""
    Context information is below.
    ---------------------
    {context}
    ---------------------
    You are a Professor who is proficient in asking questions. Your task is to set up 6 questions for students to answer based on the provided context.
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


# 函数：处理每个样本，生成新问题并扩展数据集
def process_sample(sample):
    context = sample['context']
    generated_questions = context_extension(context)

    new_samples = []

    # 添加原始问题
    new_samples.append({
        'context': context,
        'question': sample['question'],
        'answers': sample['answers'],
        'id': sample['id']
    })

    # 添加生成的问题
    for i, q in enumerate(generated_questions):
        new_samples.append({
            'context': context,
            'question': q,
            'answers': {'text': [], 'answer_start': []},
            'id': f"{sample['id']}_gen_{i}"
        })

    return new_samples


# 处理整个数据集
new_data = []
for sample in tqdm(dataset['train']):
    new_data.extend(process_sample(sample))

# 创建新的 Dataset 对象
new_dataset = Dataset.from_pandas(pd.DataFrame(new_data))

# 指定保存路径
save_path = 'C:/Users/admin/Desktop/HIYOHIYO'

# 保存为 Parquet 文件
new_dataset.to_parquet(save_path)

print(f"SQuAD extension database has been created and saved as '{save_path}'")
