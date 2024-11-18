import torch
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import load_dataset, load_metric
from typing import List

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载 TriviaQA 数据集
# 这里使用 TriviaQA 的验证集作为示例
dataset = load_dataset("trivia_qa", "unfiltered.natural", split="validation[:100]")  # 取100个样本进行示范

# 加载 RAG 模型和分词器
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)
model = model.to(device)

# 加载评估指标
metric_f1 = load_metric("squad_v2")  # 使用 SQuAD 的 F1 和 EM
# 注意：SQuAD v2 包含是否有答案的判断，但我们这里假设所有问题都有答案

def compute_f1_em(predictions: List[str], references: List[str]):
    formatted_predictions = [{"id": str(i), "prediction_text": pred} for i, pred in enumerate(predictions)]
    formatted_references = [{"id": str(i), "answers": {"text": [ref], "answer_start": [0]}} for i, ref in enumerate(references)]
    
    results = metric_f1.compute(predictions=formatted_predictions, references=formatted_references)
    return results

# 生成回答并收集预测和参考
predictions = []
references = []

batch_size = 8
for i in range(0, len(dataset), batch_size):
    batch = dataset[i:i+batch_size]
    questions = batch["question"]
    answers = batch["answer"]["text"]  # TriviaQA 的答案是一个列表，取第一个

    # 准备输入
    inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # 生成回答
    with torch.no_grad():
        generated = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=5,
            num_return_sequences=1,
            max_length=50
        )
    
    # 解码生成的回答
    decoded_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
    predictions.extend(decoded_preds)
    references.extend([ans[0] for ans in answers])  # 取第一个答案作为参考

    print(f"Processed {min(i + batch_size, len(dataset))} / {len(dataset)}")

# 计算 F1 和 EM
results = compute_f1_em(predictions, references)
print(f"F1 Score: {results['f1']:.2f}")
print(f"Exact Match: {results['exact_match']:.2f}")
