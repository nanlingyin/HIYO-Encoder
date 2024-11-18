class Evaluator:
    @staticmethod
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

    @staticmethod
    def f1_score(prediction, ground_truth):
        prediction_tokens = Evaluater.normalize_answer(prediction).split()
        ground_truth_tokens = Evaluater.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def exact_match_score(prediction, ground_truth):
        return int(Evaluater.normalize_answer(prediction) == Evaluater.normalize_answer(ground_truth))
'''
使用方法：
from HIYO-Evaluate import Evaluater

# 示例预测和真实答案
prediction = "The advantages of AI medicine"
ground_truth = "The advantages of AI medicine."

# 计算 F1 分数
f1 = Evaluater.f1_score(prediction, ground_truth)
print(f"F1 Score: {f1}")

# 计算精确匹配分数
em = Evaluater.exact_match_score(prediction, ground_truth)
print(f"Exact Match Score: {em}")
'''