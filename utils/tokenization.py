from models.model_loader import tokenizer


def tokenize_review(review: str):
    return tokenizer(
        [review],
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='tf'
    )
