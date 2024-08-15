# models/prediction.py
from models.model_loader import food_model, delivery_model, approval_model, vectorizer, multi_task_model
from utils.tokenization import tokenize_review


def predict_separate_models(review: str):
    sample_review_tfidf = vectorizer.transform([review])
    food_rating = food_model.predict(sample_review_tfidf)
    delivery_rating = delivery_model.predict(sample_review_tfidf)
    approval = approval_model.predict(sample_review_tfidf)

    return {
        "food_rating": int(food_rating[0]),
        "delivery_rating": int(delivery_rating[0]),
        "approval": int(approval[0])
    }


def predict_multi_task(review: str):
    encoded_review = tokenize_review(review)
    input_ids = encoded_review['input_ids']
    attention_mask = encoded_review['attention_mask']

    predictions = multi_task_model.predict([input_ids, attention_mask])
    food_rating = predictions[0].flatten()[0]
    delivery_rating = predictions[1].flatten()[0]
    approval = predictions[2].flatten()[0]

    return {
        "food_rating": int(food_rating),
        "delivery_rating": int(delivery_rating),
        "approval": int(approval)
    }
