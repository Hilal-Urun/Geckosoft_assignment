import joblib
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

food_model = joblib.load("best_food_model.pkl")
delivery_model = joblib.load("best_delivery_model.pkl")
approval_model = joblib.load("best_approval_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

multi_task_model = load_model("multi_task_model.h5", custom_objects={'TFBertModel': TFBertModel})
