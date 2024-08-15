import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.metrics import MeanSquaredError as MSE, BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping


df = pd.read_csv("/helper/generated_data.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df[['food_rating', 'delivery_rating', 'approval']], test_size=0.2, random_state=42
)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_data(texts, tokenizer, max_length=128):
    return tokenizer(
        texts.tolist(),
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )

train_encodings = tokenize_data(X_train, tokenizer)
test_encodings = tokenize_data(X_test, tokenizer)
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
input_ids = Input(shape=(128,), dtype='int32', name='input_ids')
attention_mask = Input(shape=(128,), dtype='int32', name='attention_mask')
bert_output = bert_model(input_ids, attention_mask=attention_mask)[1]
shared_dense = Dense(256, activation='relu')(bert_output)
shared_dropout = Dropout(0.2)(shared_dense)

# Task-specific layers
food_output = Dense(1, activation='linear', name='food_rating')(shared_dropout)
delivery_output = Dense(1, activation='linear', name='delivery_rating')(shared_dropout)
approval_output = Dense(1, activation='sigmoid', name='approval')(shared_dropout)

model = Model(inputs=[input_ids, attention_mask], outputs=[food_output, delivery_output, approval_output])

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss={
        'food_rating': MeanSquaredError(),
        'delivery_rating': MeanSquaredError(),
        'approval': BinaryCrossentropy(),
    },
    metrics={
        'food_rating': MSE(),
        'delivery_rating': MSE(),
        'approval': BinaryAccuracy(),
    }
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1)

model.fit(
    [train_encodings['input_ids'], train_encodings['attention_mask']],
    {
        'food_rating': y_train['food_rating'],
        'delivery_rating': y_train['delivery_rating'],
        'approval': y_train['approval']
    },
    epochs=10,
    batch_size=2,
    validation_split=0.1,
    callbacks=[early_stopping]
)

evaluation_results = model.evaluate(
    [test_encodings['input_ids'], test_encodings['attention_mask']],
    {
        'food_rating': y_test['food_rating'],
        'delivery_rating': y_test['delivery_rating'],
        'approval': y_test['approval']
    }
)

print(f"Evaluation Results: {evaluation_results}")
loss, food_mse, delivery_mse, approval_loss, food_metric, delivery_metric, approval_metric = evaluation_results

print(f"Overall Loss: {loss:.4f}")
print(f"Food Rating MSE: {food_mse:.4f}")
print(f"Delivery Rating MSE: {delivery_mse:.4f}")
print(f"Approval Loss: {approval_loss:.4f}")
print(f"Approval Accuracy: {approval_metric:.4f}")


predictions = model.predict([test_encodings['input_ids'], test_encodings['attention_mask']])
predicted_food_ratings = predictions[0]
predicted_delivery_ratings = predictions[1]
predicted_approvals = predictions[2]
print("Predicted Food Ratings:", predicted_food_ratings.flatten())
print("Predicted Delivery Ratings:", predicted_delivery_ratings.flatten())
print("Predicted Approvals:", predicted_approvals.flatten())
model.save('multi_task_model.h5')


"""Food Rating MSE: 0.2536
Delivery Rating MSE: 0.3432
Approval Loss: 0.6998
Approval Accuracy: 0.4583"""