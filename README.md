
# OVERVIEW 
This project is a machine learning-based system designed to predict food and delivery ratings, as well as approval decisions,
based on user reviews. The system leverages both separate models and a multi-task learning model to make predictions.

# CREATION OR CHOOSING OF TRAINING DATASET
- There are several options for preparation of the training data but since I have restricted time I couldn't manage
to generate more so I'm going to explain them in here.
- Although there are open source datasets, which are contains ratings of the restaurants as well as reviews, they
don't specialize ratings as food ratings, delivery ratings etc. Since training dataset has impact on the performance
of the machine learning model and also choosing the correct model for the prediction task, professional human evaluator can apply labeling
according to the reviews.
- On the other hand, crawling can be applied for getting data from google about the locations.
- Another way to use service APIs for collecting the delivery systems information for example GETIR, and create a dataset. The reason collecting information from 
original application  that, also analyzing other features to see if there is any correlation between the features and see the effect of them on each other.
- Finally, nowadays people are using openAI APIs for generating fake datasets, this can be applied for generating training dataset,too.

# THEORETICAL ASPECTS

Separate Models:

- The task of predicting food ratings, delivery ratings, and approval decisions can be seen as distinct tasks, each requiring specialized processing.
By using separate models, each model can focus on optimizing for its specific task.
This approach can used if there are no obvious correlation between prediction goals. 
And for this approach, there are 2 different model types regression for food and delivery ratings and, classification for approval.
- For classical machine learning problems, my general approach is following :
  - First checking the literature to see which models are used.
  - Then having a list of models to train and according to the training accuracy choosing the best one as final model.
  - After deciding the best model, with the best hyperparameters, I train the model at the beginning and with using test,and validation  data see the final performance.
  - Then saving the model for loaded at runtime. 
  
Multi-Task Learning Model:
- Multi-task learning is used when tasks are related, have correlation and can benefit from shared representations. 
- In this case, predicting food ratings, delivery ratings, and approval decisions from the same review text likely involves overlapping linguistic features. However,
the data is generated with randomness, in practice it doesn't seem related.
- There are common pre-trained models for this kind of tasks e.g LSTM, BERT, Multi-Task Deep Neural Networks, T5 etc.
- For this task, I've chosen BERT-based multitask learning model because of the timing, which is capable of predicting all three tasks simultaneously. 
- The model was fine-tuned on the specific tasks and saved as a .h5 file.

# Application Components

1. FastAPI Application (app.py)
The main FastAPI application handles HTTP requests and routes them to the appropriate functions for prediction.
It includes two endpoints:
- /predict-separate/: Uses separate models to make predictions.
- /predict-multi-task/: Uses the multi-task learning model to make predictions.

2. Model Loading (models/model_loader.py)
This module is responsible for loading the models and tokenizer when the application starts. The models include:
- food_model.pkl: Predicts food ratings.
- delivery_model.pkl: Predicts delivery ratings.
- approval_model.pkl: Predicts approval decisions.
- multi_task_model.h5: Predicts all three tasks using a multi-task learning approach.
- tfidf_vectorizer.pkl: A TF-IDF vectorizer for text preprocessing in the separate models.

3. Tokenization (utils/tokenization.py)
This module contains functions for text preprocessing, including tokenization using the BERT tokenizer. 
The tokenized text is then passed to the models for prediction.

4. Prediction Logic (models/prediction.py)
This module contains the core logic for making predictions using either the separate models or the multi-task learning model.


# External Services
1. Pretrained Models
The application uses a pre-trained BERT model from the Hugging Face Transformers library. 
This requires downloading the model when the application first runs.

2. Docker
Docker is required to build and run the application in a containerized environment.
This ensures the application runs consistently across different platforms.

# HOW TO RUN with Python

- pip install -r requirements.txt
- python main.py


# HOW TO RUN with Docker 
- install docker 
- docker build -t fastapi-app .
- docker run -d --name fastapi-container -p 8000:8000 fastapi-app

