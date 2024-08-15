import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, accuracy_score

df = pd.read_csv("/helper/generated_data.csv")
food_models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(),
    "SVR": SVR(),
}

food_params = {
    "LinearRegression": {},
    "RandomForest": {"n_estimators": [10, 50, 100]},
    "SVR": {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10]},
}

delivery_models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(),
    "SVR": SVR(),
}

delivery_params = {
    "LinearRegression": {},
    "RandomForest": {"n_estimators": [10, 50, 100]},
    "SVR": {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10]},
}

approval_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "SVC": SVC(),
}

approval_params = {
    "LogisticRegression": {"C": [0.1, 1, 10]},
    "RandomForest": {"n_estimators": [10, 50, 100]},
    "SVC": {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10]},
}


def grid_search(models, params, X_train, y_train):
    best_models = {}
    for name, model in models.items():
        grid = GridSearchCV(model, params[name], cv=3, scoring='neg_mean_squared_error' if isinstance(model, (
            LinearRegression, RandomForestRegressor, SVR)) else 'accuracy')
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
    return best_models


def evaluate_models(models, X_test, y_test, task_type='regression'):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        if task_type == 'regression':
            score = mean_squared_error(y_test, y_pred, squared=False)
        else:
            score = accuracy_score(y_test, y_pred)
        results[name] = score
    return results


def choosing_best_models(df):
    X_train, X_test, y_train_food, y_test_food = train_test_split(
        df['review'], df['food_rating'], test_size=0.2, random_state=42
    )
    _, _, y_train_delivery, y_test_delivery = train_test_split(
        df['review'], df['delivery_rating'], test_size=0.2, random_state=42
    )
    _, _, y_train_approval, y_test_approval = train_test_split(
        df['review'], df['approval'], test_size=0.2, random_state=42
    )
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    best_food_models = grid_search(food_models, food_params, X_train_tfidf, y_train_food)
    best_delivery_models = grid_search(delivery_models, delivery_params, X_train_tfidf, y_train_delivery)
    best_approval_models = grid_search(approval_models, approval_params, X_train_tfidf, y_train_approval)

    food_results = evaluate_models(best_food_models, X_test_tfidf, y_test_food, task_type='regression')
    delivery_results = evaluate_models(best_delivery_models, X_test_tfidf, y_test_delivery, task_type='regression')
    approval_results = evaluate_models(best_approval_models, X_test_tfidf, y_test_approval, task_type='classification')

    best_food_model = min(food_results, key=food_results.get)
    best_delivery_model = min(delivery_results, key=delivery_results.get)
    best_approval_model = max(approval_results, key=approval_results.get)
    return best_food_model, food_results[best_food_model], best_delivery_model, delivery_results[
        best_delivery_model], best_approval_model, approval_results[best_approval_model]


best_food_model, food_result, best_delivery_model, delivery_result, best_approval_model, approval_results = choosing_best_models(
    df)

print(f"Best Food Rating Model: {best_food_model} with RMSE: {food_result}")
print(f"Best Delivery Rating Model: {best_delivery_model} with RMSE: {delivery_result}")
print(f"Best Approval Model: {best_approval_model} with Accuracy: {approval_results}")

"""
Best Food Rating Model: LinearRegression with RMSE: 0.59
Best Delivery Rating Model: LinearRegression with RMSE: 0.54
Best Approval Model: LogisticRegression with Accuracy: 0.58"""


def final_models(df):
    X_train, X_test, y_train_food, y_test_food = train_test_split(df['review'], df['food_rating'], test_size=0.2,
                                                                  random_state=42)
    _, _, y_train_delivery, y_test_delivery = train_test_split(df['review'], df['delivery_rating'], test_size=0.2,
                                                               random_state=42)
    _, _, y_train_approval, y_test_approval = train_test_split(df['review'], df['approval'], test_size=0.2,
                                                               random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    food_model = LinearRegression()
    food_model.fit(X_train_tfidf, y_train_food)
    y_pred_food = food_model.predict(X_test_tfidf)
    food_rmse = mean_squared_error(y_test_food, y_pred_food, squared=False)

    delivery_model = LinearRegression()
    delivery_model.fit(X_train_tfidf, y_train_delivery)
    y_pred_delivery = delivery_model.predict(X_test_tfidf)
    delivery_rmse = mean_squared_error(y_test_delivery, y_pred_delivery, squared=False)

    approval_model = LogisticRegression()
    approval_model.fit(X_train_tfidf, y_train_approval)
    y_pred_approval = approval_model.predict(X_test_tfidf)
    approval_accuracy = accuracy_score(y_test_approval, y_pred_approval)

    print(f"Food Rating RMSE: {food_rmse:.2f}")
    print(f"Delivery Rating RMSE: {delivery_rmse:.2f}")
    print(f"Approval Accuracy: {approval_accuracy:.2f}")

    joblib.dump(food_model, 'best_food_model.pkl')
    joblib.dump(delivery_model, 'best_delivery_model.pkl')
    joblib.dump(approval_model, 'best_approval_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

final_models(df)