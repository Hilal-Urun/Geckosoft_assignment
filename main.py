import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.prediction import predict_separate_models, predict_multi_task

app = FastAPI()


class ReviewInput(BaseModel):
    review: str


@app.post("/predict-separate/")
async def predict_separate_endpoint(review_input: ReviewInput):
    try:
        prediction = predict_separate_models(review_input.review)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-multi-task/")
async def predict_multi_task_endpoint(review_input: ReviewInput):
    try:
        prediction = predict_multi_task(review_input.review)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
