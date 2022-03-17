from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from typing import Optional

import uvicorn

from celery_worker import create_order, predict, celery
from model import Order, Data

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define what the app does
@app.get("/greet")
async def index(fname: Optional[str] = None, lname: Optional[str] = None):

    if not fname and not lname:
        # If both first name and last name are missing, return an error
        response = { "status" : "error" }
    elif fname and not lname:
        # If first name is present but last name is missing
        response = { "data" : f"Hello, {fname} !" }
    elif not fname and lname:
        # If first name is missing but last name is present
        response = { "data" : f"Hello, Mr. {lname} !" }
    else :
        # if none of the above is true, then both names must be present
        response = { "data" : f"Is your name {fname} {lname} ?" }

    return jsonable_encoder(response)

@app.post('/order')
def add_order(order: Order):
    # use delay() method to call the celery task
    create_order.delay(order.customer_name, order.order_quantity)
    return {"message": "Order Received! Thank you for your patience."}
    
@app.post('/predict')
def add_predict(data: Data):
    task = predict.delay(data.acc_x, data.acc_y, data.acc_z)
    return {"message": "Prediction started", "task_id": task.task_id}

@app.get("/tasks/<task_id>")
def get_status(task_id):
    task_result = celery.AsyncResult(task_id)
    result = {
        "task_id": task_id,
        "task_status": task_result.status,
        "task_result": task_result.result
    }
    return result

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0')
