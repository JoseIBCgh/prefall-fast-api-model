from time import sleep
from celery import Celery
from celery.utils.log import get_task_logger
import os
# Initialize celery
#celery = Celery('tasks', broker='amqp://guest:guest@127.0.0.1:5672//', backend='rpc')
celery = Celery(__name__)
celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "amqp://guest:guest@127.0.0.1:5672//")
celery.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379")
# Create logger - enable to display messages on task logger
celery_log = get_task_logger(__name__)
# Create Order - Run Asynchronously with celery
# Example process of long running task
import pickle
import numpy
with open('model.pkl', 'rb') as fid:
    clf = pickle.load(fid)

@celery.task
def create_order(name, quantity):
    
    # 5 seconds per 1 order
    complete_time_per_item = 5
    
    # Keep increasing depending on item quantity being ordered
    sleep(complete_time_per_item * quantity)
# Display log    
    celery_log.info(f"Order Complete!")
    return {"message": f"Hi {name}, Your order has completed!",
            "order_quantity": quantity}

@celery.task
def predict(acc_x, acc_y, acc_z):
    array = [acc_x, acc_y, acc_z]
    array = numpy.array(array)
    array = array.transpose((1,0))
    prediction = clf.predict_proba(array)
    average = numpy.average(prediction, axis=0)
    celery_log.info(f"Prediction Complete!")
    return {"message": "Prediction complete", "prediction": list(zip(clf.classes_, average)), 
    "fall_probability": fall_probability(average, clf.classes_), "intercept": clf.intercept_.tolist(), 
    "coef": clf.coef_.tolist()}

def fall_probability(prediction, classes):
    index = numpy.where(classes == "Fall")[0][0]
    return prediction[index]
    boolArray = list(map(lambda x: "Fall" in x, classes))
    predictionFall = prediction[boolArray]
    return numpy.sum(predictionFall)