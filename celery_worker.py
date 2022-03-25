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
import plotly
import plotly.graph_objs as go
import json
import pandas

with open('model_linear.pkl', 'rb') as fid:
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
    array = [numpy.average(acc_x), numpy.average(acc_y), numpy.average(acc_z)]
    array = numpy.array(array).reshape(1,-1)
    prediction = clf.predict_proba(array)[0]
    celery_log.info(f"Prediction Complete!")
    print(clf.calibrated_classifiers_)
    return {"model_id": model_id(),"message": "Prediction complete", "prediction": list(zip(clf.classes_, prediction)), 
    "intercept": intercept().tolist(), "coef": coef().tolist(), "training_data": training_data()}

def coef():
    coef_avg = 0
    for m in clf.calibrated_classifiers_:
        coef_avg = coef_avg + m.base_estimator.coef_
        print(m.base_estimator.coef_)
    coef_avg  = coef_avg/len(clf.calibrated_classifiers_)
    return coef_avg

def intercept():
    intercept_avg = 0
    for m in clf.calibrated_classifiers_:
        intercept_avg = intercept_avg + m.base_estimator.intercept_
    intercept_avg  = intercept_avg/len(clf.calibrated_classifiers_)
    return intercept_avg

def training_data():
    df = pandas.read_csv("datos_aug.csv")
    subdf = df.groupby('Position').head(20)
    classes = df['Position'].unique()
    result = {}
    for c in classes:
        result[c] = subdf[subdf['Position'] == c].iloc[:3].to_json()
    return result

def model_id():
    return 1
    sum = 0
    for m in clf.calibrated_classifiers_:
        sum = sum + m
    return sum / len(clf.calibrated_classifiers_)

def fall_probability(prediction, classes):
    boolArray = list(map(lambda x: "Fall" in x, classes))
    predictionFall = prediction[boolArray]
    return numpy.sum(predictionFall)

def generate_plot():
    fig = go.FigureWidget()
    for coef, intercept in zip(clf.coef_,clf.intercept_):
        z = lambda x,y: (-intercept-coef[0]*x-coef[1]*y) / coef[2]
        tmp = numpy.linspace(-2,2,51)
        x,y = numpy.meshgrid(tmp,tmp)
        fig.add_surface(x=x, y=y, z=z(x,y), colorscale='Greys', showscale=False)

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
