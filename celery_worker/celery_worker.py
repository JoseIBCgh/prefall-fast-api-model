from time import sleep
from celery import Celery
from celery.utils.log import get_task_logger
import os
# Initialize celery
#celery = Celery('tasks', broker='amqp://guest:guest@127.0.0.1:5672//', backend='rpc')
celery = Celery(__name__)
celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "amqp://guest:guest@srv.ibc.bio:32837//")
celery.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://srv.ibc.bio:32828/0") 
# Create logger - enable to display messages on task logger
celery_log = get_task_logger(__name__)
# Create Order - Run Asynchronously with celery
# Example process of long running task
import pickle
import numpy
import warnings
'''
import plotly
import plotly.graph_objs as go
'''
import json
import pandas as pd
import io

import libraries


modelo = pickle.load(open('modelo.pkl', 'rb'))


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
def predict(csv_content):

    datos = pd.read_csv(io.StringIO(csv_content))


    #Renombramos columnas para adaptarlo al script libraries
    datos.rename(columns={'acc_x': 'ax', 'acc_y': 'ay', 'acc_z': 'az'}, inplace=True)

    datos.rename(columns={'gyr_x': 'GYR_X', 'gyr_y': 'GYR_Y', 'gyr_z': 'GYR_Z'}, inplace=True)

    datos.rename(columns={'mag_x': 'MAG_X', 'mag_y': 'MAG_Y', 'mag_z': 'MAG_Z'}, inplace=True)

    datos.rename(columns={'lacc_x': 'LACC_X', 'lacc_y': 'LACC_Y', 'lacc_z': 'LACC_Z'}, inplace=True)

    datos.rename(columns={'quat_x': 'QUAT_X', 'quat_y': 'QUAT_Y', 'quat_z': 'QUAT_Z', 'quat_w' : 'QUAT_W'}, inplace=True)
    
    datos.rename(columns={'time': 'TIME'}, inplace=True)
    """
    datos['LACC_X'] = datos['ax']
    datos['LACC_Y'] = datos['ay']
    datos['LACC_Z'] = datos['az']

    datos['QUAT_X'] = datos['ax']
    datos['QUAT_Y'] = datos['ay']
    datos['QUAT_Z'] = datos['az']
    datos['QUAT_W'] = datos['az']
    """

    # Filtro de outliers general. Saltamos los primeros segundos para eliminar datos anomalos
    datos = libraries.filtro_acelerometro(datos.iloc[500:-500, :])

    #Definimos que el sujeto está caminando durante todo el proceso de medición. *A futuro esto puede modificarse
    datos['caminar'] = [1 for i in range(len(datos))]


    #Obtenemos las fases de la marcha
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # Filtro de tramos caminados
        if any(datos['caminar'] == 1):
            if any(datos['caminar'][-10:] == 1):
                datos['caminar'].iloc[-10:] = 0
            datos['estado']=libraries.fases_marcha_global(datos)

    # Agrupa los datos por fases
    #datos=datos.dropna()
    fases = datos.groupby('estado')

    # Itera sobre cada grupo de fases
    fila = {}
    for i, (fase, grupo) in enumerate(fases, start=1):
        # Calcula la duración de la fase
        duracion = grupo['TIME'].iloc[-1] - grupo['TIME'].iloc[0]
        duracion=len(grupo)/100

        #acelerometro
        ax_mean = grupo['ax'].mean()
        ay_mean = grupo['ay'].mean()
        az_mean = grupo['az'].mean()

        ax_std = grupo['ax'].std()
        ay_std = grupo['ay'].std()
        az_std = grupo['az'].std()

        #CALCULAMOS VALORES MEDIOS DE SENSORES PARA CADA FASE DE MARCHA
        #Aceleracion lineal
        lax_mean = grupo['LACC_X'].mean()
        lay_mean = grupo['LACC_Y'].mean()
        laz_mean = grupo['LACC_Z'].mean()

        #giroscopio
        gx_mean = grupo['GYR_X'].mean()
        gy_mean = grupo['GYR_Y'].mean()
        gz_mean = grupo['GYR_Z'].mean()

        #magnetometro
        mx_mean = grupo['MAG_X'].mean()
        my_mean = grupo['MAG_Y'].mean()
        mz_mean = grupo['MAG_Z'].mean()

        #quaterniones
        qx_mean = grupo['QUAT_X'].mean()
        qy_mean = grupo['QUAT_Y'].mean()
        qz_mean = grupo['QUAT_Z'].mean()
        qw_mean = grupo['QUAT_Z'].mean()

        #CALCULAMOS DESVIACION TIPICA DE SENSROES PARA CADA FASE DE MARCHA
        #Aceleracion lineal
        lax_std = grupo['LACC_X'].std()
        lay_std = grupo['LACC_Y'].std()
        laz_std = grupo['LACC_Z'].std()

        #giroscopio
        gx_std = grupo['GYR_X'].std()
        gy_std = grupo['GYR_Y'].std()
        gz_std = grupo['GYR_Z'].std()

        #magnetometro
        mx_std = grupo['MAG_X'].std()
        my_std = grupo['MAG_Y'].std()
        mz_std = grupo['MAG_Z'].std()

        #quaterniones
        qx_std = grupo['QUAT_X'].std()
        qy_std = grupo['QUAT_Y'].std()
        qz_std = grupo['QUAT_Z'].std()
        qw_std = grupo['QUAT_Z'].std()


        # Agrega las características al diccionario fila
        fila.update({
            'duracion_f'+str(i): duracion,
            'ax_mean_f'+str(i): ax_mean, 'ay_mean_f'+str(i): ay_mean, 'az_mean_f'+str(i): az_mean,
            'ax_std_f'+str(i): ax_std, 'ay_std_f'+str(i): ay_std, 'az_std_f'+str(i): az_std,
            'lax_mean_f'+str(i): lax_mean, 'lay_mean_f'+str(i): lay_mean, 'laz_mean_f'+str(i): laz_mean,
            'lax_std_f'+str(i): lax_std, 'lay_std_f'+str(i): lay_std, 'laz_std_f'+str(i): laz_std,
            'gx_mean_f'+str(i): gx_mean, 'gy_mean_f'+str(i): gy_mean, 'gz_mean_f'+str(i): gz_mean,
            'gx_std_f'+str(i): gx_std, 'gy_std_f'+str(i): gy_std, 'gz_std_f'+str(i): gz_std,
            'mx_mean_f'+str(i): mx_mean, 'my_mean_f'+str(i): my_mean, 'mz_mean_f'+str(i): mz_mean,
            'mx_std_f'+str(i): mx_std, 'my_std_f'+str(i): my_std, 'mz_std_f'+str(i): mz_std,
        })

    df_fila = pd.DataFrame([fila])
    prediccion=modelo.predict_proba([df_fila.iloc[0].tolist()])[0][1]

    texto="Probabildiad de caida del paciente : "+str(prediccion)

    celery_log.info(f"Prediction Complete!")
    celery_log.info(texto)
    return {"model_id": model_id(),"message": "Prediction complete", 
    "probability": prediccion, "datos": fila}

def coef():
    coef_avg = 0
    for m in clf.calibrated_classifiers_:
        coef_avg = coef_avg + m.base_estimator.coef_
    coef_avg  = coef_avg/len(clf.calibrated_classifiers_)
    return coef_avg

def intercept():
    intercept_avg = 0
    for m in clf.calibrated_classifiers_:
        intercept_avg = intercept_avg + m.base_estimator.intercept_
    intercept_avg  = intercept_avg/len(clf.calibrated_classifiers_)
    return intercept_avg

def training_data():
    df = pd.read_csv("datos_aug.csv")
    subdf = df.groupby('Position').head(20)
    subdf = subdf.drop(columns=["Unnamed: 0"])
    subdf = subdf.reset_index(drop=True)
    print(subdf)
    classes = df['Position'].unique()
    result = {}
    for c in classes:
        result[c] = subdf[subdf['Position'] == c].iloc[:,:3].to_json()
    print(result)
    return result

def model_id():
    return 1
    from random import randrange
    return randrange(100000)
    sum = 0
    for m in clf.calibrated_classifiers_:
        sum = sum + m
    return sum / len(clf.calibrated_classifiers_)

def fall_probability(prediction, classes):
    boolArray = list(map(lambda x: "Fall" in x, classes))
    predictionFall = prediction[boolArray]
    return numpy.sum(predictionFall)
'''
def generate_plot():
    fig = go.FigureWidget()
    for coef, intercept in zip(clf.coef_,clf.intercept_):
        z = lambda x,y: (-intercept-coef[0]*x-coef[1]*y) / coef[2]
        tmp = numpy.linspace(-2,2,51)
        x,y = numpy.meshgrid(tmp,tmp)
        fig.add_surface(x=x, y=y, z=z(x,y), colorscale='Greys', showscale=False)

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
'''