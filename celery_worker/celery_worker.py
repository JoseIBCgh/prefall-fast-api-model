from time import sleep
from celery import Celery
from celery.utils.log import get_task_logger
import os
# Initialize celery
#celery = Celery('tasks', broker='amqp://guest:guest@127.0.0.1:5672//', backend='rpc')
celery = Celery(__name__)
celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "amqp://guest:guest@rabbitmq:5673//")
celery.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6380/0") 
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

with open('model_linear.pkl', 'rb') as fid:
    clf = pickle.load(fid)

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
    array = [1, 1, 1]
    array = numpy.array(array).reshape(1,-1)
    prediction = clf.predict_proba(array)[0]

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

    # Crea un dataframe vacío para almacenar las características
    datos_final = pd.DataFrame(columns=['duracion_f1',
                                            'lax_mean_f1', 'lay_mean_f1', 'laz_mean_f1',
                                            'gx_mean_f1', 'gy_mean_f1', 'gz_mean_f1',
                                            'mx_mean_f1', 'my_mean_f1', 'mz_mean_f1',
                                            'qx_mean_f1', 'qy_mean_f1', 'qz_mean_f1','qw_mean_f1',
                                            'duracion_f2',
                                            'lax_mean_f2', 'lay_mean_f2', 'laz_mean_f2',
                                            'gx_mean_f2', 'gy_mean_f2', 'gz_mean_f2',
                                            'mx_mean_f2', 'my_mean_f2', 'mz_mean_f2',
                                            'qx_mean_f2', 'qy_mean_f2', 'qz_mean_f2','qw_mean_f2',
                                            'duracion_f3',
                                            'lax_mean_f3', 'lay_mean_f3', 'laz_mean_f3',
                                            'gx_mean_f3', 'gy_mean_f3', 'gz_mean_f3',
                                            'mx_mean_f3', 'my_mean_f3', 'mz_mean_f3',
                                            'qx_mean_f3', 'qy_mean_f3', 'qz_mean_f3','qw_mean_f3',
                                            'duracion_f4',
                                            'lax_mean_f4', 'lay_mean_f4', 'laz_mean_f4',
                                            'gx_mean_f4', 'gy_mean_f4', 'gz_mean_f4',
                                            'mx_mean_f4', 'my_mean_f4', 'mz_mean_f4',
                                            'qx_mean_f4', 'qy_mean_f4', 'qz_mean_f4','qw_mean_f4',
                                            ])

    # Itera sobre cada grupo de fases
    i=1
    for fase, grupo in fases:
        # Calcula la duración de la fase
        duracion = grupo['TIME'].iloc[-1] - grupo['TIME'].iloc[0]
        duracion=len(grupo)/100

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
        lax_mean = grupo['LACC_X'].std()
        lay_mean = grupo['LACC_Y'].std()
        laz_mean = grupo['LACC_Z'].std()

        #giroscopio
        gx_mean = grupo['GYR_X'].std()
        gy_mean = grupo['GYR_Y'].std()
        gz_mean = grupo['GYR_Z'].std()

        #magnetometro
        mx_mean = grupo['MAG_X'].std()
        my_mean = grupo['MAG_Y'].std()
        mz_mean = grupo['MAG_Z'].std()

        #quaterniones
        qx_mean = grupo['QUAT_X'].std()
        qy_mean = grupo['QUAT_Y'].std()
        qz_mean = grupo['QUAT_Z'].std()
        qw_mean = grupo['QUAT_Z'].std()


        # Agrega las características al dataframe de características
        datos_final = datos_final.append({'duracion_f'+str(i): duracion, 
                                                    'lax_mean_f'+str(i): lax_mean, 'lay_mean_f'+str(i): lay_mean, 'laz_mean_f'+str(i): laz_mean,
                                                    'gx_mean_f'+str(i): gx_mean, 'gy_mean_f'+str(i): gy_mean, 'gz_mean_f'+str(i): gz_mean,
                                                    'mx_mean_f'+str(i): mx_mean, 'my_mean_f'+str(i): my_mean, 'mz_mean_f'+str(i): mz_mean,
                                                    'qx_mean_f'+str(i): qx_mean, 'qy_mean_f'+str(i): qy_mean, 'qz_mean_f'+str(i): qz_mean,'qw_mean_f'+str(i): qw_mean,

                                                    }, ignore_index=True)

        i=i+1



    #Combinamos los resultados
    datos_final = pd.concat([datos_final.iloc[0], datos_final.iloc[1], datos_final.iloc[2],datos_final.iloc[3]]).dropna()

    # Crear un nuevo dataframe con una sola fila y las columnas resultantes
    datos_final = pd.DataFrame([datos_final.tolist()], columns=datos_final.index)

    #Predecimos la probabilidad de caida con el modelo
    prediccion=modelo.predict_proba([datos_final.iloc[0].tolist()])[0][1]
    texto="Probabildiad de caida del paciente : "+str(prediccion)

    celery_log.info(f"Prediction Complete!")
    celery_log.info(texto)
    return {"model_id": model_id(),"message": "Prediction complete", 
    "probability": prediccion, "datos": datos_final.to_dict(orient='records')}

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