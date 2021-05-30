from ..data.make_dataset import extrac_img_features
from app import cos, client
from cloudant.query import Query

from app import ROOT_DIR, CLOUDANT_DB, BUCKET_NAME, FILE_MODEL_CONFIG, FILE_IMG_PROS_CONFIG
import pickle
import bz2


def predict_pipeline(c_image, model_info_db_name=CLOUDANT_DB):

    """
        Función para gestionar el pipeline completo de inferencia
        del modelo.

        Args:
            image (str):  Stream conexion a la imagen.

        Kwargs:
            model_info_db_name (str):  base de datos a usar para almacenar
            la info del modelo.

        Returns:
            list. Lista con las prediccion hecha.
    """

    # obteniendo la información del modelo en producción
    model_info = get_best_model_info(model_info_db_name)

    # Carga de la configuración del modelo // contenida en la informacion del modelo!
    model_config = model_info['model_config']

    # Carga parametros de la funcion de pre-processado
    img_pros_config = load_img_pros_config(model_info_db_name)['img_pros_config']

    # cargando y transformando los datos de entrada
    img_features = extrac_img_features(c_image, model_config, img_pros_config)

    # Descargando el objeto del modelo
    model_name = model_info['name']+'.pkl'
    print('------> Loading the model {} object from the cloud'.format(model_name))
    model = load_model(model_name)
    #model = pickle.load(open(model_name, 'rb'))  ## funcion temporal hasta reparar la carga desde Cloud

    #filename = model_info['name']+'.pkl.bz2'
    #sfile = bz2.BZ2File(filename, 'r')
    #model = pickle.load(sfile)

    return model.predict(img_features).tolist()



def load_model(name, bucket_name=BUCKET_NAME):
    """
         Función para cargar el modelo en IBM COS

         Args:
             name (str):  Nombre de objeto en COS a cargar.

         Kwargs:
             bucket_name (str):  depósito de IBM COS a usar.

        Returns:
            obj. Objeto descargado.
     """
    return cos.get_object_in_cos(name, bucket_name)


def get_best_model_info(db_name):
    """
         Función para cargar la info del modelo de IBM Cloudant

         Args:
             db_name (str):  base de datos a usar.

         Kwargs:
             bucket_name (str):  depósito de IBM COS a usar.

        Returns:
            dict. Info del modelo.
     """
    db = client.get_database(db_name)
    query = Query(db, selector={'status': {'$eq': 'in_production'}})
    return query()['docs'][0]


def load_model_config(db_name):
    """
        Función para cargar la info del modelo desde IBM Cloudant.

        Args:
            db_name (str):  Nombre de la base de datos.

        Returns:
            dict. Documento con la configuración del modelo.
    """
    db = client.get_database(db_name)
    query = Query(db, selector={'_id': {'$eq': FILE_MODEL_CONFIG}})
    return query()['docs'][0]



def load_img_pros_config(db_name):
    """
        Función para cargar la info del modelo desde IBM Cloudant.

        Args:
            db_name (str):  Nombre de la base de datos.

        Returns:
            dict. Documento con la configuración del modelo.
    """
    db = client.get_database(db_name)
    query = Query(db, selector={'_id': {'$eq': FILE_IMG_PROS_CONFIG}})
    return query()['docs'][0]