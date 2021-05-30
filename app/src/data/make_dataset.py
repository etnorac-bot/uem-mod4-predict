from skimage import io
import numpy as np
from app import cos
from ..features.feature_engineering import preprocess_img ,feature_engineering

import os
from pathlib import Path
from PIL import Image


def extrac_img_features(c_img, model_config, img_pros_config):
    """
        Función que permite crear el set de variables para la inferencia del modelo.

        Args:
           img (List):  Ruta de la imagen.
           model_info (dict):  Información del modelo en producción.
           img_pros_config (dict):  Información de config de la funcion de centrado.

        Returns:
          img_hog. Array con los parametros de la imagen procesados para inferencia.
    """

    print('---> Getting picture')
    img = get_raw_data_from_request(c_img)
    print('---> Pre-processing picture')
    img = preprocess_img(img,img_pros_config)
    print('---> Feature engineering')
    img_hog = feature_engineering(img, model_config)

    return img_hog

def get_raw_data_from_request(c_img):

    """
        Función para obtener nuevas observaciones desde request

        Args:
           image:  conexion a la imagen a cargar

        Returns:
           import_img. Imagen cargada con skimage. io
    """

    image = Image.open(c_img.stream).convert("L")  # convert("L") transforma a escala de grises
    # podría eliminar el paso de transformar a gris en el procesado pues esto se asegura de que ya es así

    return image




