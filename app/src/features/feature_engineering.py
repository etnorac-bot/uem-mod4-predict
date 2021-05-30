import numpy as np
from skimage import transform, filters, io
from skimage.feature import hog

def feature_engineering(img, HOG_config):

    """
        Función de extraccion de los features de la imagen

        Args:
           img (Imagen):  La imagen de entrada.
           HOG_config (json): La configuracion de HOGs (en la info del modelo en produccion)
        Returns:
           img_hog. Lista de HOG concatenados
    """

    orient = HOG_config['HOG_orientation']  # Numero de orientaciones del HOG
    ppc_list = HOG_config['HOG_ppc'] # Tamaños de celdas para el HOG

    img = (0.5 - img / 255) # Cambio de orientacion (podríamos eliminarlo si lo quitamos tambien del Train)

    img_hog = np.concatenate([hog(img, orientations=orient,
                                       pixels_per_cell=(ppc, ppc),
                                       cells_per_block=(1, 1),
                                       visualize=False,
                                       block_norm='L1')[np.newaxis, :] for ppc in ppc_list],axis=1)

    return img_hog



def preprocess_img(img, img_pros_config):

    """
        Función de extraccion de los features de la imagen

        Args:
           img (Imagen):  La imagen de entrada.
           img_pros_config (json): La configuracion de la funcion de centrado de la imagen
        Returns:
           img: Image 24x24 en uint8 (0, 255)
    """

    ## Primero pasamos a escala de grises
    print('------> Image to grayscale')
    img = img2gray(img)

    ## Pasamos a Trazo oscuro sobre fondo claro
    print('------> Check image is dark ink over light paper')
    img = light_canvas(img)

    ## Maximizamos el contraste, borramos el fondo y pasamos a escala 0 255
    print('------> Cleaning image ')
    img = cleaning(img, img_pros_config)

    ## Recortamos y centramos la imagen
    print('------> Auto-crop image ')
    img = recorte(img, img_pros_config)

    # Finalmente la pasamos a 28x28 pixeles
    img = transform.resize(image=img, output_shape=(28, 28), anti_aliasing=True)
    img = np.array( img * 255    , dtype=np.uint8)

    ###################  Debug de la imagen, pero podría presentarla a la salida para feed-back y ampliar DB
    #io.imsave('imagen_result.jpg', img)
    ################

    return img



def img2gray(img):

    """
        Función para pasar la imagen a escala de grises (como en el set de training)
        caso de ser una imagen de entrada en color.

        Args:
           input_img (Imagen):  La imagen de entrada.
        Returns:
           exit_img(array de reales): La imagen en escala de grises.
    """
    img = np.array(img, dtype=float) # Operamos en float para no saturar los canales a 256
    if len(img.shape) == 2:  ## Imagen en escala de grises
        print('---------> Picture already in grayscale')
    elif len(img.shape) == 3:
        if img.shape[2] == 4:  ## Imagen en rbga
            print('---------> Picture from RGBA to grayscale')
            img = rgba2gray(img)
        elif exit_img.shape[2] == 3:  ## Imagen en rbg
            print('---------> Picture from RGB to grayscale')
            img = rgb2gray(img)


    return img



def rgb2gray(img):
    """
        Función para pasar la imagen RGB a escala de grises
        Args:
           input_img (Imagen):  Imag en formato RGBA.
        Returns:
           exit_img: La imagen en escala de grises.
    """
    return (img[:,:,0]+img[:,:,1]+img[:,:,2])

def rgba2gray(img):
    """
        Función para pasar la imagen RGBA a escala de grises
        Args:
           input_img (Imagen):  Imag en formato RGBA.
        Returns:
           exit_img: La imagen en escala de grises.
    """
    return (img[:,:,0]+img[:,:,1]+img[:,:,2])*img[:,:,3]


def light_canvas(img):

    """
        Función para pasar la imagen a que sea trazo oscuro sobre fondo claro
        le da la vuelta a la imagen.
        Se basa en que la media va a ser el fondo.

        Args:
           input_img (array de reales:  La imagen de entrada.
        Returns:
           exit_img(array de reales): La imagen donde la media es menor que el valos medio entre extremos.
    """

    if img.mean() < ((img.max()+img.min())/2):  # Si la imagen es negro sobre blanco
        print('---------> Picture invertion')
        img = img.max() - img

    return img


def cleaning(img, img_pros_config):

    """
        Función que ajusta el contraste al maximo y limpia el fondo saturandolo a blanco desde un limite.

        Args:
           input_img (array de reales):  La imagen de entrada.
           img_pros_config (json): La configuracion de la funcion de centrado de la imagen
        Returns:
           exit_img(array en uint8): La imagen donde la media es menor que el valos medio entre extremos.
    """

    # Aumentamos el contraste al maximo incluso saturando a partir de ciertos percentiles.
    print('---------> Max contrast ')
    percen_upp = img_pros_config['percen_upp']
    percen_low = img_pros_config['percen_low']
    img = ((img - np.percentile(img, percen_low)) /
           (np.percentile(img, percen_upp) - np.percentile(img, percen_low))) * 255
    img = np.array(np.clip(img, 0, 255), dtype=np.uint8)


    # Saturado del fondo
    print('---------> Background saturation ')
    sat_limit = img_pros_config['sat_limit'] # Limite para empezar a saturar
    for j in range(img.shape[1]):
        for i in range(img.shape[0]):
            if img[i, j] > sat_limit:
                img[i, j] = 255

    return img



def recorte(img, img_pros_config):

    """
        Función que busca la imagen y recorta el numero.
        Ajusta el recorte basado en cuanto negro ha visto y añade un porcentaje del tamaño entre cortes
        para volver a meter el numero dentro y que la imagen sea cuadrada

        Args:
           input_img (array en uint8):  La imagen limpia y contrastada.
        Returns:
           exit_img(array en uint8): La imagen recordata.
    """

    limite = img_pros_config['limite'] # % de cuanto oscuro dejo fuera del limite de recorte
    f_lados = img_pros_config['f_lados'] # fraccion de tamaño que añado a los lados

    # Para operar más facil hago que lo negro sea 255 y lo blanco 0
    img = 255 - img

    # Busco los limitenes
    print('---------> Finding number limits ')
    for i in range(img.shape[0]):
        if img[i:, :].sum() < img.sum() * limite:
            y_min = i
            break

    for i in range(img.shape[0]):
        if img[:img.shape[0] - i, :].sum() < img.sum() * limite:
            y_max = img.shape[0] - i
            break

    for i in range(img.shape[1]):
        if img[:, i:].sum() < img.sum() * limite:
            x_min = i
            break

    for i in range(img.shape[1]):
        if img[:, :img.shape[1] - i].sum() < img.sum() * limite:
            x_max = img.shape[1] - i
            break

    print('---------> Center of image ')
    # Me quedo con el lado más largo
    delta_x = x_max - x_min
    delta_y = y_max - y_min
    delta = max(delta_x, delta_y)

    # Encuentro el centro de la imagen
    x_cent = x_min + delta_x / 2
    y_cent = y_min + delta_y / 2

    # Defino el corte con el lado más largo más una fraccion
    x_min_p = np.uint(np.rint(x_cent - f_lados * delta / 2).clip(0, img.shape[1]))
    x_max_p = np.uint(np.rint(x_cent + f_lados * delta / 2).clip(0, img.shape[1]))
    y_min_p = np.uint(np.rint(y_cent - f_lados * delta / 2).clip(0, img.shape[0]))
    y_max_p = np.uint(np.rint(y_cent + f_lados * delta / 2).clip(0, img.shape[0]))

    # Devuelvo la imagen recorgada y devuelvo al formato de entrada donde 255 es blando y 0 negro.
    return 255 - img[y_min_p:y_max_p, x_min_p:x_max_p]
