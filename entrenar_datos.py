import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

# Desactive la notación científica para mayor claridad
np.set_printoptions(suppress=True)

# Cargar el modelo
model = tensorflow.keras.models.load_model('keras_model.h5')

# Cree la matriz de la forma correcta para alimentar el modelo de keras
# La 'longitud' o número de imágenes que puede poner en la matriz es
# determinado por la primera posición en la tupla de forma, en este caso 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Reemplaza esto con la ruta a tu imagen
image = Image.open('test5.jpg')

# redimensione la imagen a 224x224 con la misma estrategia que en TM2:
# cambiar el tamaño de la imagen para que sea de al menos 224x224 y luego recortar desde el centro
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#convierta la imagen en una matriz numpy
image_array = np.asarray(image)

# mostrar la imagen redimensionada
image.show()

# Normalizar la imagen
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Carga la imagen en la matriz
data[0] = normalized_image_array

# ejecutar la inferencia
prediction = model.predict(data)
print(prediction)

for i in prediction:
  if i[0] > 0.8:
    print("Usted tiene neumonia")
  elif i[1] > 0.8:
    print("Usted tiene covid")
  elif i[2] > 0.8:
    print("Usted está sano ")
  elif i[3] > 0.8:
    print("Usted tiene neumonia viral")
  else:
    print("No hay datos entrenados sobre este caso")