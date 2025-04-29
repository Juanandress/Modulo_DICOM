import pydicom
import numpy as np
import os
from PIL import Image
import os

#import matplotlib.pyplot as plt


def extraerDatosYMedatados(file_path, output_meta, output_img, index):
    # Cargar el archivo DICOM
    fileDicom = pydicom.dcmread(file_path)

    # Extraer metadatos
    metaData = {elem.keyword: elem.value for elem in fileDicom if elem.keyword}

    # Guardar Metadatos en un  txt
    saveMetadata = os.path.join(output_meta, f"metadata_{index}.txt")
    with open(saveMetadata, "w") as f:
        for key, value in metaData.items():
            f.write(f"{key}: {value}\n")
    print(f"Metadatos guardados en: {saveMetadata}")
    extraerImagenJPG(fileDicom, output_img, index)

def extraerImagenJPG(fileDicom, output_img, index):
    # Extraer datos de imagen
    if hasattr(fileDicom, "pixel_array"):
        pixel_array = fileDicom.pixel_array

        # Comprobar el rango de los píxeles antes de la normalización
        print(f"Imagen {index}: Valor mínimo de píxel:", np.min(pixel_array))
        print(f"Imagen {index}: Valor máximo de píxel:", np.max(pixel_array))

        # Obtener valores de la ventana
        window_center = fileDicom.get('WindowCenter', None)
        window_width = fileDicom.get('WindowWidth', None)

        if window_center and window_width:
            # Ajustar la ventana
            window_center = float(window_center)
            window_width = float(window_width)

            # Aplicar la ventana
            lower_bound = window_center - window_width // 2
            upper_bound = window_center + window_width // 2
            pixel_array = np.clip(pixel_array, lower_bound, upper_bound)

            # Normalizar a 0-255
            pixel_array = pixel_array - np.min(pixel_array)
            pixel_array = pixel_array / np.max(pixel_array)
            pixel_array = (pixel_array * 255).astype(np.uint8)

        # Crear la imagen desde el array de píxeles
        #image = Image.fromarray(pixel_array)
        image = Image.fromarray(pixel_array).convert('L')  # 'L' fuerza la escala de grises

        # Guardar la imagen como PNG
        image_file = os.path.join(output_img, f"image_{index}.jpg")
        image.save(image_file)
        print(f"Imagen {index} guardada en: {image_file}")
    else:
        print(f"El archivo DICOM {index} no contiene datos de imagen.")

def main():
    """ Función principal para procesar archivos DICOM en un directorio. """
    dicom_directory = "Modulo_DICOM\ArchivosDICOM"
    # Obtener la ruta del directorio donde se encuentra el script
    project_root = os.path.dirname(os.path.abspath(__file__))  # 
    output_metadata = os.path.join(project_root, "ResultadoDICOM", "metadata")
    output_images = os.path.join(project_root, "ResultadoDICOM", "images")


    # Crear ambos directorios si no existen
    os.makedirs(output_metadata, exist_ok=True)
    os.makedirs(output_images, exist_ok=True)

    dicom_files = [f for f in os.listdir(dicom_directory) if f.endswith('.dcm')]

    # Procesar n archivos que deseemos, para este caso 5 a modo de prueba
    for index, dicom_file in enumerate(dicom_files[:15]):
        dicom_file_path = os.path.join(dicom_directory, dicom_file)
        extraerDatosYMedatados(dicom_file_path, output_metadata, output_images, index + 1)

if __name__ == "__main__":
    main()