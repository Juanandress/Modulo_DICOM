import pydicom
import numpy as np
import os
from PIL import Image

def extraerDatosYMedatados(file_path, output_dir, index):
    # Cargar el archivo DICOM
    fileDicom = pydicom.dcmread(file_path)

    # Extraer metadatos
    metaData = {elem.keyword: elem.value for elem in fileDicom if elem.keyword}

    # Guardar Metadatos en un  txt
    saveMetadata = os.path.join(output_dir, f"metadata_{index}.txt")
    with open(saveMetadata, "w") as f:
        for key, value in metaData.items():
            f.write(f"{key}: {value}\n")
    print(f"Metadatos guardados en: {saveMetadata}")

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
        image = Image.fromarray(pixel_array)

        # Guardar la imagen como PNG
        image_file = os.path.join(output_dir, f"image_{index}.png")
        image.save(image_file)
        print(f"Imagen {index} guardada en: {image_file}")
    else:
        print(f"El archivo DICOM {index} no contiene datos de imagen.")

# Ruta del directorio con archivos DICOM y directorio de salida
dicom_directory = "E:\ImagenesDICOM\listaImagenes"
output_directory = r"E:\ResultadoDICOM"

# Crear el directorio de salida si no existe
os.makedirs(output_directory, exist_ok=True)

# Listar todos los archivos DICOM en el directorio
dicom_files = [f for f in os.listdir(dicom_directory) if f.endswith('.dcm')]

# Procesar n archivos que deseemos, para este caso 5 a modo de prueba
for index, dicom_file in enumerate(dicom_files[:5]):
    dicom_file_path = os.path.join(dicom_directory, dicom_file)
    extraerDatosYMedatados(dicom_file_path, output_directory, index + 1)
