import shutil

import pydicom
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from pydicom.multival import MultiValue

def extraerDatosYMedatados(file_path, output_dir, index):
    # Cargar el archivo DICOM
    fileDicom = pydicom.dcmread(file_path)

    # Extraer metadatos
    metaData = {elem.keyword: elem.value for elem in fileDicom if elem.keyword}

    # Guardar Metadatos en un  txt
    saveMetadata = os.path.join(output_dir, f"metadata_{index}.txt")
    with open(saveMetadata, "w", encoding="utf-8") as f:
        for key, value in metaData.items():
            f.write(f"{key}: {str(value)}\n")
    print(f"Metadatos guardados en: {saveMetadata}")

    # Extraer datos de imagen
    if hasattr(fileDicom, "pixel_array"):
        pixel_array = getattr(fileDicom,"pixel_array",None)
        if pixel_array is None:
            print(f"El archivo dicom{index} no contiene datos de imagen o está corrupto")
            return


        # Comprobar el rango de los píxeles antes de la normalización
        print(f"Imagen {index}: Valor mínimo de píxel:", np.min(pixel_array))
        print(f"Imagen {index}: Valor máximo de píxel:", np.max(pixel_array))

        def obtenerValorUnico(valor):
            if isinstance(valor, MultiValue):
                return float(valor[0])
            return float(valor)

        # Obtener valores de la ventana
        window_center = fileDicom.get('WindowCenter', None)
        window_width = fileDicom.get('WindowWidth', None)

        if window_center and window_width:
            # Ajustar la ventana
            window_center = obtenerValorUnico(window_center)
            window_width = obtenerValorUnico(window_width)

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

        # Guardar la imagen
        image_file = os.path.join(output_dir, f"image_{index}.png")
        image.save(image_file)
        print(f"Imagen {index} guardada en: {image_file}")
    else:
        print(f"El archivo DICOM {index} no contiene datos de imagen.")

# Ruta del directorio con archivos DICOM y directorio de salida
dicom_directory = "D:/ImagenesDICOM/listaImagenes"
output_directory = Path("D:/ResultadoDICOM")
if output_directory.exists():
    for file in output_directory.iterdir():
        if file.is_file() or file.is_symlink():
            file.unlink() #Elimina un archivo o un acceso directo
        elif file.is_dir():
            shutil.rmtree(file)#Borra carpetas y su contenido

# Crear el directorio de salida si no existe
output_directory.mkdir(parents=True, exist_ok=True)


# Listar todos los archivos DICOM en el directorio
dicom_files = [f for f in os.listdir(dicom_directory) if f.endswith('.dcm')]

# Procesar n archivos que deseemos, limite configurable con numArchivos
numArchivos= input("¿Cuántos archivos desea procesar? (Presione enter para procesar todos los archivos):")
numArchivos= int(numArchivos) if numArchivos.isdigit() else None
dicom_files = dicom_files[:numArchivos]if numArchivos else dicom_files

for index, dicom_file in enumerate(tqdm(dicom_files, desc="Procesando archivos...")):
    dicom_file_path = os.path.join(dicom_directory, dicom_file)
    extraerDatosYMedatados(dicom_file_path, output_directory, index + 1)
