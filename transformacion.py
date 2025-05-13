import pydicom
import numpy as np
import os
from PIL import Image
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import csv
from torchvision.models import resnet18, ResNet18_Weights

# Cargar el modelo preentrenado correctamente
resnet18_model = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet18_model.eval()  # Establecer en modo evaluación

# Extraer características quitando la última capa FC
extractorCaracteristicas = torch.nn.Sequential(*list(resnet18_model.children())[:-1])

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ajustar tamaño para ResNet-18
    transforms.ToTensor(),  # Convertir a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización basada en ImageNet
])


def extraccionCaracteristicas(image_path):
    """ Extrae características de una imagen usando ResNet-18 """
    image = Image.open(image_path).convert("RGB")  # Convertir a RGB para compatibilidad con el modelo
    image = transform(image).unsqueeze(0)  # Añadir batch dimension

    with torch.no_grad():
        caracteristicas = extractorCaracteristicas(image)

    return caracteristicas.squeeze().numpy()  # Convertir a numpy

def guardarCaracteristicasCsv(dicCaracteristicas, output_csv):
    """ Guarda los vectores de características en un archivo CSV """
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Escribir encabezados (nombre de la imagen + índices de los vectores)
        caracteristicaslength = len(next(iter(dicCaracteristicas.values())))  # Obtener tamaño del vector
        headers = ['Imagen'] + [f'Feature_{i}' for i in range(caracteristicaslength)]
        writer.writerow(headers)
        
        # Escribir los datos
        for imageName, vectorCaracteristicas in dicCaracteristicas.items():
            writer.writerow([imageName] + vectorCaracteristicas.tolist())

    print(f"Características guardadas en: {output_csv}")

def extraerDatosYMedatados(file_path, output_meta, output_img,output_csv, index):
    # Cargar el archivo DICOM
    fileDicom = pydicom.dcmread(file_path)

    # Extraer metadatos sin pixeles
    metaData = {elem.keyword: elem.value for elem in fileDicom if elem.keyword and elem.keyword != "PixelData"}

    # Guardar Metadatos en un  txt
    saveMetadata = os.path.join(output_meta, f"metadata_{index}.txt")
    with open(saveMetadata, "w") as f:
        for key, value in metaData.items():
            f.write(f"{key}: {value}\n")
    print(f"Metadatos guardados en: {saveMetadata}")
    extraerImagenJPG(fileDicom, output_img, output_csv, index)

def extraerImagenJPG(fileDicom, output_img, output_csv, index):
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

         # 🔹 Extraer características con ResNet-18
        vectorCaracteristicas = extraccionCaracteristicas(image_file)
        #diccionarioCaracteristicas[f'image_{index}.jpg'] = vectorCaracteristicas
        #print(f"Características extraídas para imagen {index}: {vectorCaracteristicas.shape}")
        with open(output_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f'image_{index}.jpg'] + vectorCaracteristicas.tolist())

        print(f"Características extraídas y guardadas para imagen {index}")

    else:
        print(f"El archivo DICOM {index} no contiene datos de imagen.")

def main():


    dicom_directory = "ArchivosDICOM"
    project_root = os.path.dirname(os.path.abspath(__file__))  #
    output_metadata = os.path.join(project_root, "ResultadoDICOM", "metadata")
    output_images = os.path.join(project_root, "ResultadoDICOM", "images")
    output_csv_dir = os.path.join(project_root, "ResultadoDICOM", "caracteristicas")

    output_csv = os.path.join(output_csv_dir, "caracteristicas.csv")


    # Crear  directorios si no existen
    os.makedirs(output_metadata, exist_ok=True)
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_csv_dir, exist_ok=True)
    #dicom_files = [f for f in os.listdir(dicom_directory) if f.endswith('.dcm')]
    print("Ruta absoluta buscada:", os.path.abspath(dicom_directory))
    print("¿Existe la carpeta?", os.path.exists(dicom_directory))

    dicom_files = []
    for subdir, _, files in os.walk(dicom_directory):  # Recorrer subdirectorios
        for file in files:
            if file.endswith(".dcm"):
                dicom_files.append(os.path.abspath(os.path.join(subdir, file)))

    # Procesar n archivos que deseemos, para este caso 5 a modo de prueba
    for index, dicom_file in enumerate(dicom_files[:50]):
        dicom_file_path = dicom_file
        extraerDatosYMedatados(dicom_file_path, output_metadata, output_images, output_csv, index + 1)

if __name__ == "__main__":
    main()