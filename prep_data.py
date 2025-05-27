import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def limpiarCsv():
    df = pd.read_csv("ResultadoDICOM/fusionado/dataset_fusionado.csv")
    
    columns_to_drop = [
    "AccessionNumber", "InstanceCreatorUID", "SeriesInstanceUID", "StudyInstanceUID",
    "SOPInstanceUID", "AcquisitionDate", "InstanceCreationDate","InstanceCreationTime", "StudyDate",
    "ReferringPhysicianName", "PatientName", "PatientID", "PatientIdentityRemoved",
    "OrganExposed", "PatientBirthDate", "StudyTime", "SeriesTime",
    "PresentationIntentType", "PresentationLUTShape", "StudyID", "ViewCodeSequence", 
    "AcquisitionContextSequence", "BodyPartExamined", "ContentDate","ContentTime", "DeidentificationMethod", 
    "DeidentificationMethodCodeSequence","ImageType", "ImagerPixelSpacing","InstanceNumber", "LongitudinalTemporalInformationModified",
    "Manufacturer", "PatientSex", "PositionerType","SOPClassUID",
    "SeriesDate","SpecificCharacterSet","WindowCenterWidthExplanation","RescaleType", "AnatomicRegionSequence",
    "(0008,0102) Coding Scheme Designator            SH", "Modality", "Id"
    ]

    # Diccionario de mapeo para la codificación ordinal de Code Meaning
    view_mapping = {
        'cranio-caudal': 0,  
        'medio-lateral oblique': 1   
    }

    # Diccionario de mapeo para la codificación ordinal de voilut
    voilut_mapping ={
        'LINEAR':0,
        'LINEAR_EXACT': 1,
        'SIGMOID':2
    }
    ImageLaterality_mapping ={
        'L':0,
        'R':1
    }

    # Eliminar columnas no necesarias
    df = df.drop(columns=columns_to_drop, errors="ignore")

    #Se realiza el renombrado de columna
    df.rename(columns={"(0008,0104) Code Meaning                        LO": "ViewMeaning"}, inplace=True)
    df["ViewMeaning"] = df["ViewMeaning"].astype(str).str.replace("'", "")
    # Se realiza el mapeo
    df["ViewMeaning"] = df["ViewMeaning"].map(view_mapping)

    # Se realiza el mapeo
    df["VOILUTFunction"] = df["VOILUTFunction"].map(voilut_mapping)

    df["ImageLaterality"]= df["ImageLaterality"].map(ImageLaterality_mapping)
    # codificacion de BurnedInAnnotation 1-0
    df["BurnedInAnnotation"] = df["BurnedInAnnotation"].map({"NO": 0, "YES": 1})
    # limpiar edad quitar caracteres especiales
    df["PatientAge"] = df["PatientAge"].str.extract(r'(\d+)').astype(int)
    
    df["PatientOrientation"] = df["PatientOrientation"].astype(str).str.replace("[", "").str.replace("]", "").str.replace("'", "").str.replace(",", "").str.strip()
    df_cod = pd.get_dummies(df, columns=["PatientOrientation","PhotometricInterpretation", "PixelIntensityRelationship","DetectorType"])  # One-Hot Encoding
    
    # Identificar columnas generadas por One-Hot Encoding
    col_convert = [col for col in df_cod.columns if col.startswith(("PatientOrientation_", "PhotometricInterpretation_", "PixelIntensityRelationship_", "DetectorType_"))]
    
    # Aplicar .astype(int) solo en estas columnas
    df_cod[col_convert] = df_cod[col_convert].astype(int)
    df_cod["Diagnostico"] = 1  # Asumimos que todas son inicialmente negativ
    df_cod.loc[-8:, "Diagnostico"] = 0  # Seleccionamos las últimas 8 filas
    df_cod.iloc[-8:, df_cod.columns.get_loc("Diagnostico")] = 0
    df_cod.to_csv('dataset_cod.csv', index = False)
    print("CSV cod guardado ")
    return df_cod
    

def normalizar(df):

    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df_normalized.to_csv('breast_cancer.csv', index = False)
    print("CSV norm guardado ")

def main():
    df = limpiarCsv()
    normalizar(df)


if __name__ == "__main__":
    main()