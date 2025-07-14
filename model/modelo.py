import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import os

def categorizar_causa(row):
    """Función para crear categorías más específicas basadas en combinaciones de características"""
    if row['probabilidad_terminar'] <= 2:
        if row['estres_estudios'] >= 4 and row['apoyo_familiar'] <= 2:
            return 'estrés_crónico_falta_apoyo'
        elif row['comprension'] <= 2 and row['relacion_profesores'] <= 2:
            return 'dificultad_académica'
        elif row['emocion_general'] <= 2 and row['amistades_escuela'] <= 2:
            return 'aislamiento_social'
        elif row['responsabilidades'] >= 4:
            return 'sobrecarga_responsabilidades'
        elif row['valor_educacion'] <= 2:
            return 'falta_motivación'
        elif row['apoyo_familiar'] <= 1:
            return 'problemas_familiares'
        else:
            return 'multifactorial'
    else:
        return 'ninguno'

def entrenar_modelo_causa():
    try:
        # Verificar si existe el archivo de datos
        if not os.path.exists("datos_estudiantes.csv"):
            raise FileNotFoundError("No se encontró el archivo datos_estudiantes.csv")
        
        # Cargar datos
        df = pd.read_csv("datos_estudiantes.csv")
        
        if df.empty:
            raise ValueError("El archivo CSV está vacío")
        
        # Mapear variables categóricas
        mapeos = {
            'emocion_general': {'bien': 5, 'indiferente': 3, 'mal': 1},
            'amistades_escuela': {'si': 5, 'no': 1},
            'responsabilidades': {'si': 1, 'no': 5}
        }
        df.replace(mapeos, inplace=True)

        # Convertir y limpiar datos
        columnas = [
            'comprension', 'emocion_general', 'estres_estudios',
            'apoyo_familiar', 'amistades_escuela', 'relacion_profesores',
            'responsabilidades', 'valor_educacion', 'probabilidad_terminar'
        ]
        
        # Verificar columnas existentes
        columnas_disponibles = [col for col in columnas if col in df.columns]
        if len(columnas_disponibles) < 5:
            raise ValueError(f"Solo {len(columnas_disponibles)} columnas disponibles de {len(columnas)} requeridas")
        
        df[columnas_disponibles] = df[columnas_disponibles].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=columnas_disponibles)
        
        if df.empty:
            raise ValueError("No hay datos válidos después de la limpieza")
        
        # Crear categorías específicas
        df['causa_desercion'] = df.apply(categorizar_causa, axis=1)
        
        # Verificar que tenemos múltiples clases
        if len(df['causa_desercion'].unique()) < 2:
            raise ValueError("Se necesitan al menos 2 clases para entrenamiento")

        # Entrenar modelo
        X = df[columnas_disponibles]
        y = df['causa_desercion']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if len(X_train) == 0:
            raise ValueError("No hay datos de entrenamiento después de la división")
        
        # Escalar datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Crear y entrenar modelo
        modelo = LogisticRegression(
            multi_class='multinomial', 
            max_iter=2000, 
            class_weight='balanced',
            random_state=42
        )
        modelo.fit(X_train_scaled, y_train)

        # Evaluar modelo
        train_score = modelo.score(X_train_scaled, y_train)
        X_test_scaled = scaler.transform(X_test)
        test_score = modelo.score(X_test_scaled, y_test)
        
        print(f" Modelo entrenado - Precisión entrenamiento: {train_score:.2f}, prueba: {test_score:.2f}")
        print(" Distribución de clases:")
        print(y.value_counts())
        print(" Categorías disponibles:", modelo.classes_)

        # Guardar modelos
        joblib.dump(modelo, 'modelo_entrenado.pkl')
        joblib.dump(scaler, 'escalador.pkl')
        joblib.dump(modelo.classes_, 'clases_modelo.pkl')
        
        print(" Modelos guardados correctamente")
        
    except Exception as e:
        print(f" Error al entrenar modelo: {str(e)}")
        # Limpiar modelos en caso de error
        for file in ['modelo_entrenado.pkl', 'escalador.pkl', 'clases_modelo.pkl']:
            if os.path.exists(file):
                os.remove(file)
        raise  # Re-lanzar la excepción para manejo externo

if __name__ == '__main__':
    entrenar_modelo_causa()