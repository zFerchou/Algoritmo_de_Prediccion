import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

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
    df = pd.read_csv("datos_estudiantes.csv")
    
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
    df[columnas] = df[columnas].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    # Crear categorías específicas
    df['causa_desercion'] = df.apply(categorizar_causa, axis=1)

    # Entrenar modelo
    X = df[columnas]
    y = df['causa_desercion']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = LogisticRegression(multi_class='multinomial', max_iter=2000, class_weight='balanced')
    modelo.fit(X_scaled, y)

    # Guardar
    joblib.dump(modelo, 'modelo_entrenado.pkl')
    joblib.dump(scaler, 'escalador.pkl')
    joblib.dump(modelo.classes_, 'clases_modelo.pkl')  # Guardar las clases
    
    print("✅ Modelo entrenado con categorías específicas")
    print("Categorías disponibles:", modelo.classes_)

if __name__ == '__main__':
    entrenar_modelo_causa()