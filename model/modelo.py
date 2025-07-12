# modelo_causa.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

CSV_FILE = 'datos_estudiantes.csv'

def entrenar_modelo_causa():
    df = pd.read_csv(CSV_FILE)
    
    # Reemplazar espacios en los nombres de columnas por guiones bajos
    df.columns = df.columns.str.replace(' ', '_')

    columnas = ['comprension', 'estres_estudios', 'apoyo_familiar', 'relacion_profesores']
    df[columnas] = df[columnas].apply(pd.to_numeric, errors='coerce')

    # Asegurarse que existe la columna 'causa_desercion'
    if 'causa_desercion' not in df.columns:
        raise ValueError("La columna 'causa_desercion' no existe en el CSV.")

    # Eliminar datos faltantes
    df = df.dropna(subset=columnas + ['causa_desercion'])

    X = df[columnas]
    y = df['causa_desercion']  # Valores como: 'depresión', 'economía', 'tristeza', 'se queda'

    # Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Modelo multiclase
    modelo = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    modelo.fit(X_scaled, y)

    # Guardar modelo y escalador
    joblib.dump(modelo, 'modelo_entrenado_causa.pkl')
    joblib.dump(scaler, 'escalador_causa.pkl')

    print("✅ Modelo de causa entrenado y guardado correctamente.")

if __name__ == '__main__':
    entrenar_modelo_causa()
