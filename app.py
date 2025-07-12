from flask import Flask, render_template, request
import pandas as pd
import os
import joblib

app = Flask(__name__)
CSV_FILE = 'datos_estudiantes.csv'

# Cargar modelo y escalador si existen
if os.path.exists('modelo_entrenado.pkl') and os.path.exists('escalador.pkl'):
    modelo = joblib.load('modelo_entrenado.pkl')  # Este debe ser un modelo multiclase
    escalador = joblib.load('escalador.pkl')
else:
    modelo = None
    escalador = None

@app.route('/formulario')
def formulario():
    return render_template('formulario.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/resultados', methods=['POST'])
def resultados():
    # Recolectar datos del formulario
    datos = {
        'comprension': request.form.get('comprension'),
        'emocion_general': request.form.get('emocion_general'),
        'estres_estudios': request.form.get('estres_estudios'),
        'apoyo_familiar': request.form.get('apoyo_familiar'),
        'amistades_escuela': request.form.get('amistades_escuela'),
        'relacion_profesores': request.form.get('relacion_profesores'),
        'responsabilidades': request.form.get('responsabilidades'),
        'valor_educacion': request.form.get('valor_educacion'),
        'probabilidad_terminar': request.form.get('probabilidad_terminar')
    }

    # Guardar datos en CSV
    if os.path.isfile(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        df = pd.concat([df, pd.DataFrame([datos])], ignore_index=True)
    else:
        df = pd.DataFrame([datos])

    df.to_csv(CSV_FILE, index=False)

    # Evaluar causa de abandono con el modelo si está disponible
    if modelo and escalador:
        entrada = [[
            float(datos['comprension']),
            float(datos['emocion_general']),
            float(datos['estres_estudios']),
            float(datos['apoyo_familiar']),
            float(datos['amistades_escuela']),
            float(datos['relacion_profesores']),
            float(datos['responsabilidades']),
            float(datos['valor_educacion']),
            float(datos['probabilidad_terminar'])
        ]]

        entrada_esc = escalador.transform(entrada)

        prediccion = modelo.predict(entrada_esc)[0]
        probabilidades = modelo.predict_proba(entrada_esc)[0]
        confianza = round(max(probabilidades) * 100, 2)

        datos['motivo_predicho'] = prediccion  # e.g., "depresión", "economía", etc.
        datos['confianza'] = confianza
    else:
        datos['motivo_predicho'] = 'Modelo no entrenado'
        datos['confianza'] = 'N/A'

    return render_template('resultados.html', **datos)

@app.route('/dashboard')
def dashboard():
    if not os.path.isfile(CSV_FILE):
        return render_template('dashboard.html', total_estudiantes=0, riesgo=0, porcentaje_riesgo=0, causas={})

    df = pd.read_csv(CSV_FILE)

    for col in ['probabilidad_terminar', 'estres_estudios', 'apoyo_familiar', 'relacion_profesores', 'comprension']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    riesgo_df = df[df['probabilidad_terminar'] <= 2]
    porcentaje_riesgo = round(len(riesgo_df) / len(df) * 100, 2) if len(df) > 0 else 0

    causas = {
        'Estrés alto': len(riesgo_df[riesgo_df['estres_estudios'] >= 4]),
        'Bajo apoyo familiar': len(riesgo_df[riesgo_df['apoyo_familiar'] <= 2]),
        'Mala relación con profesores': len(riesgo_df[riesgo_df['relacion_profesores'] <= 2]),
        'Baja comprensión': len(riesgo_df[riesgo_df['comprension'] <= 2])
    }

    return render_template('dashboard.html',
                           porcentaje_riesgo=porcentaje_riesgo,
                           causas=causas,
                           total_estudiantes=len(df),
                           riesgo=len(riesgo_df))

if __name__ == '__main__':
    app.run(debug=True)
