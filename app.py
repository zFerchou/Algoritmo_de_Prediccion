from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__)
CSV_FILE = 'datos_estudiantes.csv'

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

    # Pasar datos al template resultados.html
    return render_template('resultados.html', **datos)

@app.route('/dashboard')
def dashboard():
    if not os.path.isfile(CSV_FILE):
        # No hay datos aún
        return render_template('dashboard.html', total_estudiantes=0, riesgo=0, porcentaje_riesgo=0, causas={})

    df = pd.read_csv(CSV_FILE)

    # Convertir columnas a numéricas para análisis
    for col in ['probabilidad_terminar', 'estres_estudios', 'apoyo_familiar', 'relacion_profesores', 'comprension']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    riesgo_df = df[df['probabilidad_terminar'] <= 2]  # estudiantes en riesgo

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
