from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
import joblib
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
CSV_FILE = 'datos_estudiantes.csv'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Configuración
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargar modelo, escalador y clases si existen
if all(os.path.exists(f) for f in ['modelo_entrenado.pkl', 'escalador.pkl', 'clases_modelo.pkl']):
    modelo = joblib.load('modelo_entrenado.pkl')
    escalador = joblib.load('escalador.pkl')
    clases_modelo = joblib.load('clases_modelo.pkl')
else:
    modelo = None
    escalador = None
    clases_modelo = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calcular_probabilidad_abandono(entrada):
    entrada_esc = escalador.transform(entrada)
    probabilidades = modelo.predict_proba(entrada_esc)[0]
    
    if 'ninguno' in clases_modelo:
        prob_ninguno = probabilidades[np.where(clases_modelo == 'ninguno')[0][0]]
        return round((1 - prob_ninguno) * 100, 2), modelo.predict(entrada_esc)[0], round(max(probabilidades) * 100, 2)
    else:
        return round((1 - max(probabilidades)) * 100, 2), modelo.predict(entrada_esc)[0], round(max(probabilidades) * 100, 2)

def procesar_estudiantes(df):
    estudiantes_analizados = []
    
    if modelo and escalador and clases_modelo is not None:
        columnas_numericas = [
            'comprension', 'emocion_general', 'estres_estudios',
            'apoyo_familiar', 'amistades_escuela', 'relacion_profesores',
            'responsabilidades', 'valor_educacion', 'probabilidad_terminar'
        ]
        
        for col in columnas_numericas:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=columnas_numericas)
        
        for index, row in df.iterrows():
            entrada = [[
                row['comprension'],
                row['emocion_general'],
                row['estres_estudios'],
                row['apoyo_familiar'],
                row['amistades_escuela'],
                row['relacion_profesores'],
                row['responsabilidades'],
                row['valor_educacion'],
                row['probabilidad_terminar']
            ]]

            try:
                prob_abandono, motivo, confianza = calcular_probabilidad_abandono(entrada)
                riesgo_alto = row['probabilidad_terminar'] <= 2
                
                estudiantes_analizados.append({
                    'id': index,
                    'nombre': row.get('nombre', f'Estudiante {index}'),
                    'probabilidad_abandono': prob_abandono,
                    'riesgo_alto': riesgo_alto,
                    'motivo_principal': motivo,
                    'confianza': confianza,
                    'fecha_registro': row.get('fecha_registro', 'N/A')
                })
            except Exception as e:
                print(f"Error procesando estudiante {index}: {str(e)}")
    
    estudiantes_analizados.sort(key=lambda x: x['probabilidad_abandono'], reverse=True)
    return estudiantes_analizados

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/formulario')
def formulario():
    return render_template('formulario.html')

@app.route('/resultados', methods=['POST'])
def resultados():
    datos = {
        'nombre': request.form.get('nombre', ''),
        'comprension': request.form.get('comprension'),
        'emocion_general': request.form.get('emocion_general'),
        'estres_estudios': request.form.get('estres_estudios'),
        'apoyo_familiar': request.form.get('apoyo_familiar'),
        'amistades_escuela': request.form.get('amistades_escuela'),
        'relacion_profesores': request.form.get('relacion_profesores'),
        'responsabilidades': request.form.get('responsabilidades'),
        'valor_educacion': request.form.get('valor_educacion'),
        'probabilidad_terminar': request.form.get('probabilidad_terminar'),
        'fecha_registro': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Inicializar variables de resultado
    datos.update({
        'probabilidad_abandono': 'N/A',
        'motivo_principal': 'No evaluado',
        'confianza_principal': 'N/A',
        'motivos_secundarios': [],
        'riesgo_alto': False,
        'error': None
    })

    # Mapear variables categóricas a numéricas
    mapeos = {
        'emocion_general': {'bien': 5, 'indiferente': 3, 'mal': 1},
        'amistades_escuela': {'si': 5, 'no': 1},
        'responsabilidades': {'si': 1, 'no': 5}
    }
    for key, mapping in mapeos.items():
        datos[key] = mapping.get(datos[key], 3)  # 3 como valor neutral si falta

    # Guardar datos en CSV
    try:
        if os.path.isfile(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            df = pd.concat([df, pd.DataFrame([datos])], ignore_index=True)
        else:
            df = pd.DataFrame([datos])
        df.to_csv(CSV_FILE, index=False)
    except Exception as e:
        datos['error'] = f"Error al guardar datos: {str(e)}"

    # Evaluar causa de abandono con el modelo si está disponible
    if modelo and escalador and clases_modelo is not None:
        try:
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

            prob_abandono, prediccion, confianza = calcular_probabilidad_abandono(entrada)
            
            # Obtener top 3 causas (excluyendo 'ninguno' si existe)
            probabilidades = modelo.predict_proba(entrada)[0]
            top_3_indices = np.argsort(probabilidades)[-3:][::-1]
            causas_predichas = [
                (clases_modelo[i], round(probabilidades[i]*100, 2))
                for i in top_3_indices if 'ninguno' not in str(clases_modelo[i]).lower()
            ]

            datos.update({
                'probabilidad_abandono': prob_abandono,
                'motivo_principal': prediccion if 'ninguno' not in str(prediccion).lower() else 'Bajo riesgo',
                'confianza_principal': confianza,
                'motivos_secundarios': causas_predichas[1:] if len(causas_predichas) > 1 else [],
                'riesgo_alto': float(datos['probabilidad_terminar']) <= 2
            })

        except ValueError as e:
            datos['error'] = f"Error en valores numéricos: {str(e)}"
        except Exception as e:
            datos['error'] = f"Error en el modelo: {str(e)}"
    else:
        datos['error'] = "Modelo no disponible"

    return render_template('resultados.html', **datos)

@app.route('/dashboard')
def dashboard():
    if not os.path.isfile(CSV_FILE):
        return render_template('dashboard.html', 
                            total_estudiantes=0, 
                            riesgo=0, 
                            porcentaje_riesgo=0, 
                            causas={},
                            causas_detalladas={})

    try:
        df = pd.read_csv(CSV_FILE)

        # Convertir columnas numéricas
        columnas_numericas = [
            'comprension', 'emocion_general', 'estres_estudios',
            'apoyo_familiar', 'amistades_escuela', 'relacion_profesores',
            'responsabilidades', 'valor_educacion', 'probabilidad_terminar'
        ]
        df[columnas_numericas] = df[columnas_numericas].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=columnas_numericas)

        # Identificar estudiantes en riesgo
        riesgo_df = df[df['probabilidad_terminar'] <= 2]
        porcentaje_riesgo = round(len(riesgo_df) / len(df) * 100, 2) if len(df) > 0 else 0

        # Causas básicas
        causas_basicas = {
            'Estrés alto': len(riesgo_df[riesgo_df['estres_estudios'] >= 4]),
            'Bajo apoyo familiar': len(riesgo_df[riesgo_df['apoyo_familiar'] <= 2]),
            'Mala relación con profesores': len(riesgo_df[riesgo_df['relacion_profesores'] <= 2]),
            'Baja comprensión': len(riesgo_df[riesgo_df['comprension'] <= 2])
        }

        # Causas detalladas
        causas_detalladas = {}
        if modelo and 'causa_desercion' in df.columns:
            causas_detalladas = df[df['probabilidad_terminar'] <= 2]['causa_desercion'].value_counts().to_dict()

        return render_template('dashboard.html',
                            porcentaje_riesgo=porcentaje_riesgo,
                            causas=causas_basicas,
                            causas_detalladas=causas_detalladas,
                            total_estudiantes=len(df),
                            riesgo=len(riesgo_df))
    except Exception as e:
        return render_template('dashboard.html',
                            total_estudiantes=0,
                            riesgo=0,
                            porcentaje_riesgo=0,
                            causas={},
                            causas_detalladas={},
                            error=f"Error al procesar datos: {str(e)}")

@app.route('/analisis_grupal', methods=['GET', 'POST'])
def analisis_grupal():
    if request.method == 'POST':
        if 'archivo_csv' not in request.files:
            return render_template('analisis_grupal.html', 
                                estudiantes=[], 
                                total_estudiantes=0,
                                mensaje="No se seleccionó ningún archivo")
        
        file = request.files['archivo_csv']
        
        if file.filename == '':
            return render_template('analisis_grupal.html', 
                                estudiantes=[], 
                                total_estudiantes=0,
                                mensaje="No se seleccionó ningún archivo")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                df = pd.read_csv(filepath)
                estudiantes_analizados = procesar_estudiantes(df)
                os.remove(filepath)
                
                return render_template('analisis_grupal.html',
                                    estudiantes=estudiantes_analizados,
                                    total_estudiantes=len(estudiantes_analizados),
                                    mensaje=None)
            except Exception as e:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return render_template('analisis_grupal.html', 
                                    estudiantes=[], 
                                    total_estudiantes=0,
                                    mensaje=f"Error al procesar archivo: {str(e)}")
    
    # Método GET - Mostrar datos existentes
    if not os.path.isfile(CSV_FILE):
        return render_template('analisis_grupal.html', 
                            estudiantes=[], 
                            total_estudiantes=0,
                            mensaje="No hay datos de estudiantes registrados")

    try:
        df = pd.read_csv(CSV_FILE)
        estudiantes_analizados = procesar_estudiantes(df)
        
        return render_template('analisis_grupal.html',
                            estudiantes=estudiantes_analizados,
                            total_estudiantes=len(estudiantes_analizados),
                            mensaje=None)
    except Exception as e:
        return render_template('analisis_grupal.html',
                            estudiantes=[],
                            total_estudiantes=0,
                            mensaje=f"Error al procesar datos: {str(e)}")

@app.route('/exportar_analisis')
def exportar_analisis():
    if not os.path.isfile(CSV_FILE):
        return "No hay datos para exportar", 404

    try:
        df = pd.read_csv(CSV_FILE)
        export_path = 'analisis_estudiantes_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'
        df.to_csv(export_path, index=False)
        return f"Análisis exportado correctamente a {export_path}", 200
    except Exception as e:
        return f"Error al exportar: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)