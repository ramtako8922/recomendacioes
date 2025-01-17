import joblib
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Servir archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Cargar el modelo y el mapeo de ciudades
modelo, mapeo_ciudades = joblib.load('modelo_recomendacion.pkl')

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    Month: int = Form(...),
    Accommodation_type: int = Form(...),
    Transportation_type: int = Form(...),
    Traveler_gender: int = Form(...),
    Traveler_age: int = Form(...),
    Transportation_cost: float = Form(...),
    Temperature_Centigrade: float = Form(...),
    Duration: int = Form(...)
):
    # Definir las características de entrada
    entrada_usuario = {
        'Month': Month,
        'Accommodation type': Accommodation_type,
        'Transportation type': Transportation_type,
        'Traveler gender': Traveler_gender,
        'Traveler age': Traveler_age,
        'Transportation cost': Transportation_cost,
        'Temperature Centigrade': Temperature_Centigrade,
        'Duration': Duration
    }

    # Crear un DataFrame con las características de entrada
    entrada_df = pd.DataFrame([entrada_usuario])

    # Asegurar que las columnas coincidan con las del modelo
    for col in modelo.feature_names_in_:
        if col not in entrada_df.columns:
            entrada_df[col] = 0  # Añadir columnas faltantes con valor 0

    # Reordenar las columnas para que coincidan con el modelo
    entrada_df = entrada_df[modelo.feature_names_in_]

    # Realizar la predicción
    prediccion = modelo.predict(entrada_df)

    # Mapear la predicción al nombre de la ciudad
    ciudad_recomendada = {v: k for k, v in mapeo_ciudades.items()}
    ciudad_predicha = ciudad_recomendada.get(prediccion[0], "Ciudad desconocida")
    # Devolver la respuesta en HTML con la ciudad recomendada
    
    return templates.TemplateResponse("form.html", {"request": request, "ciudad_recomendada": ciudad_predicha})

if __name__ == '__main__':
    import uvicorn
    # uvicorn.run(app, host="127.0.0.1", port=8000)