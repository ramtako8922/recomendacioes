import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar los datos
df = pd.read_csv('turismo_data.csv')

# Preprocesamiento y selección de características
df = df.dropna()
df = df.drop(['Traveler name', 'Country', 'Humidity (Porcentage)', 'Traveler nationality', 'Trip ID', 'Start date', 'End date'], axis=1)

# Mapeo de ciudades
ciudades = ['London', 'Phuket', 'Bali', 'New York', 'Tokyo', 'Paris', 'Sydney', 'Rio de Janeiro', 'Amsterdam', 'Dubai', 'Cancun', 'Barcelona', 'Honolulu', 'Berlin', 'Marrakech', 'Edinburgh', 'Rome', 'Bangkok', 'Santorini', 'Cairo', 'Mexico', 'Madrid', 'Vancouver', 'Seoul', 'Los Angeles', 'Cape Town', 'Auckland', 'Phnom Penh', 'Athens', 'Krabi', 'Hawaii']
ciudades_unicas = list(set(ciudades))
mapeo_ciudades = {ciudad: i + 1 for i, ciudad in enumerate(ciudades_unicas)}

# Aplicar el mapeo
df['Destination'] = df['Destination'].map(mapeo_ciudades)

# Verificar si hay valores NaN en Destination
if df['Destination'].isna().sum() > 0:
    print("Existen valores NaN en la columna 'Destination' después del mapeo.")
    print("Valores problemáticos:")
    print(df[df['Destination'].isna()])
    df = df.dropna(subset=['Destination'])

# Mapeo de meses
mapeo_mes = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
df['Month'] = df['Month'].map(mapeo_mes)

# Mapeo de tipo de alojamiento y transporte
mapeo_accommodation = {'Airbnb': 1, 'Guesthouse': 2, 'Hostel': 3, 'Hotel': 4, 'Resort': 5, 'Riad': 6, 'Vacation rental': 7, 'Villa': 8}
df['Accommodation type'] = df['Accommodation type'].map(mapeo_accommodation)

mapeo_transport = {'Airplane': 1, 'Bus': 2, 'Car': 3, 'Car rental': 4, 'Ferry': 5, 'Subway': 6, 'Train': 7}
df['Transportation type'] = df['Transportation type'].map(mapeo_transport)

# Codificación binaria para el género
df['Traveler gender'] = df['Traveler gender'].map({'Male': 0, 'Female': 1})

# Asegurar que las nuevas características estén en el DataFrame
if 'Traveler age' not in df.columns:
    df['Traveler age'] = 0  # Valor predeterminado si no existe en los datos

if 'Transportation cost' not in df.columns:
    df['Transportation cost'] = 0  # Valor predeterminado en caso de ausencia

if 'Temperature Centigrade' not in df.columns:
    df['Temperature Centigrade'] = 25  # Valor predeterminado en caso de ausencia
if 'Duration' not in df.columns:
    df['Duration'] = 0  # Valor predeterminado en caso de ausencia

# Separación de características y destino
x = df.drop('Destination', axis=1)
y = df['Destination']

# Verificar valores NaN en X o y
if x.isna().sum().sum() > 0 or y.isna().sum() > 0:
    print("Existen valores NaN en las características o etiquetas.")
    print("NaN en X:", x.isna().sum())
    print("NaN en y:", y.isna().sum())
    x = x.dropna()
    y = y.dropna()

# Entrenamiento del modelo
x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
modelo = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
modelo.fit(x_train, y_train)

# Guardar el modelo y el mapeo de ciudades
joblib.dump((modelo, mapeo_ciudades), "modelo_recomendacion.pkl")

# Evaluación del modelo
predicciones = modelo.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predicciones):.2f}")

# Definir las características de entrada
entrada_usuario = {
    'Month': 3,  # Marzo
    'Accommodation type': 5,  # Resort
    'Transportation type': 1,  # Airplane
    'Traveler gender': 0,  # Hombre
    'Traveler age': 23,  # Edad del viajero
    'Transportation cost': 50,  # Costo del transporte
    'Temperature Centigrade': 23,  # Temperatura en grados centígrados
    'Duration': 6 # Dias de duracion del viaje
}

# Crear un DataFrame con todas las columnas de x_train
entrada_df = pd.DataFrame([entrada_usuario])

# Asegurar que las columnas coincidan con las de x_train
for col in x_train.columns:
    if col not in entrada_df.columns:
        entrada_df[col] = 0  # Añadir columnas faltantes con valor 0

# Reordenar las columnas para que coincidan con el modelo
entrada_df = entrada_df[x_train.columns]

# Realizar la predicción
prediccion = modelo.predict(entrada_df)

# Mapear la predicción al nombre de la ciudad
ciudad_recomendada = {v: k for k, v in mapeo_ciudades.items()}
ciudad_predicha = ciudad_recomendada[prediccion[0]]

print(f"Ciudad recomendada: {ciudad_predicha}")
