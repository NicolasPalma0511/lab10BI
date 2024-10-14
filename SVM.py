import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

data = pd.read_excel('Data10.xlsx')

label_encoder = LabelEncoder()
data['Estado'] = label_encoder.fit_transform(data['Estado'])
data['Tipo'] = label_encoder.fit_transform(data['Tipo'])

# Seleccionar características
X = data[['Precio actual', 'Precio final', 'Tipo']]
y = data['Estado']

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalamiento de características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar el modelo SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Evaluar el modelo
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))

# Nuevo registro para hacer predicción
nuevo_dato = [[50, 18, label_encoder.transform(['a'])[0]]]  # Convertir 'a' a su valor numérico

# Escalar el nuevo dato
nuevo_dato_scaled = scaler.transform(nuevo_dato)

# Hacer predicción
prediccion = svm.predict(nuevo_dato_scaled)
estado_predicho = label_encoder.inverse_transform(prediccion)

print(f"Predicción para el nuevo registro: {estado_predicho[0]}")

