import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_excel('Data10.xlsx')

data['Tipo'] = LabelEncoder().fit_transform(data['Tipo'])

# Seleccionar características
X = data[['Precio actual', 'Precio final']]

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encontrar el número óptimo de clusters (método del codo)
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Graficar el método del codo
plt.plot(range(1, 11), inertia)
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.title('Método del codo')
plt.show()

# Aplicar K-Means con el número de clusters elegido (ej: 3)
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualizar resultados
print(data[['Numero', 'Precio actual', 'Precio final', 'Cluster']])

# Graficar los datos agrupados
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=data['Cluster'], cmap='viridis', marker='o')

# Añadir etiquetas y título
plt.xlabel('Precio actual (escalado)')
plt.ylabel('Precio final (escalado)')
plt.title('Agrupación K-Means de los Datos')
plt.colorbar(label='Cluster')
plt.show()
