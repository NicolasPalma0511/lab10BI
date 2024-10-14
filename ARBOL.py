import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn import tree
import matplotlib.pyplot as plt

data = pd.read_excel('Data10.xlsx')

label_encoder = LabelEncoder()
data['Estado'] = label_encoder.fit_transform(data['Estado'])
data['Tipo'] = label_encoder.fit_transform(data['Tipo'])

# Seleccionar características
X = data[['Precio actual', 'Precio final', 'Tipo']]
y = data['Estado']

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo de árbol de decisión
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Evaluar el modelo
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Visualizar el árbol de decisión
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=['Precio actual', 'Precio final', 'Tipo'], class_names=label_encoder.classes_, filled=True)
plt.show()
