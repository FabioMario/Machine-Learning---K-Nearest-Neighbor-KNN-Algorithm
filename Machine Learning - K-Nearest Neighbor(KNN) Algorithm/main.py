import pandas as pd

import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

#Importo el dataset
dataset = pd.read_csv('teleCust1000t.csv')

# Cuantas columnas y filas tiene el dataset
df = pd.read_csv("teleCust1000t.csv")
amount_of_columns = len(df.axes[1]) # Columnas
amount_of_rows = len(df.axes[0]) # Filas

# Splitea el dataset en 2 partes, 80% para train y 20% para test
X = dataset.iloc[:, 0:amount_of_columns-1].values # Es el dataset sin la ultima columna la cual no puede ser parte del training.
y = dataset.iloc[:, amount_of_columns-1].values # Es la ultima columna la cual es la que se quiere predecir.
test=0.2 # Se utilizan el 20% de los datos para testear.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test) # El dataset se divide en train y test.

# Scaling
sc = StandardScaler() # StandardScaler es una clase que permite estandarizar las variables. Estandarizar es poner todas las variables en la misma escala.
X_train = sc.fit_transform(X_train) # Se ajusta y transforma el dataset de train.
X_test = sc.transform(X_test) # Se transforma el dataset de test.

# Cantidad de vecinos la cual tiene que ser impar.
k = round(math.sqrt(len(y_test)))
if k % 2 == 0:
    k = k + 1

classifier = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2) # n_neighbors es la cantidad de vecinos (la cual es impar), metric es minowski lo cual es la distancia euclidiana, p=2 es la distancia euclidiana.
classifier.fit(X_train, y_train) # Los métodos de ajuste suelen ser responsables de numerosas operaciones. Por lo general, deberían comenzar por borrar los atributos ya almacenados en el estimador y luego realizar la validación de parámetros y datos. También son responsables de estimar los atributos a partir de los datos de entrada y almacenar los atributos del modelo y finalmente devolver el estimador ajustado.

# Se predice el dataset de test.
y_pred = classifier.predict(X_test)

# Se crea la matriz de confusion.
cm = confusion_matrix(y_test, y_pred)

# Accuracy
accuracy = accuracy_score(y_test, y_pred) # Se calcula la accuracy. Que es la cantidad de predicciones correctas sobre el total de predicciones.
print("Accuracy: ", accuracy)

# F1 Score
f1 = f1_score(y_test, y_pred, average='weighted') # Se calcula el f1 score. Que da una mejor medida de los casos clasificados incorrectamente que la métrica de precisión.
print("F1 Score: ", f1)


