"""Iris flower classifier using sklearn.datasets and MLPClassifier """
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Cargar el dataset de iris
iris = load_iris()

# Separar los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Crear el modelo MLP
model = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=200)

# Entrenar el modelo
model.fit(X_train, y_train)

# Predecir las etiquetas de las flores en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión total
accuracy = accuracy_score(y_test, y_pred)
print("Precisión total:", accuracy)

# Mostrar la precisión por categorías
target_names = iris.target_names
print(classification_report(y_test, y_pred, target_names=target_names))