"""Iris flower classifier using iris.csv and MLP"""
import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

def indices_general(MC: confusion_matrix, nombres: numpy.ndarray):
    """Calculo de la precisión global y por categoria, y el error global de la matriz de confusión

    Args:
        MC (confusion_matrix): Matriz de confussion se muestran los aciertos y desaciertos de la prediccion por clasificación
        nombres (ndarray, optional): El array que contiene las diferentes clasificaciones. Defaults to None.

    Returns:
        tuples(tuples): Retorna una tupla de tuplas, donde cada una de estas contiene una descripcion de los datos calculados y dichos datos
    """

    accuracy_global = numpy.sum(MC.diagonal()) / numpy.sum(MC)
    err_global = 1 - accuracy_global
    accuracy_category = pandas.DataFrame(MC.diagonal()/numpy.sum(MC, axis=1)).T

    if not nombres.size:
        accuracy_category.colums = nombres
    
    return (("Matriz de confusión: ",MC), ("Precision Global: ", accuracy_global), ("Error Global", err_global), ("Precisión por categoría", accuracy_category))


if __name__ == "__main__":
    filename = "iris.csv"    
    datos = pandas.read_csv(filename, delimiter=";", decimal=".")
    """
    print(datos.shape)
    print(datos.head())
    print(datos.info())
    """
    # x es la variable predictoria
    # y es la variable a predecir
    x = datos.iloc[:,:4]
    y = datos.iloc[:,4:5]
    print(x,y, end="\n----------------\n")

    x_tarin, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=0)
    # print(x_tarin, x_test, y_train, y_test, sep="\n+++++++++++++\n")

    mlp = MLPClassifier(max_iter=1000)

    """print("---------------")
    print(y_train)
    print("---------------")
    print(y_train.iloc[:,0])"""

    mlp.fit(x_tarin, y_train.iloc[:,0])
    
    accuracy = mlp.score(x_test, y_test.iloc[:,0])

    print("La predicciones del Testing son: {}".format(mlp.predict(x_test)), end="\n----------------------------\n")

    MC = confusion_matrix(y_test, mlp.predict(x_test)) 

    resultados = indices_general(MC, numpy.unique(y))

    for i in resultados:
        print(f"{i[0]}\n{i[1]}")

    print(accuracy)