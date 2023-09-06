import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Definir la función del perceptrón
def perceptron(tamaño_entrada, tasa_aprendizaje, épocas, X_entrenamiento, y_entrenamiento):
    pesos = np.random.rand(tamaño_entrada)
    sesgo = np.random.rand()

    for época in range(épocas):
        errores = 0
        for i in range(len(X_entrenamiento)):
            entrada_neta = np.dot(X_entrenamiento[i], pesos) + sesgo
            salida = 1 if entrada_neta > 0 else 0
            error = y_entrenamiento[i] - salida
            if error != 0:
                errores += 1
                pesos += tasa_aprendizaje * error * X_entrenamiento[i]
                sesgo += tasa_aprendizaje * error
        if errores == 0:
            print(f'Entrenamiento completado en la época {época + 1}')
            break
    return pesos, sesgo

# Función para probar el perceptrón entrenado
def probar_perceptron(pesos, sesgo, X_prueba, y_prueba):
    predicciones_correctas = 0
    for i in range(len(X_prueba)):
        entrada_neta = np.dot(X_prueba[i], pesos) + sesgo
        salida = 1 if entrada_neta > 0 else 0
        if salida == y_prueba[i]:
            predicciones_correctas += 1
    precisión = (predicciones_correctas / len(X_prueba)) * 100
    return precisión

# Leer los datos de entrenamiento y prueba desde archivos CSV
datos_entrenamiento = pd.read_csv('XOR_trn.csv', header=None)
datos_prueba = pd.read_csv('XOR_tst.csv', header=None)

X_entrenamiento = datos_entrenamiento.iloc[:, :-1].values
y_entrenamiento = datos_entrenamiento.iloc[:, -1].values

X_prueba = datos_prueba.iloc[:, :-1].values
y_prueba = datos_prueba.iloc[:, -1].values

# Parámetros de entrenamiento
tamaño_entrada = X_entrenamiento.shape[1]
tasa_aprendizaje = float(input('Ingrese la tasa de aprendizaje: '))
máximas_épocas = int(input('Ingrese el número máximo de épocas de entrenamiento: '))

# Entrenar el perceptrón
pesos, sesgo = perceptron(tamaño_entrada, tasa_aprendizaje, máximas_épocas, X_entrenamiento, y_entrenamiento)

# Probar el perceptrón entrenado
precisión = probar_perceptron(pesos, sesgo, X_prueba, y_prueba)
print(f'Precisión en los datos de prueba: {precisión:.5f}%')

# Mostrar gráficamente los patrones y la recta que los separa
if tamaño_entrada == 2:
    plt.scatter(X_entrenamiento[:, 0], X_entrenamiento[:, 1], c=y_entrenamiento)
    valores_x = np.linspace(-1.5, 1.5, 100)
    valores_y = -(pesos[0] / pesos[1]) * valores_x - (sesgo / pesos[1])
    plt.plot(valores_x, valores_y, label='línea de separación', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Problema XOR - Línea de separación del perceptrón')
    plt.show()
else:
    print('La visualización gráfica solo es compatible con 2 características.')
    