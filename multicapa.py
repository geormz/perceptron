import numpy as np
import tensorflow as tf

# Datos de entrenamiento y prueba
X_entrenamiento = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
y_entrenamiento = np.array([[0], [1], [1], [0]])
X_prueba = X_entrenamiento
y_prueba = y_entrenamiento

# Crear el modelo de red neuronal
modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),   # Capa de entrada con 2 neuronas
    tf.keras.layers.Dense(2, activation='relu'),  # Capa oculta con 2 neuronas y función de activación ReLU
    tf.keras.layers.Dense(1, activation='sigmoid')  # Capa de salida con 1 neurona y función de activación sigmoide
])

# Compilar el modelo
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
modelo.fit(X_entrenamiento, y_entrenamiento, epochs=10, verbose=1)

# Evaluar el modelo en los datos de prueba
pérdida, precisión = modelo.evaluate(X_prueba, y_prueba)
print(f'Precisión en los datos de prueba: {precisión * 100:.2f}%')
