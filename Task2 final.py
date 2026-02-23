#Task 2 – Neural Networks Integration
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import heapq

from Task1final import (
    procesar_laberinto,
    ProblemaLaberinto,
    reconstruir_camino,
    heuristica_manhattan
)
 

df = pd.read_csv("./final_data_colors.csv")


df["costo"] = df["label"].map({"Green": 3, "Blue": 10, "Grey": 1, "Yellow": 5}) 
#tenemos que preguntar si se elimina o no jiji

# Mapear etiquetas a clases numéricas (para clasificación)
class_map = {
    "Green": 0,
    "Blue": 1,
    "Grey": 2,
    "Yellow": 3
}

def map_label(label):
    if label in class_map:
        return class_map[label]
    else:
        return 4  # Clase para peligro/desconocido
    
df["class"] = df["label"].map(map_label)

# Inputs 
X = df[["red", "green", "blue"]].values

# normalizar los valores de RGB
X = X / 255.0

# labels
y = df["class"].values



def one_hot(y, num_classes):
    onehot = np.zeros((len(y), num_classes))
    onehot[np.arange(len(y)), y] = 1
    return onehot

y_onehot = one_hot(df["class"].values, 5)

#dividir el data set entre los datos de prueba y de test
X_train, X_test, y_train, y_test = train_test_split( X, y_onehot, test_size=0.2, random_state=42)

#inicialización de la red
input_size = 3
hidden_size = 8
output_size = 5

np.random.seed(42) #semilla fija

#pesos
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))


#funciones de activación 
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)



def forward(X):
    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


#Loss
def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / m




def backward(X, y_true, Z1, A1, A2, lr=0.1):
    global W1, b1, W2, b2
    
    m = X.shape[0]
    
    dZ2 = A2 - y_true
    dW2 = A1.T.dot(dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dA1 = dZ2.dot(W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = X.T.dot(dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    # SGD update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2


#Entrenamiento de la red neuronal
epochs = 200
learning_rate = 0.05
#SGD
for epoch in range(epochs):
    for i in range(len(X_train)):
        xi = X_train[i:i+1]
        yi = y_train[i:i+1]
        
        Z1, A1, Z2, A2 = forward(xi)
        backward(xi, yi, Z1, A1, A2, learning_rate)
    
    if epoch % 20 == 0:
        _, _, _, A2_full = forward(X_train)
        loss = compute_loss(y_train, A2_full)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


#Test 20%
def accuracy(X, y_true):
    _, _, _, A2 = forward(X)
    preds = np.argmax(A2, axis=1)
    labels = np.argmax(y_true, axis=1)
    return np.mean(preds == labels)

acc = accuracy(X_test, y_test)
#Resultado final
print("Test Accuracy:", acc)

##################################################################
#Task 2.2


def reconstruir_imagen_semantica(matriz_colores, tile_size):
    filas, cols, _ = matriz_colores.shape
    
    img = Image.new("RGB", (cols * tile_size, filas * tile_size))
    pixels = img.load()
    
    for fila in range(filas):
        for col in range(cols):
            r, g, b = matriz_colores[fila, col]
            
            for i in range(tile_size):
                for j in range(tile_size):
                    x = col * tile_size + j
                    y = fila * tile_size + i
                    pixels[x, y] = (int(r), int(g), int(b))
    
    return img


class NodoAStar:
    def __init__(self, estado, padre=None, accion=None, costo=0, heuristica=0):
        self.estado = estado
        self.padre = padre
        self.accion = accion
        self.costo = costo  # g(n)
        self.heuristica = heuristica  # h(n)
        self.f = costo + heuristica  # f(n)

    def __lt__(self, other):
        return self.f < other.f
    

def obtener_costo(r, g, b):
    x_input = np.array([[r, g, b]]) / 255.0
    Z1 = x_input.dot(W1) + b1
    A1 = np.maximum (0, Z1)
    Z2= A1.dot(W2) + b2
    expZ2 = np.exp(Z2 - np.max(Z2))
    A2 = softmax(Z2)
    class_pred = np.argmax(A2)

    cost_map = {
        0: 3,   # Green
        1: 10,  # Blue
        2: 1,   # Grey
        3: 5 ,   # Yellow
        4: 15   # Peligroso/desconocido (costo alto)
    }

    return cost_map.get(class_pred, 15)  # Retorna el costo basado en la clase predicha, default a 1 si no se encuentra
    
    
    
    
def A_est(problema, matriz_colores):
    nodo_inicial = NodoAStar(
        problema.inicial(),
        costo=0,
        heuristica=heuristica_manhattan(
            problema.inicial(),
            problema.finales
        )
    )
    frontera = []
    heapq.heappush(frontera, nodo_inicial)

    explorados=set()
    costos_g={nodo_inicial.estado:0}


    while frontera:
        nodo=heapq.heappop(frontera)

        if problema.goalTest(nodo.estado):
            return reconstruir_camino(nodo)
        
        explorados.add(nodo.estado)

        for accion in problema.acciones(nodo.estado):
            estado_hijo=problema.resultado(nodo.estado, accion)

            r, g, b = matriz_colores[estado_hijo[0], estado_hijo[1]]
            nuevo_costo_g=nodo.costo + obtener_costo(r, g, b)

            if estado_hijo not in explorados:
                if estado_hijo not in costos_g or nuevo_costo_g <costos_g[estado_hijo]:
                    costos_g[estado_hijo]=nuevo_costo_g
                    h=heuristica_manhattan(estado_hijo, problema.finales)
                    nodo_hijo = NodoAStar(
                        estado_hijo,
                        padre=nodo,
                        accion=accion,
                        costo=nuevo_costo_g,
                        heuristica=h
                    )
                    heapq.heappush(frontera, nodo_hijo)
    return None


print("")
print("EJECUTANDO TASK 2.2 - A* SEMÁNTICO")

# Cargar imagen del laberinto semántico
imagen = "image.png"  
# imagen = "./2.2.jpeg"  

grid, inicio, metas, matriz_colores = procesar_laberinto(imagen, 10)

problema = ProblemaLaberinto(grid, inicio, metas, matriz_colores)

camino = A_est(problema, matriz_colores)

if camino:
    print(f"Camino encontrado con {len(camino)} pasos.")

    # Calcular costo total real
    costo_total = 0
    for estado in camino:
        r, g, b = matriz_colores[estado[0], estado[1]]
        costo_total += obtener_costo(r, g, b)

    print(f"Costo total del camino: {costo_total}")

    # Reconstruir imagen pixelada semántica (como Task 1)
    img_base = reconstruir_imagen_semantica(matriz_colores, tile_size=10)
    img_base.save("resultado_task2_procesado.png")
    img_base.show()

    pixels = img_base.load()

    # Dibujar camino encima de la versión reconstruida
    for (fila, col) in camino:
        for i in range(10):
            for j in range(10):
                x = col * 10 + j
                y = fila * 10 + i
                pixels[x, y] = (255, 0, 0)  # rojo

    img_base.save("resultado_task2_semantico.png")
    img_base.show()

else:
    print("No se encontró camino.")