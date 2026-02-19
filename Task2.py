#Task 2 – Neural Networks Integration
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("./final_data_colors.csv")


df["costo"] = df["label"].map({"Green": 3, "Blue": 10, "Grey": 1, "Yellow": 5}) 
#tenemos que preguntar si se elimina o no jiji
df = df.dropna(subset=["costo"])

# Mapear etiquetas a clases numéricas (para clasificación)
class_map = {
    "Green": 0,
    "Blue": 1,
    "Grey": 2,
    "Yellow": 3
}

df["class"] = df["label"].map(class_map)

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

y_onehot = one_hot(y, 4)

#dividir el data set entre los datos de prueba y de test
X_train, X_test, y_train, y_test = train_test_split( X, y_onehot, test_size=0.2, random_state=42)

#inicialización de la red
input_size = 3
hidden_size = 16
output_size = 4

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
