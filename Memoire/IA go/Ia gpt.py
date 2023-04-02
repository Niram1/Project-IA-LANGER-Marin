# Importation des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

# Fonction pour calculer l'image moyenne d'un chiffre donné
def average_digit(data, digit):
    # Sélectionner les échantillons correspondant au chiffre donné
    selected = data[data[:, 0] == digit][:, 1:]
    # Calculer la moyenne des échantillons sélectionnés
    return np.mean(selected, axis=0)

# Charger les données MNIST
mnist_data = MNIST()
X_train, y_train = mnist_data.load_training()
X_test, y_test = mnist_data.load_testing()

# Calculer l'image moyenne pour le chiffre "8"
average_eight = average_digit(np.array(X_train), 8)

# Afficher l'image moyenne pour le chiffre "8"
plt.imshow(average_eight.reshape(28, 28), cmap="gray")
plt.show()

# Fonction pour effectuer une prédiction simple en utilisant la sigmoïde
def predict(x, W, b):
    return 1 / (1 + np.exp(-np.dot(W, x) - b))

# Fonction pour évaluer la précision du modèle simple
def evaluate(data, digit, threshold, W, b):
    predictions = [predict(x, W, b) for x in data[:, 1:]]
    correct = sum(1 for y, p in zip(data[:, 0], predictions) if (y == digit) == (p >= threshold))
    return correct / len(predictions)

# Évaluation de la précision du modèle simple
train_acc = evaluate(np.array(X_train), 8, 0.5, average_eight, -50)
test_acc = evaluate(np.array(X_test), 8, 0.5, average_eight, -50)
test_eight_acc = evaluate(np.array(X_test)[np.array(y_test) == 8], 8, 0.5, average_eight, -50)

print(f"Précision sur l'ensemble d'entraînement : {train_acc * 100:.2f}%")
print(f"Précision sur l'ensemble de test : {test_acc * 100:.2f}%")
print(f"Précision sur l'ensemble de test (uniquement les '8') : {test_eight_acc * 100:.2f}%")

# Fonctions pour charger et mettre en forme les données MNIST
def load_data():
    mnist_data = MNIST()
    X_train, y_train = mnist_data.load_training()
    X_test, y_test = mnist_data.load_testing()
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def reshape_data(X, y):
    X = X.reshape(len(X), -1).astype(np.float32) / 255
    y = np.eye(10)[y]
    return X, y

# Classe pour représenter l'erreur quadratique moyenne
class MSE:
    @staticmethod
    def loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def gradient(y_true, y_pred):
        return -2 * (y_true - y_pred)

# Classe pour représenter un réseau neuronal séquentiel
class SequentialNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

        # Entraînement du réseau neuronal
    def train(self, X, y, loss_function, epochs, learning_rate, batch_size):
        n_samples = len(X)
        for epoch in range(epochs):
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                # Propagation avant
                y_pred = self.forward_pass(X_batch)

                # Calcul de la perte
                loss = loss_function.loss(y_batch, y_pred)

                # Propagation arrière
                gradients = loss_function.gradient(y_batch, y_pred)
                self.backward_pass(gradients, learning_rate)

            # Afficher la perte pour chaque époque
            print(f"Époque {epoch + 1}/{epochs} - Perte : {loss:.4f}")

    # Propagation avant à travers le réseau
    def forward_pass(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    # Propagation arrière à travers le réseau
    def backward_pass(self, gradients, learning_rate):
        for layer in reversed(self.layers):
            gradients = layer.backward(gradients, learning_rate)

    # Prédiction à partir d'un ensemble d'échantillons
    def predict(self, X):
        return np.argmax(self.forward_pass(X), axis=1)

# Classe pour représenter une couche dense
class DenseLayer:
    def __init__(self, input_dim, output_dim, activation_function):
        self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2 / input_dim)
        self.b = np.zeros(output_dim)
        self.activation_function = activation_function

    # Propagation avant à travers la couche
    def forward(self, X):
        self.X = X
        self.Z = np.dot(X, self.W.T) + self.b
        return self.activation_function.activation(self.Z)

    # Propagation arrière à travers la couche
    def backward(self, gradients, learning_rate):
        dA = gradients * self.activation_function.gradient(self.Z)
        dW = np.dot(dA.T, self.X)
        self.W -= learning_rate * dW
        return np.dot(dA, self.W)

# Classe pour représenter la fonction d'activation ReLU
class ReLU:
    @staticmethod
    def activation(Z):
        return np.maximum(0, Z)

    @staticmethod
    def gradient(Z):
        return (Z > 0).astype(float)

# Charger et mettre en forme les données MNIST
X_train, X_test, y_train, y_test = load_data()
X_train, y_train = reshape_data(X_train, y_train)
X_test, y_test = reshape_data(X_test, y_test)

# Créer le réseau neuronal
network = SequentialNetwork()
network.add(DenseLayer(784, 128, ReLU()))
network.add(DenseLayer(128, 10, ReLU()))

# Entraîner le réseau neuronal
network.train(X_train, y_train, MSE(), epochs=20, learning_rate=0.001, batch_size=128)

# Évaluer la précision du modèle
train_acc = np.mean(network.predict(X_train) == np.argmax(y_train, axis=1))
test_acc = np.mean(network.predict(X_test) == np.argmax(y_test, axis=1))

print(f"Précision sur l'ensemble d'entraînement : {train_acc * 100:.2f}%")
print(f"Précision sur l'ensemble de test : {test_acc * 100:.2f}%")

# Fonction pour afficher les prédictions incorrectes
def display_incorrect_predictions(X, y_true, y_pred, num_images=10):
    incorrect_indices = np.where(y_true != y_pred)[0]
    random_indices = np.random.choice(incorrect_indices, num_images, replace=False)

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
    for idx, ax in zip(random_indices, axes.ravel()):
        ax.imshow(X[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"Vrai: {y_true[idx]}, Prédit: {y_pred[idx]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# Afficher les prédictions incorrectes
y_train_true = np.argmax(y_train, axis=1)
y_train_pred = network.predict(X_train)
display_incorrect_predictions(X_train, y_train_true, y_train_pred)

y_test_true = np.argmax(y_test, axis=1)
y_test_pred = network.predict(X_test)
display_incorrect_predictions(X_test, y_test_true, y_test_pred)

# Sauvegarder le modèle
network.save("mon_modele.h5")

# Charger un modèle sauvegardé
from keras.models import load_model
modele_charge = load_model("mon_modele.h5")

# Vérifier si le modèle chargé a les mêmes performances
modele_charge_acc = modele_charge.evaluate(X_test, y_test, verbose=0)[1]
print(f"Précision du modèle chargé sur l'ensemble de test : {modele_charge_acc * 100:.2f}%")
