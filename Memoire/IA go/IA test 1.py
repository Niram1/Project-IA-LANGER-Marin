import gzip
import numpy as np


def encode_label(j):  # <1>
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# <1> Nous utilisons l'encodage one-hot pour les indices en vecteurs de longueur 10.


def shape_data(data):
    features = [np.reshape(x, (784, 1)) for x in data[0]]  # <1>

    labels = [encode_label(y) for y in data[1]]  # <2>

    return zip(features, labels)  # <3>

# <1> Nous aplatissons les images d'entrée en vecteurs de caractéristiques de longueur 784.
# <2> Tous les labels sont encodés en one-hot.
# <3> Ensuite, nous créons des paires de caractéristiques et de labels.


def load_data_impl():
    # Définit le chemin du fichier contenant les données MNIST
    path = 'C:\\Users\\Marin\\Documents\\Université\\Memoire\\IA go\\mnist.npz'
    
    # Charge le fichier en mémoire en utilisant numpy
    f = np.load(path)
    
    # Extrait les ensembles de données d'entraînement (x_train et y_train)
    x_train, y_train = f['x_train'], f['y_train']
    
    # Extrait les ensembles de données de test (x_test et y_test)
    x_test, y_test = f['x_test'], f['y_test']
    
    # Ferme le fichier
    f.close()
    
    # Retourne les données d'entraînement et de test sous forme de tuples
    return (x_train, y_train), (x_test, y_test)

def load_data():
    train_data, test_data = load_data_impl()
    return shape_data(train_data), shape_data(test_data)

# <4> Décompresser et charger les données MNIST génère trois ensembles de données.
# <5> Nous ne conservons pas les données de validation ici et redimensionnons les deux autres ensembles de données.


import numpy as np
from dlgo.nn.layers import sigmoid_double

def average_digit(data, digit):
    # Filtre les données en ne gardant que celles correspondant au chiffre spécifié (digit)
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]

    # Convertit la liste de données filtrées en un tableau numpy
    filtered_array = np.asarray(filtered_data)

    # Calcule la moyenne des données filtrées en fonction de l'axe 0 (chaque pixel)
    return np.average(filtered_array, axis=0)

"""
Cette fonction prend en entrée un ensemble de données et un chiffre spécifique (0-9).
 Elle filtre les données pour ne conserver que celles correspondant au chiffre spécifié, puis calcule et retourne la valeur moyenne pour chaque pixel des images filtrées.

"""

train, test = load_data()
avg_eight = average_digit(train, 8)  # <2>

# <1> Nous calculons la moyenne de tous les échantillons de nos données représentant un chiffre donné.
# <2> Nous utilisons le huit moyen comme paramètres pour un modèle simple de détection des huit.


from matplotlib import pyplot as plt

"""
img = (np.reshape(avg_eight, (28, 28)))
plt.imshow(img)
plt.show()
"""
# Affiche le chiffre moyen


x_3 = train[2][0]    # <1>
x_18 = train[17][0]  # <2>

W = np.transpose(avg_eight)
np.dot(W, x_3)   # <3>
np.dot(W, x_18)  # <4>

print(np.dot(W, x_3))
print(np.dot(W, x_18))

# <1> L'échantillon d'entraînement à l'index 2 est un "4".
# <2> L'échantillon d'entraînement à l'index 17 est un "8"
# <3> Cela donne environ 20,1.
# <4> Ce terme est beaucoup plus grand, environ 54,2.

def sigmoid_double(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid(z):
    return np.vectorize(sigmoid_double)(z)

print(sigmoid(np.dot(W, x_3)))
print(sigmoid(np.dot(W, x_18)))

print(sigmoid_double(np.dot(W, x_3)))
print(sigmoid_double(np.dot(W, x_18)))

""""

def predict(x, W, b):  # <1>
    return sigmoid_double(np.dot(W, x) + b)


b = -45  # <2>

print(predict(x_3, W, b))   # <3>
print(predict(x_18, W, b))  # <4>

# <1> Une prédiction simple est définie en appliquant la fonction sigmoïde à la sortie de np.dot(W, x) + b.
# <2> Sur la base des exemples calculés jusqu'à présent, nous fixons le terme de biais à -45.
# <3> La prédiction pour l'exemple avec un "4" est proche de zéro.
# <4> La prédiction pour un "8" est de 0,96 ici. Notre heuristique semble donner des résultats intéressants.


def evaluate(data, digit, threshold, W, b):  # <1>
    total_samples = 1.0 * len(data)
    correct_predictions = 0
    for x in data:
        if predict(x[0], W, b) > threshold and np.argmax(x[1]) == digit:  # <2>
            correct_predictions += 1
        if predict(x[0], W, b) <= threshold and np.argmax(x[1]) != digit:  # <3>
            correct_predictions += 1
    return correct_predictions / total_samples

# <1> Comme métrique d'évaluation, nous choisissons la précision, le rapport des prédictions correctes parmi toutes.
# <2> Prédire une instance de huit comme "8" est une prédiction correcte.
# <3> Si la prédiction est inférieure à notre seuil et que l'échantillon n'est pas un "8", nous avons également prédit correctement.


evaluate(data=train, digit=8, threshold=0.5, W=W, b=b)  # <1>

evaluate(data=test, digit=8, threshold=0.5, W=W, b=b)   # <2>

eight_test = [x for x in test if np.argmax(x[1]) == 8]
evaluate(data=eight_test, digit=8, threshold=0.5, W=W, b=b)  # <3>

# <1> La précision sur les données d'entraînement de notre modèle simple est de 78% (0.7814)
# <2> La précision sur les données de test est légèrement inférieure, à 77% (0.7749)
# <3> L'évaluation uniquement sur l'ensemble des huit dans l'ensemble de test donne une précision de 67% (0.6663)

"""