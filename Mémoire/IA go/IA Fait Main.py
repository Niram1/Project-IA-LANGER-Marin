import gzip
import numpy as np
import random
from tqdm import tqdm
import pickle


def encode_label(j):  # <1>
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# <1> Nous utilisons l'encodage one-hot pour les indices en vecteurs de longueur 10.


def shape_data(data):
    features = [np.reshape(x, (784, 1)) / 255.0 for x in data[0]]  # <1>

    labels = [encode_label(y) for y in data[1]]  # <2>

    return list(zip(features, labels))
  # <3>

# <1> Nous aplatissons les images d'entrée en vecteurs de caractéristiques de longueur 784.
# <2> Tous les labels sont encodés en one-hot.
# <3> Ensuite, nous créons des paires de caractéristiques et de labels.


def load_data_impl():
    # Définit le chemin du fichier contenant les données MNIST
    path = 'C:\\Users\\Marin\\Documents\\Université\\Memoire\\IA go\\Fichier mnist\\mnist.npz'
    
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


# img = (np.reshape(avg_eight, (28, 28)))
# plt.imshow(img)
# plt.show()



# Affiche le chiffre moyen

W = np.transpose(avg_eight)


# x_3 = train[2][0]    # <1>
# x_18 = train[17][0]  # <2>

# W = np.transpose(avg_eight)
# np.dot(W, x_3)   # <3>
# np.dot(W, x_18)  # <4>

# print(np.dot(W, x_3))
# print(np.dot(W, x_18))


# <1> L'échantillon d'entraînement à l'index 2 est un "4".
# <2> L'échantillon d'entraînement à l'index 17 est un "8"


def sigmoid_double(x):
    return 1.0 / (1.0 + np.exp(-x))

# La fonction sigmoid_double prend en argument un nombre x et retourne sa valeur transformée à l'aide de la fonction sigmoïde. 
# La fonction sigmoïde est une fonction mathématique couramment utilisée en apprentissage automatique pour transformer une valeur en une valeur comprise entre 0 et 1.

def sigmoid(z):
    return np.vectorize(sigmoid_double)(z)

# La fonction sigmoid prend en argument un tableau NumPy z et applique la fonction sigmoïde à chaque élément de ce tableau à l'aide de la fonction np.vectorize. Elle retourne le tableau transformé.

def predict(x, W, b):  # <1>
    return sigmoid_double(np.dot(W, x) + b)

# La fonction predict prend en argument une entrée x (un tableau NumPy), une matrice de poids W et un biais b. 
# Elle calcule la sortie de la couche de neurones à l'aide de la formule de propagation avant : multiplication de la matrice de poids W par l'entrée x,
# addition du biais b au résultat de la multiplication, puis application de la fonction sigmoïde à cette somme. La fonction retourne la sortie résultante.

b = -45  # <2>

# print(predict(x_3, W, b))   # <3>
# print(predict(x_18, W, b))  # <4>


# <1> Une prédiction simple est définie en appliquant la fonction sigmoïde à la sortie de np.dot(W, x) + b.
# <2> Sur la base des exemples calculés jusqu'à présent, nous fixons le terme de biais à -45.
# <3> La prédiction pour l'exemple avec un "4" est proche de zéro (1.52 10 puissance -11).
# <4> La prédiction pour un "8" est de 0,999 ici. Notre heuristique semble donner des résultats intéressants.

def evaluate(data, digit, threshold, W, b):  # fonction qui évalue les performances du modèle
    total_samples = 1.0 * len(data)  # calcule le nombre total d'échantillons
    correct_predictions = 0  # initialise le nombre de prédictions correctes à 0
    for x in data:  # boucle sur chaque échantillon dans les données d'entrée
        if predict(x[0], W, b) > threshold and np.argmax(x[1]) == digit:  # si la prédiction est correcte
            correct_predictions += 1  # incrémente le nombre de prédictions correctes
        if predict(x[0], W, b) <= threshold and np.argmax(x[1]) != digit:  # si la prédiction est incorrecte
            correct_predictions += 1  # incrémente le nombre de prédictions correctes (cela compte comme une erreur car le modèle a prédit que c'était la bonne classe)
    return correct_predictions / total_samples  # retourne le taux de prédictions correctes sur l'ensemble des échantillons

# <1> Comme métrique d'évaluation, nous choisissons la précision, le rapport des prédictions correctes parmi toutes.
# <2> Prédire une instance de huit comme "8" est une prédiction correcte.
# <3> Si la prédiction est inférieure à notre seuil et que l'échantillon n'est pas un "8", nous avons également prédit correctement.


evaluate(data=train, digit=8, threshold=0.5, W=W, b=b)  # <1>

# Cette fonction évalue les performances du modèle entraîné sur l'ensemble de données d'entraînement (train) pour reconnaître le chiffre 8.
# digit=8 spécifie que la fonction recherche des images contenant le chiffre 8.
# threshold=0.5 spécifie la valeur de seuil à utiliser pour la classification binaire.
# W et b sont les poids et les biais du modèle qui ont été entraînés précédemment.

evaluate(data=test, digit=8, threshold=0.5, W=W, b=b)   # <2>

# Cette fonction évalue les performances du modèle entraîné sur l'ensemble de données de test (test) pour reconnaître le chiffre 8.
# digit=8 spécifie que la fonction recherche des images contenant le chiffre 8.
# threshold=0.5 spécifie la valeur de seuil à utiliser pour la classification binaire.
# W et b sont les poids et les biais du modèle qui ont été entraînés précédemment.

eight_test = [x for x in test if np.argmax(x[1]) == 8]
evaluate(data=eight_test, digit=8, threshold=0.5, W=W, b=b)  # <3>

# Cette fonction évalue les performances du modèle entraîné sur un sous-ensemble des données de test (eight_test) pour reconnaître le chiffre 8.
# digit=8 spécifie que la fonction recherche des images contenant le chiffre 8.
# threshold=0.5 spécifie la valeur de seuil à utiliser pour la classification binaire.
# W et b sont les poids et les biais du modèle qui ont été entraînés précédemment.
# eight_test est une liste de tous les exemples de test contenant le chiffre 8.

#AUTEUR
# <1> La précision sur les données d'entraînement de notre modèle simple est de 78% (0.7814)
# <2> La précision sur les données de test est légèrement inférieure, à 77% (0.7749)
# <3> L'évaluation uniquement sur l'ensemble des huit dans l'ensemble de test donne une précision de 67% (0.6663)


# print(evaluate(data=train, digit=8, threshold=0.5, W=W, b=b))
# print(evaluate(data=test, digit=8, threshold=0.5, W=W, b=b))
# print(evaluate(data=eight_test, digit=8, threshold=0.5, W=W, b=b))

#MOI
# <1> La précision sur les données d'entraînement de notre modèle simple est de 68.2% (0.682)
# <2> La précision sur les données de test est légèrement inférieure, à 66.94% (0.6694)
# <3> L'évaluation uniquement sur l'ensemble des huit dans l'ensemble de test donne une précision de 81.93% (0.81930184)



class MSE:
    def __init__(self):
        pass
    
    @staticmethod
    def loss_function(predictions, labels):
        """
        Cette méthode calcule la fonction de perte MSE (Mean Squared Error) entre les prédictions et les étiquettes
        :param predictions: ndarray, tableau 2D contenant les prédictions du modèle
        :param labels: ndarray, tableau 2D contenant les étiquettes réelles
        :return: float, la fonction de perte MSE entre les prédictions et les étiquettes
        """
        diff = predictions - labels  # Calcul de la différence entre les prédictions et les étiquettes
        return 0.5 * sum(diff * diff)[0]  # Calcul de la fonction de perte MSE
    
    @staticmethod
    def loss_derivative(predictions, labels):
        """
        Cette méthode calcule la dérivée de la fonction de perte MSE par rapport aux prédictions
        :param predictions: ndarray, tableau 2D contenant les prédictions du modèle
        :param labels: ndarray, tableau 2D contenant les étiquettes réelles
        :return: ndarray, tableau 2D contenant la dérivée de la fonction de perte MSE par rapport aux prédictions
        """
        return predictions - labels  # Calcul de la dérivée de la fonction de perte MSE par rapport aux prédictions
    
# Cette classe définit deux méthodes statiques : loss_function() et loss_derivative(). La méthode loss_function() calcule la fonction de perte MSE (Mean Squared Error) entre les prédictions et les étiquettes. 
# La méthode loss_derivative() calcule la dérivée de la fonction de perte MSE par rapport aux prédictions. Les deux méthodes prennent en entrée deux tableaux 2D :
# predictions et labels, qui représentent respectivement les prédictions et les étiquettes réelles.


class Layer:
    def __init__(self):
        self.params = []    # Les paramètres de la couche, initialisés à une liste vide
        
        self.previous = None   # Référence à la couche précédente
        self.next = None   # Référence à la couche suivante
        
        self.input_data = None   # Données en entrée de la couche
        self.output_data = None  # Données en sortie de la couche
        
        self.input_delta = None   # Delta en entrée de la couche
        self.output_delta = None  # Delta en sortie de la couche

    def connect(self, layer):
        self.previous = layer  # Connexion à la couche précédente
        layer.next = self      # Mise à jour de la référence à la couche suivante

    def forward(self):
        raise NotImplementedError   # La méthode pour calculer la propagation directe n'est pas implémentée
    
    def get_forward_input(self):
        if self.previous is not None:
            return self.previous.output_data  # Récupération des données en sortie de la couche précédente
        else:
            return self.input_data   # Si c'est la première couche, on utilise les données en entrée
    
    def backward(self):
        raise NotImplementedError  # La méthode pour calculer la rétropropagation n'est pas implémentée
    
    def get_backward_input(self):
        if self.next is not None:
            return self.next.output_delta   # Récupération du delta en sortie de la couche suivante
        else:
            return self.input_delta   # Si c'est la dernière couche, on utilise le delta en entrée
    
    def clear_deltas(self):
        pass   # Effacer les deltas ne fait rien pour cette couche
    
    def update_params(self, learning_rate):
        pass   # La mise à jour des paramètres ne fait rien pour cette couche
    
    def describe(self):
        raise NotImplementedError  # La méthode pour décrire la couche n'est pas implémentée


def sigmoid_prime_double(x):
    return sigmoid_double(x) * (1 - sigmoid_double(x))  # La fonction de dérivée de la fonction sigmoid_double
    
def sigmoid_prime(z):
    return np.vectorize(sigmoid_prime_double)(z)  # Application vectorisée de la fonction sigmoid_prime_double


# Ce code définit la classe Layer qui sera utilisée pour créer les différentes couches du réseau de neurones.
# Elle est accompagnée de deux fonctions auxiliaires sigmoid_prime_double et sigmoid_prime pour calculer la dérivée de la fonction sigmoïde. 
# La classe Layer possède plusieurs attributs pour stocker les données, les paramètres et les deltas,
# ainsi que des méthodes pour gérer les connexions entre les couches, 
# effectuer la propagation directe et la rétropropagation, effacer les deltas et mettre à jour les paramètres. 
# Les méthodes forward, backward et describe sont des méthodes abstraites et doivent être implémentées dans les classes dérivées.


class ActivationLayer(Layer):
    def __init__(self, input_dim):
        super(ActivationLayer, self).__init__()

        # Initialisation des dimensions d'entrée et de sortie de la couche
        self.input_dim = input_dim
        self.output_dim = input_dim

    def forward(self):
        # Récupération des données d'entrée
        data = self.get_forward_input()
        # Application de la fonction d'activation sigmoid à ces données
        self.output_data = sigmoid(data)

    def backward(self):
        # Récupération du delta de sortie de la couche suivante
        delta = self.get_backward_input()
        # Récupération des données d'entrée
        data = self.get_forward_input()
        # Calcul du delta de la couche actuelle à partir du delta de la couche suivante et de la dérivée de la fonction d'activation sigmoid appliquée aux données d'entrée
        self.output_delta = delta * sigmoid_prime(data)

    def describe(self):
        # Affichage des informations de la couche
        print("|-- " + self.__class__.__name__)
        print(" |-- dimensions: ({},{})"
            .format(self.input_dim, self.output_dim))

# La classe ActivationLayer hérite de la classe Layer et représente une couche d'activation appliquant la fonction sigmoid à ses entrées.

# La méthode __init__ initialise les dimensions d'entrée et de sortie de la couche.

# La méthode forward calcule la sortie de la couche en appliquant la fonction sigmoid aux données d'entrée récupérées grâce à la méthode get_forward_input.

# La méthode backward calcule le delta de la couche en utilisant le delta de la couche suivante et la dérivée de la fonction d'activation sigmoid appliquée aux données d'entrée récupérées grâce à la méthode get_forward_input.

# La méthode describe affiche les informations de la couche.

class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim, weight=None, bias=None):
        super(DenseLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if weight is not None:
            self.weight = weight
        else:
            self.weight = np.random.randn(output_dim, input_dim) / np.sqrt(input_dim)
        
        if bias is not None:
            self.bias = bias
        else:
            self.bias = np.random.randn(output_dim, 1)

        # Stockage des poids et des biais dans une liste pour pouvoir les mettre à jour lors de l'apprentissage
        self.params = [self.weight, self.bias]

        # Initialisation des deltas pour les poids et les biais
        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)

    def forward(self):
        # Récupération des données en entrée
        data = self.get_forward_input()

        # Calcul de la sortie en multipliant les poids par les données en entrée et en ajoutant les biais
        self.output_data = np.dot(self.weight, data) + self.bias

    def backward(self):
        # Récupération des données en entrée et des deltas en sortie
        data = self.get_forward_input()
        delta = self.get_backward_input()

        # Calcul du delta pour les biais en ajoutant les deltas en sortie
        self.delta_b += delta

        # Calcul du delta pour les poids en multipliant les deltas en sortie par les données en entrée transposées
        self.delta_w += np.dot(delta, data.transpose())

        # Calcul du delta en entrée pour la couche précédente en multipliant les poids transposés par les deltas en sortie
        self.output_delta = np.dot(self.weight.transpose(), delta)

    def update_params(self, rate):
        # Mise à jour des poids et des biais en soustrayant les deltas multipliés par le taux d'apprentissage
        self.weight -= rate * self.delta_w
        self.bias -= rate * self.delta_b

    def clear_deltas(self):
        # Réinitialisation des deltas pour les poids et les biais
        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)

    def describe(self):
        # Affichage du nom de la couche et des dimensions des données en entrée et en sortie
        print("|--- " + self.__class__.__name__)
        print(" |-- dimensions: ({},{})"
            .format(self.input_dim, self.output_dim))
        
# La classe DenseLayer est une couche dense, c'est-à-dire qu'elle connecte chaque neurone de la couche précédente à chaque neurone de la couche suivante. Les poids et les biais sont initialisés avec des valeurs aléatoires, et les deltas sont initialisés à zéro.

# La méthode forward calcule la sortie de la couche en multipliant les poids par les données en entrée et en ajoutant les biais.

# La méthode backward calcule les deltas pour les poids et les biais en utilisant les deltas en sortie et les données en entrée. Elle calcule également le delta en entrée pour la couche précédente.

# La méthode update_params met à jour les poids et les biais en soustrayant les deltas multipliés par le taux d'apprentissage.

# La méthode clear_deltas(self) réinitialise les deltas accumulés pendant la phase de rétropropagation, pour éviter qu'ils ne s'accumulent à chaque itération de l'entraînement.

# Enfin, la méthode describe(self) permet d'afficher les informations de base sur le layer, telles que ses dimensions.
    
class SequentialNetwork:
    def __init__(self, loss=None):
        print("Initialize Network...")
        self.layers = []
        if loss is None:
            self.loss = MSE()  # initialisation de la fonction de coût à MSE si elle n'est pas fournie

    def add(self, layer):
        self.layers.append(layer)  # ajout d'une couche à la liste des couches
        layer.describe()  # affichage de la description de la couche
        if len(self.layers) > 1:
            self.layers[-1].connect(self.layers[-2])  # connexion de la couche courante à la couche précédente

    def train(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        n = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)  # mélange des données
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            # création de mini-lots pour l'entraînement
            for mini_batch in tqdm(mini_batches, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
                self.train_batch(mini_batch, learning_rate)  # entraînement pour chaque mini-lot
            if test_data:
                n_test = len(test_data)
                print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data), n_test))
                # évaluation sur les données de test à chaque fin d'époque
            else:
                print("Epoch {0} complete".format(epoch))

    def train_batch(self, mini_batch, learning_rate):
        self.forward_backward(mini_batch)  # propagation et rétropropagation pour chaque mini-lot

        self.update(mini_batch, learning_rate)  # mise à jour des paramètres pour chaque mini-lot

    def update(self, mini_batch, learning_rate):
        learning_rate = learning_rate / len(mini_batch)
        for layer in self.layers:
            layer.update_params(learning_rate)  # mise à jour des paramètres pour chaque couche
        for layer in self.layers:
            layer.clear_deltas()  # réinitialisation des deltas pour chaque couche

    def forward_backward(self, mini_batch):
        for x, y in mini_batch:
            self.layers[0].input_data = x
            for layer in self.layers:
                layer.forward()  # propagation avant pour chaque couche
            self.layers[-1].input_delta = self.loss.loss_derivative(self.layers[-1].output_data, y)
            # calcul de la dérivée de la fonction de coût pour la dernière couche
            for layer in reversed(self.layers):
                layer.backward()  # rétropropagation pour chaque couche en partant de la dernière

    def single_forward(self, x):
        self.layers[0].input_data = x
        for layer in self.layers:
            layer.forward()  # propagation avant pour chaque couche
        return self.layers[-1].output_data

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.single_forward(x)), np.argmax(y)) for (x, y) in test_data]
        # prédiction pour chaque exemple de test et comparaison avec la réponse attendue
        return sum(int(x == y) for (x, y) in test_results)  # somme des prédictions correctes

    def save_model(self, file_path):
        model_data = []
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                model_data.append({
                    'type': 'DenseLayer',
                    'input_dim': layer.input_dim,
                    'output_dim': layer.output_dim,
                    'weight': layer.weight,
                    'bias': layer.bias
                })
            elif isinstance(layer, ActivationLayer):
                model_data.append({
                    'type': 'ActivationLayer',
                    'input_dim': layer.input_dim
                })
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
    
def load_data():
    training_data, test_data = load_data_impl()
    return shape_data(training_data), shape_data(test_data)

training_data, test_data = load_data()

# Le code suivant crée une instance de SequentialNetwork et y ajoute plusieurs couches de neurones à l'aide des classes DenseLayer et ActivationLayer. 
# Il entraîne ensuite le réseau de neurones sur les données training_data pendant un certain nombre d'epochs,
# en utilisant une taille de mini-batch de 10, un taux d'apprentissage de 3.0 et les données test_data pour évaluer les performances.

# net = SequentialNetwork()
# net.add(DenseLayer(784, 392))  # Ajout d'une couche DenseLayer avec 784 entrées et 392 sorties
# net.add(ActivationLayer(392)) # Ajout d'une couche d'activation pour la couche précédente
# net.add(DenseLayer(392, 196)) # Ajout d'une couche DenseLayer avec 392 entrées et 196 sorties
# net.add(ActivationLayer(196)) # Ajout d'une couche d'activation pour la couche précédente
# net.add(DenseLayer(196, 10))  # Ajout d'une couche DenseLayer avec 196 entrées et 10 sorties
# net.add(ActivationLayer(10))  # Ajout d'une couche d'activation pour la couche précédente

# # Entraînement du réseau de neurones

# net.train(training_data, epochs=10, mini_batch_size=10, learning_rate=3.0, test_data=test_data)


# Explications :

# La première ligne crée une instance de la classe SequentialNetwork qui représente un réseau de neurones séquentiel.
# Les six lignes suivantes ajoutent des couches de neurones au réseau à l'aide des classes DenseLayer et ActivationLayer. 
# Chaque couche DenseLayer est suivie d'une couche ActivationLayer qui applique une fonction d'activation à la sortie de la couche précédente. 
# Ces couches sont créées avec des dimensions spécifiques, qui dépendent des dimensions de l'entrée et de la sortie de chaque couche. 
# La première couche a 784 entrées (la taille des images MNIST) et 392 sorties, la deuxième couche a 392 entrées et 196 sorties, et la dernière couche a 196 entrées et 10 sorties (le nombre de classes pour la classification MNIST).
# La dernière ligne entraîne le réseau de neurones sur les données training_data pendant cinq epochs en utilisant une taille de mini-batch de 10, un taux d'apprentissage de 3.0 et les données test_data pour évaluer les performances. 
# Pendant l'entraînement, les données d'entraînement sont mélangées et divisées en mini-batchs. 
# Pour chaque mini-batch, le réseau de neurones est entraîné en appelant la méthode train_batch(), qui effectue une propagation avant et une propagation arrière à travers le réseau pour calculer les gradients des paramètres du réseau.
# Les paramètres sont ensuite mis à jour en appelant la méthode update_params() pour chaque couche du réseau. La méthode clear_deltas() est également appelée pour chaque couche pour effacer les gradients des paramètres calculés pendant le mini-batch. 
# En fin d'époch, les performances du réseau sont évaluées en appelant la méthode evaluate() avec les données de test et en comparant les sorties du réseau avec les étiquettes de classe réelles

def compute_model_accuracy(network, test_data):
    correct_predictions = 0
    total_samples = len(test_data)
    
    for x, y in test_data:
        predicted_label = np.argmax(network.single_forward(x))
        true_label = np.argmax(y)
        
        if predicted_label == true_label:
            correct_predictions += 1
            
    accuracy = correct_predictions / total_samples
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
# compute_model_accuracy(net, test_data)

# net.save_model('Model.pkl')


def load_model(file_path):
    with open(file_path, 'rb') as f:
        model_data = pickle.load(f)
    
    loaded_model = SequentialNetwork()
    
    for layer_data in model_data:
        if layer_data['type'] == 'DenseLayer':
            loaded_model.add(DenseLayer(layer_data['input_dim'], layer_data['output_dim'], layer_data['weight'], layer_data['bias']))
        elif layer_data['type'] == 'ActivationLayer':
            loaded_model.add(ActivationLayer(layer_data['input_dim']))
            
    return loaded_model

# Charger le modèle à partir du fichier Model.pkl
loaded_model = load_model('Model.pkl')

# Vérifier l'accuracy du modèle chargé sur les données de test
compute_model_accuracy(loaded_model, test_data)

# Continuer l'entraînement du modèle chargé
loaded_model.train(training_data, epochs=5, mini_batch_size=10, learning_rate=3.0, test_data=test_data)

# Calculer l'exactitude du modèle chargé et entraîné
compute_model_accuracy(loaded_model, test_data)

# Sauvegarder le modèle après l'entraînement supplémentaire
loaded_model.save_model('Model_updated.pkl')
