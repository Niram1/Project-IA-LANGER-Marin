import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import to_categorical

# Charger les données MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Redimensionner les données d'entrée
X_train = X_train.reshape(-1, 784).astype('float32') / 255
X_test = X_test.reshape(-1, 784).astype('float32') / 255

# Convertir les étiquettes en catégories one-hot
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Créer le modèle
model = Sequential()
model.add(Dense(392, input_shape=(784,), activation='relu'))
model.add(Dense(196, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compiler le modèle
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, batch_size=10, epochs=10, validation_data=(X_test, y_test))

# Sauvegarder le modèle
model.save('Model_keras.pkl')

# from keras.models import load_model

# # Charger le modèle sauvegardé
# loaded_model = load_model("Model_keras.pkl")

# # Vérifier l'accuracy du modèle chargé sur les données de test
# loaded_model_acc = loaded_model.evaluate(X_test, y_test, verbose=0)[1]
# print(f"Précision du modèle chargé sur l'ensemble de test : {loaded_model_acc * 100:.2f}%")
