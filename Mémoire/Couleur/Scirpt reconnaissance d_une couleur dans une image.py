# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:36:49 2023

@author: Marin
"""

# Initialisation des poids du neurone
neurone_init=[1, 1, 1] 

# Taux d'apprentissage
epsilon = 0.2

# Importer les bibliothèques nécessaires
from PIL import Image
import os


# Définition de la fonction pour calculer la couleur moyenne d'une image
def couleur_moyenne(image_path):
    # Ouverture de l'image
    image = Image.open(image_path)
    # Chargement des pixels de l'image
    pixels = image.load()
    # Calcul de la taille de l'image
    largeur, hauteur = image.size
    # Initialisation des variables pour stocker la somme des valeurs de rouge, vert et bleu de chaque pixel
    rouge_total = vert_total = bleu_total = 0
    # Parcours de tous les pixels de l'image
    for x in range(largeur):
        for y in range(hauteur):
            # Extraction des valeurs de rouge, vert et bleu du pixel courant
            r, v, b = pixels[x, y]
            # Ajout des valeurs de rouge, vert et bleu à la somme totale
            rouge_total += r
            vert_total += v
            bleu_total += b
    # Calcul de la taille totale de l'image
    taille = largeur * hauteur
    # Calcul de la couleur moyenne de l'image
    rouge_moyen = rouge_total // taille
    vert_moyen = vert_total // taille
    bleu_moyen = bleu_total // taille
    # Retour de la couleur moyenne sous la forme d'un tuple de 3 entiers représentant les valeurs de rouge, vert et bleu
    return (rouge_moyen, vert_moyen, bleu_moyen)

"""
# Définir le chemin d'accès au dossier contenant les images
dossier_images = "C:/Users/Marin/Documents/Université/Memoire/Couleur/Banque_images_couleurs"

# Créer une liste de toutes les images dans le dossier
liste_images = os.listdir(dossier_images)

# Créer une liste de tuples contenant les informations sur chaque image
liste_entrees_objectifs = []
for image in liste_images:
    # Afficher l'image
    img = Image.open(os.path.join(dossier_images, image))
    img.show()

    # Demander à l'utilisateur si l'image doit être utilisée pour entraîner le neurone
    reponse = input("Cette image doit-elle être utilisée pour entraîner le neurone ? (o/n) ")

    # Ajouter un tuple à la liste des entrées et objectifs en fonction de la réponse de l'utilisateur
    if reponse.lower() == "o":
        objectif = 1
    else:
        objectif = 0
    codeRGB = couleur_moyenne(os.path.join(dossier_images, image))
    liste_entrees_objectifs.append((codeRGB, objectif))
"""


# Liste d'entrées et d'objectifs de test
liste_entrees_objectifs = [
([1,0,0],1), ([0,1,1],0), ([1,1,0],0),
([1,0,0.2],1), ([0,1,0],0), ([0,0,0],0),
([1,0,1],0), ([0.7,0,0],1), ([0.5,0.5,0.5],0),
([0.9,0.2,0],1), ([0.9,0,0],1), ([1,1,1],0),
([0.2,1,0],0), ([0.8,0.2,0],1), ([0.7,0.1,0.1],1) ]


# Fonction d'activation du neurone
def activation (neurone, entree):
    # Initialisation de la somme pondérée à 0 et de la sortie à 0
    somme = 0
    s = 0
    # Pour chaque élément de l'entrée, on ajoute le produit du poids correspondant et de l'entrée à la somme pondérée
    for i in range(3):
        somme += neurone[i] * entree[i]
    # Si la somme pondérée est supérieure ou égale à 1, la sortie est 1, sinon elle est 0
    if somme >= 1:
        s = 1
    else:
        s = 0
    # On retourne la sortie calculée
    return s

"""
La fonction activation calcule la sortie du neurone en effectuant une somme pondérée des entrées multipliées par les poids correspondants. 
Si la somme pondérée est supérieure ou égale à 1, la sortie est 1, sinon elle est 0. 
Cette fonction est utilisée pour déterminer la réponse du neurone à une entrée donnée, en comparant sa sortie à l'objectif attendu.
"""

# Fonction d'apprentissage du neurone
def apprentissage(neurone, entree, objectif, epsilon):
    neuronebis = []
    # Calcul de la sortie actuelle du neurone à l'aide de la fonction d'activation
    s = activation(neurone, entree)
    # Si la sortie actuelle correspond à l'objectif attendu, on ne modifie pas les poids du neurone
    if s == objectif:
        return neurone
    else:
        # Si la sortie actuelle est fausse, on ajuste les poids du neurone en fonction de l'erreur commise
        if s == 0 and objectif == 1:
            # Si la sortie actuelle est 0 mais l'objectif attendu est 1, on ajoute un petit poids positif à chaque élément du neurone
            for i in range(3):
                neuronebis.append(neurone[i] + epsilon * entree[i])
        else:
            # Si la sortie actuelle est 1 mais l'objectif attendu est 0, on soustrait un petit poids positif à chaque élément du neurone
            for i in range(3):
                neuronebis.append(neurone[i] - epsilon * entree[i])
        # On retourne les nouveaux poids du neurone après l'ajustement
        return neuronebis

    
"""
La fonction apprentissage ajuste les poids du neurone en fonction de l'erreur commise par le neurone lors de sa réponse à une entrée donnée. 
Si la sortie du neurone correspond à l'objectif attendu, les poids ne sont pas modifiés. Sinon, les poids sont ajustés en fonction de l'erreur commise.
 Si la sortie actuelle est 0 mais l'objectif attendu est 1, on ajoute un petit poids positif à chaque élément du neurone.
 Si la sortie actuelle est 1 mais l'objectif attendu est 0, on soustrait un petit poids positif à chaque élément du neurone.
 Le taux d'apprentissage epsilon est utilisé pour déterminer l'amplitude de l'ajustement des poids. La fonction retourne les nouveaux poids du neurone après l'ajustement.
"""

# Fonction pour effectuer une époque d'apprentissage sur le neurone
def epoque_apprentissage(neurone_init, liste_entrees_objectifs):
    for i in range (len(liste_entrees_objectifs)):
        # On applique la fonction d'apprentissage pour chaque entrée/objectif de la liste
        neurone_init = apprentissage(neurone_init, liste_entrees_objectifs[i][0], liste_entrees_objectifs[i][1], epsilon)        
    return neurone_init

# Version alternative de la fonction pour effectuer une époque d'apprentissage sur le neurone
def epoque_apprentissage_v(neurone_init, liste_entrees_objectifs):
    for x in liste_entrees_objectifs:
        entree, objectif = x 
        neurone_init = apprentissage(neurone_init, entree, objectif, epsilon)        
    return neurone_init

# Fonction pour effectuer plusieurs époques d'apprentissage sur le neurone
def plusieur_epoque(neurone_init, liste_entrees_objectifs):
   
    for i in range(50):
        # On applique plusieurs époques d'apprentissage sur le neurone
        neurone_init = epoque_apprentissage(neurone_init, liste_entrees_objectifs)
    return neurone_init

"""
La fonction `plusieur_epoque` applique 50 époques d'apprentissage sur le neurone en appelant la fonction `epoque_apprentissage` à chaque itération.
 L'objectif est d'améliorer les performances de classification du neurone en affinant les poids à chaque époque d'apprentissage.

La liste `liste_entrees_objectifs` contient des entrées et des objectifs de test, chacun sous forme de tuples. 
Les entrées sont des listes de 3 nombres représentant les niveaux de rouge, vert et bleu, tandis que les objectifs sont des entiers compris entre 1 et 3.
L'objectif est de faire en sorte que le neurone puisse classer correctement les entrées en fonction des objectifs attendus.

En somme, ce code met en œuvre un algorithme de classification de type Perceptron pour entraîner un neurone à reconnaître des motifs et à les classer en fonction de l'objectif attendu.
"""


# Chargement des poids du neurone entraîné
neurone = plusieur_epoque([1, 1, 1], liste_entrees_objectifs)


# codeRGB = couleur_moyenne("Couleur_orange.jpg")
# codeRGB = couleur_moyenne("Couleur_bleu.jpg")
#codeRGB = couleur_moyenne("Couleur_rouge.jpg")
codeRGB = couleur_moyenne("C:/Users/Marin/Documents/Université/Memoire/Couleur/Banque_images_couleurs/Couleur_noir.jpg")


# Classification de l'image à l'aide du neurone entraîné
# Évaluation de la sortie du neurone en fonction de la couleur moyenne de l'image
if activation(neurone, codeRGB) == 1:
    # Si la sortie du neurone est 1, cela signifie que la couleur moyenne de l'image correspond à l'objectif attendu
    print("L'image est de la couleur qu'on avait entraînée")
else:
    # Sinon, cela signifie que la couleur moyenne de l'image ne correspond pas à l'objectif attendu
    print("L'image n'est pas de la couleur qu'on avait entraînée")
