"""
Created on Thu Feb  2 08:20:17 2023

@author: Marin
"""


neurones_init = [    {"nom": "neurone_rouge", "poids": [1, 1, 1]},
    {"nom": "neurone_vert", "poids": [1,1, 1]},
    {"nom": "neurone_bleu", "poids": [1, 1, 1]}
]

epsilon = 0.2

liste_entrees_objectifs=[[[39, 91, 98], 2], [[35, 220, 94], 1], [[183, 60, 145], 0], [[136, 60, 136], 0], [[156, 223, 172], 1], [[29, 93, 104], 2], [[245, 223, 16], 0], [[13, 61, 112], 2], [[57, 240, 160], 1], [[5, 176, 45], 1]]
import random

# Créer une liste aléatoire de liste qui comporte 3 numéros compris entre 0 et 255 (indice rgb) et qui donne comme objectif
# le numéro de la case ayant le plus gros nombre


def liste(n):
    
    liste_entrees_objectifs = []
    for i in range(n):
        entree = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        objectif = entree.index(max(entree))
        liste_entrees_objectifs.append([entree, objectif])
    return liste_entrees_objectifs

def activation(neurone, entree):
    # Calcul de la somme pondérée des entrées pour un neurone donné
    somme = 0
    for i in range(3):
        somme += neurone["poids"][i] * entree[i]
    return somme

def apprentissage(neurones, entree, objectif, epsilon):
    # Modification des poids des neurones en fonction du résultat de l'activation
    neurones_bis = []
    somme = [0, 0, 0]
    for n in range(3):
        somme[n] = activation(neurones[n], entree)
    # Détermination du neurone ayant la plus grande activation
    s = somme.index(max(somme))
    #print("Le neurone ayant la plus grande activation est :", neurones[s]["nom"])
    if s == objectif:
        for n in range(3):
            neurones_bis.append(neurones[n])
    else:
        for n in range(3):
            if n == objectif:
                # Mise à jour des poids pour le neurone cible
                neurones_bis.append({
                    "nom": neurones[n]["nom"],
                    "poids": [neurones[n]["poids"][i] + epsilon * entree[i] for i in range(3)]
                })
            else:
                # Mise à jour des poids pour les autres neurones
                neurones_bis.append({
                    "nom": neurones[n]["nom"],
                    "poids": [neurones[n]["poids"][i] - epsilon * entree[i] for i in range(3)]
                })
    return neurones_bis

def epoque_apprentissage(neurones, liste_entrees_objectifs):
    for i in range(len(liste_entrees_objectifs)):
        neurones = apprentissage(neurones, liste_entrees_objectifs[i][0], liste_entrees_objectifs[i][1], epsilon)
    return neurones

def plusieurs_epoques(neurones, liste_entrees_objectifs):
    for i in range(50):
        neurones = epoque_apprentissage(neurones, liste_entrees_objectifs)
    return neurones

neurones_finaux = plusieurs_epoques(neurones_init, liste_entrees_objectifs)
#print(neurones_finaux)
print(liste(10))
print(neurones_finaux)