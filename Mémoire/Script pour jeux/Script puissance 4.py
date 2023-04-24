
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 17:47:01 2023

@author: mlanger
"""

import tkinter as tk
import math

# Initialisation de la grille de jeu
grille = [[0 for _ in range(7)] for _ in range(6)]

def afficher_grille():
    for i in range(6):
        for j in range(7):
            if grille[i][j] == 0:
                canvas.create_oval(j*100+20, i*100+20, j*100+80, i*100+80, fill="white")
            elif grille[i][j] == 1:
                canvas.create_oval(j*100+20, i*100+20, j*100+80, i*100+80, fill="red")
            elif grille[i][j] == 2:
                canvas.create_oval(j*100+20, i*100+20, j*100+80, i*100+80, fill="yellow")
    return

def jouer_coup(colonne, joueur):
    for i in range(5, -1, -1):
        if grille[i][colonne] == 0:
            grille[i][colonne] = joueur
            afficher_grille()
            break
    return

def verifier_gagnant(joueur):
    # Vérification des lignes
    for i in range(6):
        for j in range(4):
            if (grille[i][j] == joueur and 
                grille[i][j+1] == joueur and 
                grille[i][j+2] == joueur and 
                grille[i][j+3] == joueur):
                return True
    # Vérification des colonnes
    for i in range(3):
        for j in range(7):
            if (grille[i][j] == joueur and 
                grille[i+1][j] == joueur and 
                grille[i+2][j] == joueur and 
                grille[i+3][j] == joueur):
                return True
    # Vérification des diagonales
    for i in range(3):
        for j in range(4):
            if (grille[i][j] == joueur and 
                grille[i+1][j+1] == joueur and 
                grille[i+2][j+2] == joueur and 
                grille[i+3][j+3] == joueur):
                return True
            if (grille[i][j+3] == joueur and 
                grille[i+1][j+2] == joueur and 
                grille[i+2][j+1] == joueur and 
                grille[i+3][j] == joueur):
                return True
    return False

# Initialisation de l'interface graphique
root = tk.Tk()
root.title("Jeu de Puissance 4")
canvas = tk.Canvas(root, width=700, height=600)
canvas.pack()

# Dessin des lignes de la grille
for i in range(1, 7):
    canvas.create_line(i*100, 0, i*100, 600)
for i in range(1, 7):
    canvas.create_line(0, i*100, 700, i*100)
    
    #Affichage de la grille initiale

afficher_grille()
#Variables pour suivre le tour de jeu et le joueur courant

tour = 0
joueur = 1

def on_button_click(colonne):
    global tour, joueur
    if grille[0][colonne] != 0:
        canvas.create_text(350, 300, text="Colonne pleine, choisissez une autre colonne", font=("Arial", 20), fill="black")
    else:
        tour += 1
        jouer_coup(colonne, joueur)
        if verifier_gagnant(joueur):
            canvas.create_text(350, 300, text="Joueur {} a gagné !".format(joueur), font=("Arial", 30), fill="black")
        elif tour == 42:
            canvas.create_text(350, 300, text="Match nul !", font=("Arial", 30), fill="black")
        else:
            joueur = 1 if joueur == 2 else 2

#Boutons pour jouer les coups

for i in range(7):
    button = tk.Button(root, text="Jouer", command=lambda x=i: on_button_click(x))
    button.place(x=i*100+35, y=650)

root.mainloop()

import pickle

def sauvegarder_partie(nom_fichier):
    with open(nom_fichier, "wb") as fichier:
        pickle.dump(grille, fichier)

def charger_partie(nom_fichier):
    global grille
    with open(nom_fichier, "rb") as fichier:
        grille = pickle.load(fichier)

# Bouton pour sauvegarder la partie
save_button = tk.Button(root, text="Sauvegarder la partie", command=lambda: sauvegarder_partie("partie_en_cours.pickle"))
save_button.place(x=250, y=650)

# Bouton pour charger une partie
load_button = tk.Button(root, text="Charger une partie", command=lambda: charger_partie("partie_en_cours.pickle"))
load_button.place(x=450, y=650)

import tkinter.colorchooser as cc

# Variables pour stocker les couleurs des jetons pour chaque joueur
couleur_joueur_1 = "red"
couleur_joueur_2 = "yellow"

def choisir_couleur_joueur_1():
    global couleur_joueur_1
    couleur_joueur_1 = cc.askcolor()[1]

def choisir_couleur_joueur_2():
    global couleur_joueur_2
    couleur_joueur_2 = cc.askcolor()[1]

# Boutons pour choisir les couleurs des jetons pour chaque joueur
couleur_1_button = tk.Button(root, text="Choisir la couleur des jetons du joueur 1", command=choisir_couleur_joueur_1)
couleur_1_button.place(x=20, y=650)
couleur_2_button = tk.Button(root, text="Choisir la couleur des jetons du joueur 2", command=choisir_couleur_joueur_2)
couleur_2_button.place(x=280, y=650)

#Modification de la fonction  jouer_coup() pour utiliser les couleurs choisies
def jouer_coup(colonne, joueur):
    for i in range(5, -1, -1):
        if grille[i][colonne] == 0:
            if joueur == 1:
                grille[i][colonne] = couleur_joueur_1
            else:
                grille[i][colonne] = couleur_joueur_2
            afficher_grille()
            break