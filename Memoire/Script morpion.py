# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 09:02:37 2023

@author: Marin
"""

# Afficher le plateau de jeu
def afficher_plateau(plateau):
    for i in range(3):
        for j in range(3):
            print(plateau[i][j], end=" ")
        print()

# Vérifier si une ligne a gagné
def ligne_gagne(plateau, symbole):
    for i in range(3):
        if plateau[i][0] == plateau[i][1] == plateau[i][2] == symbole:
            return True
    return False

# Vérifier si une colonne a gagné
def colonne_gagne(plateau, symbole):
    for i in range(3):
        if plateau[0][i] == plateau[1][i] == plateau[2][i] == symbole:
            return True
    return False

# Vérifier si une diagonale a gagné
def diagonale_gagne(plateau, symbole):
    if (plateau[0][0] == plateau[1][1] == plateau[2][2] == symbole) or (plateau[0][2] == plateau[1][1] == plateau[2][0] == symbole):
        return True
    return False

# Vérifier si un joueur a gagné
def gagne(plateau, symbole):
    return ligne_gagne(plateau, symbole) or colonne_gagne(plateau, symbole) or diagonale_gagne(plateau, symbole)

# Vérifier si le plateau est plein
def plateau_plein(plateau):
    for i in range(3):
        for j in range(3):
            if plateau[i][j] == " ":
                return False
    return True

# Début du jeu
plateau = [[" " for x in range(3)] for y in range(3)]
joueur = "X"
fin_de_jeu = False

while not fin_de_jeu:
    afficher_plateau(plateau)
    print("Joueur '" + joueur + "', entrez les coordonnées de la case à jouer (ligne colonne) :")
    ligne, colonne = map(int, input().split())
    if plateau[ligne][colonne] == " ":
        plateau[ligne][colonne] = joueur
        if gagne(plateau, joueur):
            afficher_plateau(plateau)
            print("Joueur '" + joueur + "' a gagné !")
            fin_de_jeu = True
        elif plateau_plein(plateau):
            afficher_plateau(plateau)
            print("Match nul !")
            fin_de_jeu = True
        else:
            if joueur == "X":
                joueur = "O"
            else:
                joueur = "X"
    else:
        print("Case déjà jouée, veuillez en choisir une autre")
        
