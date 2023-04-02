neurones_init = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
epsilon = 0.2

liste_entrees_objectifs = [([255,0,0],1), ([0,255,0],2), ([0,0,255],3),
([250,10,10],1), ([10,250,10],2), ([10,10,250],3),
([200,0,0],1), ([0,200,0],2), ([0,0,200],3),
([180,20,0],1), ([0,180,0],2), ([0,0,180],3),
([240,0,0],1), ([0,240,0],2), ([0,0,240],3),
([255,255,0],2), ([255,0,255],1), ([0,255,255],3) ]

def activation(neurone, entree):
    somme = 0
    for i in range(3):
        somme += neurone[i] * entree[i]
    return somme

def apprentissage(neurones, entree, objectif, epsilon):
    neurones_bis = []
    somme = [0, 0, 0]
    for n in range(3):
        somme[n] = activation(neurones[n], entree)
    s = somme.index(max(somme))
    if s == objectif:
        for n in range(3):
            neurones_bis.append(neurones[n])
    else:
        for n in range(3):
            if n == objectif:
                neurones_bis.append([neurones[n][i] + epsilon * entree[i] for i in range(3)])
            else:
                neurones_bis.append([neurones[n][i] - epsilon * entree[i] for i in range(3)])
    return neurones_bis

def epoque_apprentissage(neurones, liste_entrees_objectifs):
    for i in range(len(liste_entrees_objectifs)):
        neurones_actuels = apprentissage(neurones_init, liste_entrees_objectifs[i][0], liste_entrees_objectifs[i][1], epsilon)
    return neurones_actuels

def plusieurs_epoques(neurones, liste_entrees_objectifs):
    for i in range(50):
        neurones = epoque_apprentissage(neurones, liste_entrees_objectifs)
    return neurones

neurones_finaux = plusieurs_epoques(neurones_init, liste_entrees_objectifs)
print(neurones_finaux)
