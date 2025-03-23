# Les bibliothèques
import numpy as np
import copy
import matplotlib.pyplot as plt
from random import randrange
from PIL import Image

# Partie 1
# Fonction prend en paramètre un tabelau
def AfficherImg(img):
    plt.axis("off")
    plt.imshow(img, cmap = "gray")
#plt.imshow(img, cmap = "gray")#palette predefinie pour afficher une image
    plt.show()


def ouvrirImage(chemin):
    img = plt.imread(chemin)
    return img


def saveImage(img):
    plt.imsave("image1.png",img)


# Partie 2
# Q4
"""
    Fonction permet de créer une matrice de h lignes et l colonnes représentant une image noire
    Paramètres d'entrée :
                          h : La hauteur de l'image ( Nombre de lignes du matrice)
                          l : La largeur de l'image ( Nombre de colonnes du matrice)
    Paramètres de sortie :
                          tab_img : Un tableau (matrice) 
"""
def image_noire(h,l):
    tab_img = np.array([[0]*l for i in range(h)], int)
    return tab_img


# Q5
"""
    Fonction permet de créer une matrice de h lignes et l colonnes représentant une image blanche
    Paramètres d'entrée :
                          h : La hauteur de l'image ( Nombre de lignes du matrice)
                          l : La largeur de l'image ( Nombre de colonnes du matrice)
    Paramètres de sortie :
                          tab_img : Un tableau 
"""
def image_blanche(h, l):
    tab_img = np.array([[1]*l for i in range(h)], int)
    return tab_img


# Q6
"""
    Fonction permet de créer une matrice dont le contenu de chaque pixel (i,j) est donné par (i+j)%2
    Paramètres d'entrée :
                          h : La hauteur ( Nombre de lignes du matrice)
                          l : La largeur ( Nombre de colonnes du matrice)
    Paramètres de sortie :
                          tab_img : Une matrice
"""
def creerImgBlancNoir(h,l):
    tab_img = np.array([[0]*l for i in range(h)])  # Initialiser le tableau par des 0
    for i in range(h):
        for j in range(l):
            tab_img[i][j] = (i+j)%2
    return tab_img


# Q7
"""
    Fonction permet de donner le négatif de l'image donnée ( 0 devient 1 et 1 devient 0 )
    Paramètres d'entrée :
                          Img : matrice de l'image Img
    Paramètres de sortie :
                          img2 : Une matrice
"""
def negatif(Img):
    img2 = copy.deepcopy(Img)  # Faire une copie en profondeur du tableau Img
    for i in range(len(img2)):
        for j in range(len(img2[0])):
            if img2[i][j] == 0:
                img2[i][j] = 1
            else:
                img2[i][j] = 0
    return img2


# Q8
# Test des fonctions précédentes :
h = int(input("Veuiller entrer la hauteur de l'image : "))
l = int(input("Veuiller entrer la largeure de l'image : "))

print("La matrice de l'image noire : ")
print(image_noire(h, l))

print("la matrice de l'image blanche : ")
print(image_blanche(h, l))

# Image en blanc et noir :
img = creerImgBlancNoir(h, l)
AfficherImg(img)

# Le negatif de la dernière image :
AfficherImg(negatif(img))


# Partie 3
# Q9
"""
    Fonction permet de calculer la luminance de l'image donnée (La moyenne de tous les pixels )
    Paramètres d'entrée :
                          Img : une image
    Paramètres de sortie :
                          La luminance ( réel )
"""
def luminance(Img):
    T=np.asarray(Img)  # Convertir l'image en tableau
    s = 0
    cmpt = 0  # Compteur pour le nombre de pixels
    for i in range(len(T)):
        for j in range(len(T[0])):
            s += T[i][j]  # La somme de tous les pixels
            cmpt += 1
    return s/cmpt


Img1 = ouvrirImage("ImagePython.jpg")  # Ouvrir l'image 'ImagePython'
print("La luminance de l'image donnée est : ", luminance(Img1))


# Q10
"""
    Fonction permet de calculer le contraste de l'image donnée 
    Paramètres d'entrée :
                          Img : une image
    Paramètres de sortie :
                          Le contraste ( réel )
"""
def contrast(Img):
    Moy = luminance(Img)
    T=np.asarray(Img)  # Convertir l'image en tableau
    N = len(T) * len(T[0])
    c = 0
    for i in range(len(T)):
        for j in range(len(T[0])):
            c += (T[i][j] - Moy)**2
    return (1/N) * c


Img1 = ouvrirImage("ImagePython.jpg")  # Ouvrir l'image 'ImagePython'
print("Le contraste de l'image donnée est : ", contrast(Img1))


# Q11
"""
    Fonction qui retourne la valeur maximale d'un pixel dans l'image donnée
    Paramètres d'entrée :
                          Img : une image
    Paramètres de sortie :
                          max : Le maximum ( entier )
"""
def profondeur(Img):
    T=np.asarray(Img)  # Convertir l'image en tableau
    max = T[0][0]
    for i in range(len(T)):
        for j in range(len(T[0])):
            if (max < T[i][j]):
                max = T[i][j]
    return max


Img1 = ouvrirImage("ImagePython.jpg")  # Ouvrir l'image 'ImagePython'
print("La profondeur de l'image donnée est : ", profondeur(Img1))


# Q12
"""
    Fonction qui retourne la matrice représentant l'image donnée
    Paramètres d'entrée :
                          Img : une image
    Paramètres de sortie :
                          Un tableau
"""
def Ouvrir(Img):
    return (np.asarray(Img))  # Convertir l'image en tableau


Img1 = ouvrirImage("ImagePython.jpg")  # Ouvrir l'image 'ImagePython'
print("La matrice représentant l'image donnée : ")
print(Ouvrir(Img1))


# Partie 4
# Q13
"""
    Fonction qui retourne l’image inverse de l’image donnée
    Paramètres d'entrée :
                          Img : une image
    Paramètres de sortie :
                          Img_inv : une image
"""
def inverser(Img) :
    T = np.asarray(Img)  # Convertir l'image en tableau
    T1 = copy.deepcopy(T)  # Faire une copie en profondeur du tableau T
    for i in range(len(T1)):
        for j in range(len(T1[0])):
            T1[i][j] = 256-T1[i][j]
    Img_inv = Image.fromarray(T1)  # Convertir le tableau en image
    return Img_inv


Img1 = ouvrirImage("ImagePython.jpg")  # Ouvrir l'image 'ImagePython'
AfficherImg(inverser(Img1))


# Q14
"""
    Fonction qui retourne la transformée de l’image img par la symétrie d’axe vertical
    passant par le milieu de l’image
    Paramètres d'entrée :
                          Img : une image
    Paramètres de sortie :
                          Img_vertic : une image
"""
def flipH(Img):
    T= np.asarray(Img)  # Convertir l'image en tableau
    T1 = copy.deepcopy(T)  # Faire une copie du tableau T
    m = len(T1[0])
    for i in range(len(T1)):
        for j in range(m // 2):
            T1[i][j], T1[i][m - j - 1] = T1[i][m - j - 1], T1[i][j]
    Img_vertic = Image.fromarray(T1)  # Convertir le tableau en image
    return Img_vertic


Img1 = ouvrirImage("ImagePython.jpg")  # Ouvrir l'image 'ImagePython'
AfficherImg(flipH(Img1))


# Q15
"""
    Fonction qui retourne une nouvelle image obtenue en posant img1 au dessus img2
    Paramètres d'entrée :
                          img1 : une image
                          img2 : une image
    Paramètres de sortie :
                          image_V : Image
"""
def poserV(img1, img2):
    tab1_img1 = np.asarray(img1)  # Convertir l'image en tableau
    tab2_img1 = copy.deepcopy(tab1_img1)  # Faire une copie du tableau tab1_img1
    tab1_img2 = np.asarray(img2)  # Convertir l'image en tableau
    tab2_img2 = copy.deepcopy(tab1_img2)  # Faire une copie du tableau tab1_img2
    n = len(tab2_img1)
    m = len(tab2_img1[0])
    # Le nombre de lignes du tab_V est 2*n
    # Le nombre de colonnes du tab_V est m
    tab_V = np.array([[0]*m for k in range(2*n)])  # Initialisation du tab_V par des 0
    for j in range(m):
        for i in range(n):
            tab_V[i][j] = tab2_img1[i][j]  # Poser la première image img1
        i1 = 0
        for i in range(n, 2*n):
            tab_V[i][j] = tab2_img2[i1][j]  # Poser la deuxième image img2
            i1 += 1
    image_V = Image.fromarray(tab_V)  # Convertir le tableau en image
    return image_V


Img1 = ouvrirImage("ImagePython.jpg")  # Ouvrir l'image 'ImagePython'
AfficherImg(poserV(inverser(Img1), flipH(Img1)))


# Q16
"""
    Fonction qui retourne une nouvelle image obtenue en posant img1 à droite de img2
    Paramètres d'entrée :
                          img1 : une image
                          img2 : une image
    Paramètres de sortie :
                          image_H : Image
"""
def poserH(img1, img2):
    tab1_img1 = np.asarray(img1)  # Convertir l'image en tableau
    tab2_img1 = copy.deepcopy(tab1_img1)  # Faire une copie du tableau tab1_img1
    tab1_img2 = np.asarray(img2)  # Convertir l'image en tableau
    tab2_img2 = copy.deepcopy(tab1_img2)  # Faire une copie du tableau tab1_img2
    n = len(tab2_img1)
    m = len(tab2_img1[0])
    # Le nombre de lignes du tab_H est n
    # Le nombre de colonnes du tab_H est 2*m
    tab_H = np.array([[0]*(2*m) for k in range(n)])  # Initialisation du tab_H par des 0
    for i in range(n):
        for j in range(m):
            tab_H[i][j] = tab2_img1[i][j]  # Poser la première image img1
        j1 = 0
        for j in range(m, 2*m):
            tab_H[i][j] = tab2_img2[i][j1]  # Poser la deuxième image img2
            j1 += 1
    image_H = Image.fromarray(tab_H)  # Convertir le tableau en image
    return image_H


Img1 = ouvrirImage("ImagePython.jpg")  # Ouvrir l'image 'ImagePython'
AfficherImg(poserH(inverser(Img1), flipH(Img1)))


# Q22

M=[[[210, 100, 255], [100, 50, 255], [90, 90, 255], [90, 90, 255], [90, 90, 255], [90, 80, 255]],[[190, 255,89],[ 201, 255,29],[200, 255,100],[100, 255,90],[20, 255,200], [100, 255,80]],[[255,0, 0],[ 255,0, 0],[255,0, 0],[255,0, 0],[255,0, 0], [255,0, 0]] ]
# Affichage de l'image de la matrice M
plt.imshow(M)
plt.axis("off")
plt.show()
print("M[0][1][1] = ", M[0][1][1])
print("M[1][0][1] = ", M[1][0][1])
print("M[2][1][0] = ", M[2][1][0])

# Q23
"""
La quantité de mémoire nécessaire en octets(8 bits) pour stocker le tableau 
représentant une image RGB est (3 x n x m) telque n le nombre de lignes et 
m le nombre de colonnes.
Justification : pour calculer la quantité de mémoire nécessaire pour ce
stockage on doit calculer d'abord le nombre d'octets présents dans chaque
pixel, et puisque chaque pixel contient 3 couleurs (R,G,B) et chaque couleur
est codé sur 1 octets, donc un pixel est codé sur 3 octets , et puisque une
image contient (n x m) pixels alors la quantité de mémoire nécessaire est 
donnée par la relation : (taille d'un pixel x n x m) d'où : 3 x n x m

"""
# Q24
"""
    Fonction qui permet d'initialiser le tableau donné d'une manière aléatoire 
    Paramètres d'entrée :
                          imageRGB : Un tableau
    Paramètres de sortie :
                          image_H : Un tableau
"""
def initImageRGB(imageRGB):
    n = int(input("Veuillez entrer le nombre de lignes du tabeau imageRGB : "))
    m = int(input("Veuillez entrer le nombre de colonnes du tabeau imageRGB : "))
    # On utilise la fonction randrange pour générer un nombre aléatoire
    imageRGB = [[[randrange(256) for k in range(3)] for j in range(m) ]for i in range(n)]
    return imageRGB


# Déclarer un tableau vide
T = []
T_RGB = initImageRGB(T)
# Affichage de l'image
plt.imshow(T_RGB)
plt.axis("off")
plt.show()


# Q25
"""
    Fonction qui retourne une image symétrique à l'image img par rapport à l'axe horizontal 
    Paramètres d'entrée :
                          img : Une image
    Paramètres de sortie :
                          image_horiz = Une image
"""
def symetriehorizontale(img):
    tab1_img = np.asarray(img)  # Convertir l'image en tableau
    tab2_img = copy.deepcopy(tab1_img)  # Faire une copie du tableau tab1_img
    n = len(tab2_img)
    for i in range(n//2):
        for j in range(len(tab2_img[0])):
            for k in range(3):
                tab2_img[i][j][k], tab2_img[n-i-1][j][k] = tab2_img[n-i-1][j][k], tab2_img[i][j][k]
    image_horiz = Image.fromarray(tab2_img)  # Convertir le tableau en image
    return image_horiz


# Ouvrir l’image avec laquelle on va tester les fonctions des images RGB:
image = ouvrirImage("imageRGB.jpg")  # Ouvrir l'image 'imageRGB'
# Affichage de l'image
plt.imshow(image)
plt.axis("off")
plt.show()

image1 = symetriehorizontale(image)
# Affichage de l'image par symetrie horizontale
plt.imshow(image1)
plt.axis("off")
plt.show()

"""
    Fonction qui retourne une image symétrique à l'image img par rapport à l'axe vertical 
    Paramètres d'entrée :
                          img : Une image
    Paramètres de sortie :
                          image_vertic = Une image
"""
def symetrieverticale(img):
    tab1_img = np.asarray(img)  # Convertir l'image en tableau
    tab2_img = copy.deepcopy(tab1_img)  # Faire une copie du tableau tab1_img
    m = len(tab2_img[0])
    for i in range(len(tab2_img)):
        for j in range(m//2):
            for k in range(3):
                tab2_img[i][j][k], tab2_img[i][m-j-1][k] = tab2_img[i][m-j-1][k], tab2_img[i][j][k]
    image_vertic = Image.fromarray(tab2_img)  # Convertir le tableau en image
    return image_vertic


image2 = symetrieverticale(image)
# Affichage de l'image par symetrie verticale
plt.imshow(image2)
plt.axis("off")
plt.show()

"""
    Fonction qui retourne une image en niveaux de gris
    Paramètres d'entrée :
                          imageRGB : Une image
    Paramètres de sortie :
                          image = Une image
"""
# Q26
def grayscale(imageRGB):
    tab1_img = np.asarray(imageRGB)  # Convertir l'image en tableau
    tab2_img = copy.deepcopy(tab1_img)  # Faire une copie du tableau tab1_img
    for i in range(len(tab2_img)):
        for j in range(len(tab2_img[0])):
            max = tab2_img[i][j][0]
            min = tab2_img[i][j][0]
            for k in range(3):
                if tab2_img[i][j][k] > max:
                    max = tab2_img[i][j][k]  # Trouver le maximum de chaque pixel
                if tab2_img[i][j][k] < min:
                    min = tab2_img[i][j][k]  # Trouver le minimum de chaque pixel
            moy = (int(max)+int(min))//2  # Calculer la moyenne
            tab2_img[i][j] = moy  # La valeur du nouveau pixel en niveaux de gris
    image = Image.fromarray(tab2_img)  # Convertir le tableau en image
    return image


image_Gris = grayscale(image)
# Affichage de l'image en niveaux de gris
plt.imshow(image_Gris)
plt.axis("off")
plt.show()
