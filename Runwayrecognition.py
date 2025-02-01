# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:20:17 2025

@author:Félix 
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np

image=cv2.imread(r"C:\Users\pradi\OneDrive - ensam.eu\Projet Dassault\3-DEVELOPPEMENT\3- Reconnaissance image\Approche1.jpg")
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Conversion mode de couleur

#taille de l'image
print(image.shape)
img_height= image.shape[0]
img_width = image.shape[1]

#On définit un triangle de zone d'interetqui prend le quart bas de l'image
region_interet_coordonnees= [(0,img_height), (img_width/2,img_height/2), (img_width,img_height)]

def region_interet(img,coordonnees):
    #Cette fonction supprime toutes les zones de non interet
    
    masque= np.zeros_like(img) #Crée une matrice de 0 a la taille de l'image
    #channel_count=img.shape[2] #Nombre de cannaux de couleurs (3 en RGB), plus besoin en niveau de gris
    #match_mask_color=(255,)*channel_count #Le masque est passé en blanc sur tous les cannaux (il sera donc transparent)
    match_mask_color=255 #Un seul canal en niveau de gris
    
    cv2.fillPoly(masque, coordonnees, match_mask_color) # Remplissage du polygone correspondant à la zone d'intérêt avec la couleur blanche
    masked_image = cv2.bitwise_and(img, masque) #Reconstruit l'image
    return masked_image
#Passage en niveau de gris
image_gris= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#Application de la fonction de detection de contours canny
canny_image= cv2.Canny(image_gris, 100, 200)    
cropped_image = region_interet(canny_image, np.array([region_interet_coordonnees], np.int32))

#Detection probabiliste des lignes
"""Ici on peut regler les criteres de detection de lignes pour filtrer les petites lignes intempestives"""
lines=cv2.HoughLinesP(cropped_image, rho=6, theta=np.pi/60, threshold=160, lines=np.array([]), minLineLength=50, maxLineGap=10)

def dessin_lignes(img, lines):
    #On dessine les lignes précédemment detectées (lines) sur l'image de base
    copie_img=np.copy(img)
    blank_image=np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8) #Image blanche de la meme taille que l'image de base
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(blank_image,(x1,y1),(x2,y2), (0,255,0),thickness=5) #Les 2 derniers arguments sont la couleur et l'épaisseur du dessin
    #Fusion du calque de lignes et de l'image originale
    img=cv2.addWeighted(img, 0.8, blank_image, 1, 0.0) #Les float sont les poids de chacune des images
    return img
    
    
image_avec_overlay=dessin_lignes(image,lines)
    
    
"""
plt.imshow(image)

plt.imshow(canny_image)"""
plt.imshow(cropped_image)
plt.imshow(image_avec_overlay)
plt.show()

