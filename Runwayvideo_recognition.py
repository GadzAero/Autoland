# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 17:00:54 2025

@author: pradi

LECTURE VIDEO
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np

#image=cv2.imread(r"C:\Users\pradi\OneDrive - ensam.eu\Projet Dassault\3-DEVELOPPEMENT\3- Reconnaissance image\Approche1.jpg")
#image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Conversion mode de couleur
def traitement(image):
    img_height= image.shape[0]
    img_width = image.shape[1]

    #On définit un triangle de zone d'interetqui prend le quart bas de l'image
    region_interet_coordonnees= [(0,img_height), (img_width/2,img_height/2), (img_width,img_height)]


        
    #Passage en niveau de gris
    image_gris= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #image_gris = cv2.GaussianBlur(image_gris, (5, 5), 0)  # Applique un flou gaussien

    #Application de la fonction de detection de contours canny
    canny_image= cv2.Canny(image_gris, 100, 200)    
    """
    #On applique une morphologie pour réduire les petits artéfacts
    kernel = np.ones((3,3), np.uint8)  # Définir un noyau de 3x3 pixels
    canny_image = cv2.morphologyEx(canny_image, cv2.MORPH_CLOSE, kernel)  # Appliquer la fermeture morphologique
    """
    cropped_image = region_interet(canny_image, np.array([region_interet_coordonnees], np.int32))

    #Detection probabiliste des lignes
    """Ici on peut regler les criteres de detection de lignes pour filtrer les petites lignes intempestives 90 15"""
    lines=cv2.HoughLinesP(cropped_image, rho=6, theta=np.pi/60, threshold=160, lines=np.array([]), minLineLength=90, maxLineGap=15)
    image_avec_overlay=dessin_lignes(image,lines)
    return image_avec_overlay


    

def region_interet(img,coordonnees):
    #Cette fonction supprime toutes les zones de non interet
    
    masque= np.zeros_like(img) #Crée une matrice de 0 a la taille de l'image
    #channel_count=img.shape[2] #Nombre de cannaux de couleurs (3 en RGB), plus besoin en niveau de gris
    #match_mask_color=(255,)*channel_count #Le masque est passé en blanc sur tous les cannaux (il sera donc transparent)
    match_mask_color=255 #Un seul canal en niveau de gris
    
    cv2.fillPoly(masque, coordonnees, match_mask_color) # Remplissage du polygone correspondant à la zone d'intérêt avec la couleur blanche
    masked_image = cv2.bitwise_and(img, masque) #Reconstruit l'image
    return masked_image

def dessin_lignes(img, lines):
    #On dessine les lignes précédemment detectées (lines) sur l'image de base
    copie_img=np.copy(img)
    blank_image=np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8) #Image blanche de la meme taille que l'image de base
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                #On va supprimer toutes les lignes avec un angle limite pour ne pas avoir les horizontales
                if x2 - x1 == 0:
                    angle = 90  # Cas particulier : ligne verticale
                else:
                    angle = np.degrees(np.arctan(abs((y2 - y1) / (x2 - x1))))
                anglemax=10
                if angle>anglemax or angle<-anglemax:
                    cv2.line(blank_image,(x1,y1),(x2,y2), (0,255,0),thickness=2) #Les 2 derniers arguments sont la couleur et l'épaisseur du dessin
                #cv2.line(blank_image,(x1,y1),(x2,y2), (0,255,0),thickness=2) #Les 2 derniers arguments sont la couleur et l'épaisseur du dessin"""
    #Fusion du calque de lignes et de l'image originale
    img=cv2.addWeighted(img, 0.8, blank_image, 1, 0.0) #Les float sont les poids de chacune des images
    return img



video=cv2.VideoCapture(r"C:\Users\pradi\OneDrive - ensam.eu\Projet Dassault\3-DEVELOPPEMENT\3- Reconnaissance image\Aproches.mp4")
cv2.namedWindow("Runway recognition DUAV", cv2.WINDOW_NORMAL)  # Permet de redimensionner la fenêtre
cv2.resizeWindow("Runway recognition DUAV", 1080, 700)  # Ajuste la taille de la fenêtre (largeur x hauteur)



while (video.isOpened()):
    ret,frame = video.read()
    frame = traitement(frame) #On réécrit chaque frame avec le traitement
    cv2.imshow("Runway recognition DUAV",frame)
    if cv2.waitKey(1) & 0XFF == ord('x') : #Fermeture si X est pressé
        break
video.release()
cv2.destroyAllWindows()
    

    
    
"""
plt.imshow(image)

plt.imshow(canny_image)
plt.imshow(cropped_image)
plt.imshow(image_avec_overlay)
plt.show()

"""