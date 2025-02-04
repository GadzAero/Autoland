# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:47:23 2025

Runway recognition: Shape and ML 
@author: pradi
"""


import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from PIL import Image
# Charger YOLOv5 une seule fois

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

model = torch.hub.load("ultralytics/yolov5", "custom", 
                        path="C:/Users/pradi/Downloads/archive/yolov5-master/runs/train/exp/weights/best.pt", 
                        force_reload=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Déplacer le modèle sur le GPU

def afficher_imge(source,nom_fenetre):
    #Afficher masque
    cv2.namedWindow(nom_fenetre, cv2.WINDOW_NORMAL)  # Permet de redimensionner la fenêtre
    cv2.resizeWindow(nom_fenetre, 600, 400)
    cv2.imshow(nom_fenetre, source)

def inference(image):
    results = model(image)  # Au lieu de model.predict(image, size=640, conf=0.25)
    #results.show() #Cette fonction ralentit enormement A NE PAS UTILISER
    
    # Vérifier les résultats de détection
    boxes=[]

    for det in results.xywh[0]:  # Results pour la première image
        xmin, ymin, xmax, ymax, confidence, class_idx = det.tolist()
        
        #print(f"Detection: {class_idx} (Confidence: {confidence:.2f})")
        #print(f"Bounding box: ({xmin}, {ymin}, {xmax}, {ymax})")
        
        boxes.append([xmin,ymin,xmax,ymax,class_idx])
    return boxes
    
    

    
#image=cv2.imread(r"C:/Users/pradi/OneDrive - ensam.eu/Projet Dassault/3-DEVELOPPEMENT/3- Reconnaissance image/Approche1.jpg")
#image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Conversion mode de couleur
def traitement(image):
    torch.cuda.empty_cache()
    #On récupere les zones d'interet
    boxes=inference(image)
    img_height= image.shape[0]
    img_width = image.shape[1]
    #On applique la detection de contour que dans les zones données par le ML
    for box in boxes:
        x_cent, y_cent,width,height, classe = box
        xmin=x_cent-width/2
        xmax=x_cent+width/2
        ymin=y_cent-height/2
        ymax=y_cent+height/2
        
        #On définit un triangle de zone d'interetqui prend le quart bas de l'image
        region_interet_coordonnees= [(xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin)]
        
        
        
        #Passage en niveau de gris
        image_gris= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #image_gris = cv2.GaussianBlur(image_gris, (5, 5), 0)  # Applique un flou gaussien
        #Application de la fonction de detection de contours canny
        canny_image= cv2.Canny( image_gris, 100, 200)
        
        #Afficher masque
        cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)  # Permet de redimensionner la fenêtre
        cv2.resizeWindow("Canny", 600, 400)
        cv2.imshow("Canny", canny_image)
        
          # Ajuste la taille de la fenêtre (largeur x hauteur)
        cropped_image = region_interet(canny_image, np.array([region_interet_coordonnees], np.int32))
        #Afficher masque
        #afficher_imge(cropped_image, "Cropped")

        #Detection probabiliste des lignes
    """Ici on peut regler les criteres de detection de lignes pour filtrer les petites lignes intempestives 90 15"""
    if len(boxes)!=0:
        lines=cv2.HoughLinesP(cropped_image, rho=6, theta=np.pi/60, threshold=160, lines=np.array([]), minLineLength=80, maxLineGap=10)
        image_avec_overlay=dessin_lignes(image,lines)
        #Afficher masque
        #afficher_imge(image_avec_overlay, "Image_ overlay")

        
        return image_avec_overlay
    else: return image

    

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
                anglemax=0
                if angle>anglemax or angle<-anglemax:
                    cv2.line(blank_image,(x1,y1),(x2,y2), (0,255,0),thickness=2) #Les 2 derniers arguments sont la couleur et l'épaisseur du dessin
                #cv2.line(blank_image,(x1,y1),(x2,y2), (0,255,0),thickness=2) #Les 2 derniers arguments sont la couleur et l'épaisseur du dessin"""
    #Fusion du calque de lignes et de l'image originale
    img=cv2.addWeighted(img, 1, blank_image, 1, 0.0) #Les float sont les poids de chacune des images
    #afficher_imge(blanc_image, "Lignes")
    return img



video=cv2.VideoCapture(r"C:/Users/pradi/OneDrive - ensam.eu/Projet Dassault/3-DEVELOPPEMENT/3- Reconnaissance image/Cut1.mp4")
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