import cv2
import numpy as np

# Charger l'image
image = cv2.imread("C:/Users/pradi/Downloads/archive/datasets/runway/images/train/EDDF07C1_3FNLImage2.png")

# Vérifier si l'image est chargée
if image is None:
    raise ValueError("L'image n'a pas pu être chargée. Vérifie le chemin du fichier.")

# Dimensions de l'image
height_img, width_img, _ = image.shape

# Coordonnées normalisées (0 à 1)
center_x, center_y = 0.45458447804687496, 0.6135531134722222 
width, height = 0.027644230781250022, 0.08913308916666669

# Convertir en pixels
center_x = int(center_x * width_img)
center_y = int(center_y * height_img)
width = int(width * width_img)
height = int(height * height_img)

# Calculer les coins du rectangle
x_min = center_x - width // 2
y_min = center_y - height // 2
x_max = center_x + width // 2
y_max = center_y + height // 2

# Couleur (B, G, R) et épaisseur
color = (0, 255, 0)  # Vert
thickness = 2  # Bordure

# Dessiner le rectangle
cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

# Afficher l'image
cv2.imshow("Image avec case", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
