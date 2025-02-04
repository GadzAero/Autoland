import cv2
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


model = torch.hub.load("ultralytics/yolov5", "custom", path="C:/Users/pradi/Downloads/archive/yolov5-master/runs/train/exp/weights/best.pt", force_reload=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


video_path = r"C:/Users/pradi/OneDrive - ensam.eu/Projet Dassault/3-DEVELOPPEMENT/3- Reconnaissance image/Cut1.mp4"
video = cv2.VideoCapture(video_path)


if not video.isOpened():
    print("Erreur lors de l'ouverture de la vidéo.")
    exit()

cv2.namedWindow("YOLOv5 Inferences", cv2.WINDOW_NORMAL)  # Créer une fenêtre pour afficher les résultats

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break  # Arrêter si la vidéo est terminée

    # Effectuer l'inférence sur chaque frame
    results = model(frame)
    boxes=[]
    for det in results.xywh[0]:  # Results pour la première image
        xmin, ymin, xmax, ymax, confidence, class_idx = det.tolist()
        """
        print(f"Detection: {class_idx} (Confidence: {confidence:.2f})")
        print(f"Bounding box: ({xmin}, {ymin}, {xmax}, {ymax})")
        """
        boxes.append([xmin,ymin,xmax,ymax,class_idx])
    #print (results.xywh[0],boxes)
    
    
    # Dessiner les boîtes de détection sur l'image originale
    for box in boxes:
        x_cent, y_cent,width,height, classe = box
        x_cent=int(x_cent)
        y_cent=int(y_cent)
        width,height=int(width),int(height)
        xmin=int(x_cent-width/2)
        xmax=int(x_cent+width/2)
        ymin=int(y_cent-height/2)
        ymax=int(y_cent+height/2)
        color=[(255, 0, 0),(255, 0, 0),(0, 255, 0),(0, 0, 255),(255, 255, 0),(255, 0, 255)]
        liste_classes=[
                    [0,"Unknown"],
                    [1,"LEDG"],
                    [2,"REDG"],
                    [3,"PAPI"],
                    [4,"AimP"],
                    [5,"CTL"]
                    ]
        #Boxes
        #cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color[int(classe)], 2)  # Boîte rouge
        #cv2.putText(frame,str(liste_classes[int(classe)][1]), (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[int(classe)], 2)
        
        #Dessin des diagonales des box left, right et centre
        #Comme les lasses gauches et droites s'emmelent, on va pas s'y fier et dire plutot si c'est à droite ou a gauche
        height_img, width_img, _ = frame.shape
        x_centre_image=width_img/2 
        y_centre_image=height_img/2 
        
        if int(classe)==1 or int(classe)==2:
            if x_cent>x_centre_image:
                #gauche
                cv2.line(frame, (xmin,ymin),(xmax,ymax), color[1], 3)
            else:
                #droite
                cv2.line(frame, (xmax,ymin),(xmin,ymax), color[2], 3)
        elif int(classe)==5:
            cv2.line(frame, (x_cent,ymin),(x_cent,ymax), color[int(classe)], 3)
        
    # Afficher l'image avec les annotations
    cv2.imshow("YOLOv5 Inferences", frame)

    # Sortie avec 'x' pour fermer la fenêtre
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

video.release()
cv2.destroyAllWindows()
