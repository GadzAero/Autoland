# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 21:36:20 2025

@author: pradi
"""
liste_image=[]
liste_noms_image=[]
class Image:
    def __init__(self,image):
         liste_image.append(self)
         self.filename=image[1:-2]
         liste_noms_image.append(self.filename)
         #print(self.filename+" a été crée")
         self.liste_objets=[]
         self.numero_image=image[1:-6]
         #print(self.numero_image)
    def ajouter_objet(self,objet):
        nouvelobj=Objet(objet,self)
        self.liste_objets.append(nouvelobj)
        return nouvelobj
    def afficher_liste_obj(self):
        return self.liste_objets
     
class Objet: 
    def __init__(self,nom,ton_image):
        self.nom=nom[:-1]
        self.image=ton_image
        self.coordonnees=[]
        self.classe=self.determine_classe()
        self.coordonnees_corrigees=[]
        
        #print("L'objet "+self.nom+ "a été crée")
    def ajouter_coordonnees(self,coordonnee):
        self.coordonnees.append(float(coordonnee))
        self.corriger_coordonnees()
    def afficher_liste_coord(self):
        return self.coordonnees_corrigees
    
    def corriger_coordonnees(self):
        if len(self.coordonnees)==4:
            img_height=360
            img_width=640
            epaisseur_box=5
            x1,y1,x2,y2=self.coordonnees
            coord_centre=((x1+x2)/2/img_width,(y1+y2)/2/img_height)
            width=abs(x2-x1)/img_width
            height=abs(y2-y1)/img_height         
            self.coordonnees_corrigees=[coord_centre[0],coord_centre[1],width,height]
            print(self.coordonnees_corrigees)
    def determine_classe(self):
        liste_classes=[
            [1,"LEDG"],
            [2,"REDG"],
            [3,"PAPI"],
            [4,"AimP"],
            [5,"CTL"]
            ]
        for cl in liste_classes:
            if self.nom==cl[1]:
                return cl[0]
        else: return 0
        
class Fichier:
    def __init__(self,adresse):
        i=0
        f = open(adresse)
        text=f.readlines()
        for ligne in text:
            #print(ligne, "i=",i)
            #Nouvelle image
            if ligne[0]=="<":
                i=0
                if ligne[1]!='/':
                    if ligne[1:-2] not in liste_noms_image:
                        #print("oui")
                        img_de_travail=Image(ligne)
                    i=i+1
            elif i==1:
                objet_de_travail=img_de_travail.ajouter_objet(ligne)
                i+=1
            elif i!=0 and i<6:
                objet_de_travail.ajouter_coordonnees(ligne)
                i=i+1
                

Fichier("C:/Users/pradi/Downloads/archive/labels/labels/lines/train.csv")
"""
for image in liste_image:
    print(image.numero_image)
    print(image.afficher_liste_obj())
    for objet in image.afficher_liste_obj():
        #print (objet.nom, objet.afficher_liste_coord())
        print()"""
#Ecriture de chaque fichier associé aux images 
path_ecriture="C:/Users/pradi/Downloads/archive/labels/labels/lines/labels_per_image/"
for image in liste_image:
    fichier=open(path_ecriture+image.numero_image+'.txt',"w")
    for objet in image.afficher_liste_obj():
        fichier.write(str(objet.classe)+" "+str(objet.afficher_liste_coord()[0])+" "+str(objet.afficher_liste_coord()[1])+" "+str(objet.afficher_liste_coord()[2])+" "+str(objet.afficher_liste_coord()[3])+"\n")
    fichier.close()
    