'''
TD 5 : Apprentissage supervisé - k plus proches voisins
Réalisé par : Alban et Morgane (TD F)
'''

import math
import random
import tkinter as tk

#Charger les données dans un tableau iris
def ChargementDonnees():
    f=open(r'iris.data','r')
    iris=[[0] * 5 for i in range(150)]

    j=0
    for line in f:
        line=line.split(',')
        #print(line)
        k=0
        for i in line:
            if k!=4:
                iris[j][k] = float(i)
            else: #sur la 5ème colonne (k=4), on a un string
                iris[j][k]=i
            #print(iris[j][k])
            k=k+1
        j=j+1
    f.close()
    return iris

#Division des données pour avoir 80% de données d'apprentissage et 20% de données test
def DivisionDonnees(iris):
    irisApprentissage=[]
    irisTest=[]

    TossACoin=0
    for i in range(len(iris)):
        TossACoin=random.randint(0,100)
        if (TossACoin<80 and len(irisApprentissage)<int(len(iris)*0.80)):
            irisApprentissage.append(iris[i])
        elif (len(irisTest)<len(iris)-int(len(iris)*0.80)):
            irisTest.append(iris[i])

    return irisApprentissage, irisTest

#Calculer la distance euclidienne entre deux données
def DistanceEuclidienne(d1,d2):
    somme=0
    for i in range(4):
        somme=somme+(d1[i]-d2[i])**2
    distance=math.sqrt(somme)
    return distance

#Liste des distances entre la donnée de test avec les autres données
def ListeDistance(donneeTest, irisApprenti):
    listeDistances = [[0] * 3 for i in range(len(irisApprenti))]
    for i in range(len(irisApprenti)):
        listeDistances[i][0]=donneeTest
        listeDistances[i][1]=irisApprenti[i]
        listeDistances[i][2]=DistanceEuclidienne(donneeTest,irisApprenti[i])
    return listeDistances

#Trier la liste dans l'ordre croissant Ã  l'aide du tri a bulles
def OrdreCroissant(liste,numeroColonne):
    n=len(liste)
    for i in range (n):
        for j in range(0, n-i-1):
            if liste[j][numeroColonne]>liste[j+1][numeroColonne]:
                liste[j],liste[j+1]=liste[j+1],liste[j]

#renvoie une liste avec les k données dont la distance euclidienne est la plus faible
def Ktop(listeDistance, k):
    listeDuTopk = []
    for i in range(k):
        listeDuTopk.append(listeDistance[i][1][4])
    return listeDuTopk

#renvoie la classification de notre donnée (en regardant la fréquence des classifications dans la liste des Ktop)
def PredictionClassification(listeKtop):
    listePrediction=[[0] * 2 for i in range(len(listeKtop))]
    n = 0
    for i in range(len(listeKtop)):
        present=False
        j=0
        while(j<len(listePrediction)):
            if (listeKtop[i]==listePrediction[j][0]):
                listePrediction[j][1]+=1
                present=True
                break
            j=j+1
        if (present==False):
            listePrediction[n][0]=listeKtop[i]
            listePrediction[n][1]=listePrediction[n][1]+1
            n=n+1

        OrdreCroissant(listePrediction,1) #on trie la liste dans l'ordre croissant des fréquences donc la fréquence la plus élevée se situe en derniÃ¨re position
        prediction=listePrediction[len(listePrediction)-1][0] #La fréquence la plus élevée se situe en dernière position

    return prediction

#retourne True si notre prédiction est bonne, False sinon
def PredictionVraie(donneeTest, irisApprenti,k):
    prediction=False
    listeDistance=ListeDistance(donneeTest,irisApprenti)
    OrdreCroissant(listeDistance,2)
    if PredictionClassification(Ktop(listeDistance,k))==donneeTest[4]:
        prediction=True
    return prediction


#Pour remplir notre matrice de confusion
def ConversionClassificationMatrice(classification):
    position=-1
    if classification=="Iris-setosa\n":
        position=0
    elif classification=="Iris-versicolor\n":
        position=1
    elif classification=="Iris-virginica\n":
        position=2
    return position

#Matrice de confusion
def MatriceDeConfusion(iris,k):
    matrice=[[0] * 3 for i in range(3)]
    irisApprenti, irisTest = DivisionDonnees(iris)

    for i in range (len(irisTest)):
        listeDistance=ListeDistance(irisTest[i],irisApprenti)
        OrdreCroissant(listeDistance,2)
        listePredi = Ktop(listeDistance, k)
        l=ConversionClassificationMatrice(PredictionClassification(listePredi))
        m=ConversionClassificationMatrice(irisTest[i][4])
        matrice[l][m]+=1
    return matrice

#Pourcentage de prédictions bonnes
def PourcentageDeVerite(matriceDeConfusion):
    somme = 0
    for i in range(len(matriceDeConfusion)):
        somme = somme + matriceDeConfusion[i][i]
    pourcentage = (somme / 30) * 100
    return pourcentage

#FenÃªtre qui affiche la matrice de confusion
def Widget(matriceDeConfusion):
    fenetre = tk.Tk()

    Reel = tk.Label(fenetre,text="REEL")
    Reel.grid(row=0,column=3)
    SetosaReel=tk.Label(fenetre,text="Setosa")
    SetosaReel.grid(row=1,column=2)
    VersicolorReel = tk.Label(fenetre, text="Versicolor")
    VersicolorReel.grid(row=1,column=3)
    VirginicaReel = tk.Label(fenetre, text="Virginica")
    VirginicaReel.grid(row=1,column=4)
    Prediction =tk.Label(fenetre,text="PREDICTION")
    Prediction.grid(row=3)
    SetosaPrediction=tk.Label(fenetre,text="Setosa")
    SetosaPrediction.grid(row=2,column=1)
    VersicolorPrediction = tk.Label(fenetre, text="Versicolor")
    VersicolorPrediction.grid(row=3,column=1)
    VirginicaPrediction = tk.Label(fenetre, text="Virginica")
    VirginicaPrediction.grid(row=4,column=1)

    SeSe=tk.Label(fenetre,text=matriceDeConfusion[0][0],fg="green")
    SeSe.grid(row=2,column=2)

    SeVe=tk.Label(fenetre,text=matriceDeConfusion[0][1],fg="red")
    SeVe.grid(row=2,column=3)

    SeVi=tk.Label(fenetre,text=matriceDeConfusion[0][2],fg="red")
    SeVi.grid(row=2,column=4)

    VeSe=tk.Label(fenetre,text=matriceDeConfusion[1][0],fg="red")
    VeSe.grid(row=3,column=2)

    VeVe=tk.Label(fenetre,text=matriceDeConfusion[1][1],fg="green")
    VeVe.grid(row=3,column=3)

    VeVi=tk.Label(fenetre,text=matriceDeConfusion[1][2],fg="red")
    VeVi.grid(row=3,column=4)

    ViSe=tk.Label(fenetre,text=matriceDeConfusion[2][0],fg="red")
    ViSe.grid(row=4,column=2)

    ViVe=tk.Label(fenetre,text=matriceDeConfusion[2][1],fg="red")
    ViVe.grid(row=4,column=3)

    ViVi=tk.Label(fenetre,text=matriceDeConfusion[2][2],fg="green")
    ViVi.grid(row=4,column=4)

    fenetre.mainloop()


#TEST
iris=ChargementDonnees()
matriceDeConfusion=MatriceDeConfusion(iris,3)
print(matriceDeConfusion)
print("Le pourcentage de vérité est de : ", round(PourcentageDeVerite(matriceDeConfusion),3), "%")
Widget(matriceDeConfusion)