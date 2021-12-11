'''
Classification Challenge
Morgane - Alban - TDF
'''

import math
import random

#Charger les données dans un tableau donnees
def ChargementDonnées(taille, nom):
    f=open(nom,'r')
    donnees=[[0] * 5 for i in range(taille)]

    j=0
    for line in f:
        line=line.split(';')
        k=0
        for i in line:
            if k!=4:
                donnees[j][k] = float(i)
            else: #sur la 5ème colonne (k=4), on a un string
                donnees[j][k]=i
            k=k+1
        j=j+1
    f.close()
    return donnees

#Division des données pour avoir 80% de données d'apprentissage et 20% de données test
def DivisionDonnées(donnees):
    donneesApprentissage=[]
    donneesTest=[]

    TossACoin=0
    for i in range(len(donnees)):
        #Soit TossACoin une variable aléatoire suivant une loi uniforme sur [0,100]
        TossACoin=random.randint(0,100)
        if (TossACoin<80 and len(donneesApprentissage)<int(len(donnees)*0.80)):
            donneesApprentissage.append(donnees[i])
        elif (len(donneesTest)<len(donnees)-int(len(donnees)*0.80)):
            donneesTest.append(donnees[i])

    return donneesApprentissage, donneesTest

#Calculer la distance euclidienne entre deux données
def DistanceEuclidienne(d1,d2):
    somme=0
    for i in range(4):
        somme=somme+(d1[i]-d2[i])**2
    distance=math.sqrt(somme)
    return distance

#Calculer la distance de Manhattan entre deux données
def DistanceManhattan(d1,d2):
    distance=0
    for i in range(4):
        distance=distance+abs(d1[i]-d2[i])
    return distance

#Liste des distances entre la donnée de test avec les autres données
def ListeDistance(donnéeTest, donneesApprenti):
    listeDistances = [[0] * 3 for i in range(len(donneesApprenti))]
    for i in range(len(donneesApprenti)):
        listeDistances[i][0]=donnéeTest
        listeDistances[i][1]=donneesApprenti[i]
        #listeDistances[i][2]=DistanceEuclidienne(donnéeTest,donneesApprenti[i])
        listeDistances[i][2]=DistanceManhattan(donnéeTest,donneesApprenti[i])
    return listeDistances

#renvoie une liste avec les k données dont la distance de Manhattan est la plus faible
def Ktop(listeDistance, k):
    listeDuTopk = []
    for i in range(k):
        listeDuTopk.append(listeDistance[i][1][4])
    return listeDuTopk

#renvoie la classification de notre donnée (en regardant la fréquence des classifications dans la liste des Ktop)
def PrédictionClassification(listeKtop):
    listePrediction=[[0] * 2 for i in range(len(listeKtop))]
    n = 0
    for i in range(len(listeKtop)):
        present=False
        j=0
        while(j<len(listePrediction)):
            if (listeKtop[i]==listePrediction[j][0]): #si la prédiction est déjà présente dans notre tableau
                listePrediction[j][1]+=1 #on incrémente la fréquence
                present=True
                break
            j=j+1
        if (present==False): #la prédiction n'est pas déjà présente dans notre tableau listePrediction
            listePrediction[n][0]=listeKtop[i]
            listePrediction[n][1]=listePrediction[n][1]+1
            n=n+1
        listePrediction.sort(key=lambda y: y[1]) #on trie la liste dans l'ordre croissant des fréquences donc la fréquence la plus élevée se situe en dernière position
        prediction=listePrediction[len(listePrediction)-1][0] #La fréquence la plus élevée se situe en dernière position

    return prediction

#retourne True si notre prédiction est bonne, False sinon
def PrédictionVraie(donnéeTest, donneesApprenti,k):
    prédiction=False
    listeDistance=ListeDistance(donnéeTest,donneesApprenti)
    listeDistance.sort(key=lambda y: y[2])
    if PrédictionClassification(Ktop(listeDistance,k))==donnéeTest[4]:
        prédiction=True
    return prédiction

#Pour remplir notre matrice de confusion
def ConversionClassificationMatrice(classification):
    position=-1
    if classification=="A\n":
        position=0
    elif classification=="B\n":
        position=1
    elif classification=="C\n":
        position=2
    elif classification=="D\n":
        position=3
    elif classification=="E\n":
        position=4
    elif classification=="F\n":
        position=5
    elif classification=="G\n":
        position=6
    elif classification=="H\n":
        position=7
    elif classification=="I\n":
        position=8
    elif classification=="J\n":
        position=9
    return position

#Matrice de confusion
def MatriceDeConfusion(donnees,donneesTest, k):
    matrice=[[0] * 10 for i in range(10)]
    donneesApprenti=donnees
    for i in range (len(donneesTest)):
        listeDistance=ListeDistance(donneesTest[i],donneesApprenti)
        listeDistance.sort(key=lambda y: y[2])
        listePredi = Ktop(listeDistance, k)
        l=ConversionClassificationMatrice(PrédictionClassification(listePredi))
        m=ConversionClassificationMatrice(donneesTest[i][4])
        matrice[l][m]+=1
    return matrice

#Matrice de confusion qui utilise la division des données
def MatriceDeConfusion2(donnees, k):
    matrice=[[0] * 10 for i in range(10)]
    donneesApprenti, donneesTest = DivisionDonnées(donnees)

    for i in range (len(donneesTest)):
        listeDistance=ListeDistance(donneesTest[i],donneesApprenti)
        listeDistance.sort(key=lambda y: y[2])
        listePredi = Ktop(listeDistance, k)
        l=ConversionClassificationMatrice(PrédictionClassification(listePredi))
        m=ConversionClassificationMatrice(donneesTest[i][4])
        matrice[l][m]+=1
    return matrice

#Pourcentage de prédictions bonnes
def PourcentageDeVérité(matriceDeConfusion, nombreDeDonnéesTest):
    somme = 0
    for i in range(len(matriceDeConfusion)):
        somme = somme + matriceDeConfusion[i][i]
    pourcentage = (somme / nombreDeDonnéesTest) * 100
    return pourcentage


#TESTS
#TEST avec division des données : on utilise seulement data.csv qu'on divise en 80% de données d'apprentissage et 20% de données test
def TestDivisionDesDonnees(k):
    donnees=ChargementDonnées(2878, "data.csv")
    matriceDeConfusion=MatriceDeConfusion2(donnees, k)
    print(matriceDeConfusion)
    print("Le pourcentage de vérité est de : ", round(PourcentageDeVérité(matriceDeConfusion, len(donnees)*0.2),k), "%")

#TEST sans division des données : data.csv contient nos données d'apprentissage et preTest contient nos données de test
def TestSansDivisionDesDonnees(k):
    donnees=ChargementDonnées(2878, "data.csv")
    donneesTest=ChargementDonnées(600, "preTest.csv")
    matriceDeConfusion=MatriceDeConfusion(donnees, donneesTest, k)
    print(matriceDeConfusion)
    print("Le pourcentage de vérité est de : ", round(PourcentageDeVérité(matriceDeConfusion, len(donneesTest)),k), "%")


#Création d'un fichier contenant nos prédictions sur les données finalTest.csv
def ClassificationChallenge(k):
    donneesApprenti=ChargementDonnées(2878, "data.csv") #on charge nos données d'apprentissage
    donneesTest = ChargementDonnées(2000, "finalTest.csv") #on charge nos données à prédire = les données test
    fichierPredictions = open(r'Predictions.txt', 'w') #on crée un fichier qui contiendra nos prédictions
    for i in range(len(donneesTest)):
        listeDistance = ListeDistance(donneesTest[i], donneesApprenti)
        listeDistance.sort(key=lambda y: y[2])
        listePredi = Ktop(listeDistance, k)
        prediction=PrédictionClassification(listePredi)
        fichierPredictions.write(prediction)
    fichierPredictions.close()

#En utilisant l'ensemble des données de data.csv et de preTest.csv comme données d'apprentissage
def ClassificationChallengeFullDonnée(k):
    donneesApprenti1=ChargementDonnées(2878, "data.csv")
    donneesApprenti2=ChargementDonnées(600,"preTest.csv")
    donneesApprenti = donneesApprenti1+donneesApprenti2
    print(len(donneesApprenti))
    donneesTest = ChargementDonnées(2000, "finalTest.csv")
    fichierPredictions = open(r'Predictions2.txt', 'w')
    for i in range(len(donneesTest)):
        listeDistance = ListeDistance(donneesTest[i], donneesApprenti)
        listeDistance.sort(key=lambda y: y[2])
        listePredi = Ktop(listeDistance, k)
        prediction=PrédictionClassification(listePredi)
        fichierPredictions.write(prediction)
    fichierPredictions.close()

#Comparaison de nos 2 fichiers de prédiction
def Comparaison():
    donnees1 = open(r'Predictions.txt', 'r')
    donnees2 = open(r'Predictions2.txt', 'r')
    donnees1liste=[]
    donnees2liste=[]
    cpt=0
    for line in donnees1:
        donnees1liste.append(line)
    for line in donnees2:
        donnees2liste.append(line)
    for i in range(len(donnees1liste)):
        if donnees1liste[i]!=donnees2liste[i]:
            cpt+=1
    print("nombre d'erreurs : ",cpt) #on trouve 52 différences entre le 2 fichiers

#L'instruction qui nous a permis de générer le fichier contenant nos prédictions pour les données de finalTest.csv
k=6
ClassificationChallenge(k)

#ClassificationChallengeFullDonnée(6)
#Comparaison()

#TestDivisionDesDonnees(6)
#TestSansDivisionDesDonnees(6)


