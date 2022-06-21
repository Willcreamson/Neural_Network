# -*- coding: utf-8 -*-
"""
Essayons de construire un réseau de neurone qui prédit si une image de taille
28x28 en niveau de gris est un chiffre compris entre 0 et 9 
Travail inspiré de la chaine DeepMath
Initialement nous avions pris 8 neurones sur les deux premières couches et 10 epochs pour une précision maximale de 81%
100 neurones dans la première couche et 100 epochs pour une précision  de 96,6%
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import matplotlib.image as img


### Partie A - Les données 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

#Télechargement des données
(X_train_data, Y_train_data),(X_test_data,Y_test_data) = mnist.load_data()
N = X_train_data.shape[0] # N= 60 000 données (ce sont 60 000 tableaux de taille 28x28)

#Données d'apprentissage X
X_train = np.reshape(X_train_data,(N,784)) #Vecteur image, redimensionné en array de 60 000 vecteurs lignes de 784 colonnes
X_train = X_train/255 #Normalisation ~Valeur pixel dans [O,1]

#Données d'apprentissage Y
#Vers une liste de taille 10

Y_train = to_categorical(Y_train_data,num_classes=10) #Retourne un chiffre entre (0 et 9) correspodant au chiffre sur l'image X qui lui est associé
#Le codage du chiffre s'effectue tel que le nombre 0 s'écrive (1,0,0,0,0,0,0,0,0,0)
# 1 s'écrive (0,1,0,0,0,0,0,0,0,0)
# 2 s'écrive (0,0,1,0,0,0,0,0,0,0) c'est à dire n s'écrive (0,0,..1,0,0) avec 1 à la n+1 ème position 

#Données de test 
X_test = np.reshape(X_test_data,(X_test_data.shape[0],784))
X_test = X_test/255 # On normalise chaque valeur des pixels
Y_test = to_categorical(Y_test_data,num_classes = 10)


### Partie B- Construction du réseau de neurones 

p = 100 # 100 neurones pour la première et la deuxième couche   
q = 10 # 10 neurones pour la troisième couche (couche de sortie)
modele = Sequential() #nom du réseau 

#Première couche : p neurones(entrée de dimension 784 = 28x28)
modele.add(Dense(p,input_dim=784,activation='sigmoid')) #La fonction d'activation sigmoid est la plus adéquate pour ce type de problème


#Deuxième couche : p neurones 
modele.add(Dense(p,activation='sigmoid'))

#Couche de sortie : 10 neurones (un par chiffre)
modele.add(Dense(q,activation='softmax')) #softmax fonction d'activation qui permet que la somme des probabilités en sortie donnent 1

# Choix de la méthode de descente en gradient 

modele.compile(loss='categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])
print(modele.summary()) #résume le modèle, compte automatiquement le nombre de poids qu'il reste à calculer

### Partie C- Calcul des poids par descente de gradient 
modele.fit(X_train,Y_train,batch_size=32,epochs=100)

### Partie D Resultats

resultat = modele.evaluate(X_test,Y_test)
print('Valeur de la fonction erreur sur les données de test(loss):',resultat[0])
print('Precision sur les données du test (accuracy:',resultat[1])
 
### Partie E Test avec des valeurs
prediction = modele.predict(X_train)
#accuracy = modele.accuracy(X_train)
resultat_ia = []
#precision = []
for i in range(100):   # On va tester les 100 premières valeurs
    resultat_ia.append(np.argmax(prediction[i]))
    #precision.append(np.argmax(accuracy[i]))
    plt.imshow(X_train_data[i])
    plt.title("résultat attendu: "+str(Y_train_data[i])+"; résultat obtenu (ia):"+str(resultat_ia[i]))
    plt.show()

