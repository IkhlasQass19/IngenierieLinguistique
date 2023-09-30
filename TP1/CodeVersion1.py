import pandas as pd
import numpy as np
def sigmoïde (x) :
    return 1 / (1 + np.exp(-x))
def derivee_sigmoide(s) :
    return s*(1-s)
def sortie_Perceptron(x,w) :
    #xb= np.append(x, 1)
    #somme = 0
    '''if len(w) != len(x):
        somme+=w[0] 
    for i in range(len(w)):
        somme += w[i] * xb[i]'''
    #print("lent of w",len(w),"kent of x",len(x))
    somme=np.dot(x,w)
    #print("result pour",x,"est :",sigmoïde(somme))
    return sigmoïde (somme)
def initialisation(k) :
    w = np.random.rand(k)  
    return w   
'''def error (x, e, w) :
    return np.power((sortie_Perceptron(x,w)-e),2)'''
def Error_gradient_local(x, e, w)  :
    s=sortie_Perceptron(x, w)
    Erreur=(s-e)*(s-e)
    DErreur=2*(s-e)
    return DErreur*derivee_sigmoide(s)*x,Erreur
def Error_gradient_Globale(X, E, w)  :
    N = len(X)
    gradient_globale = 0
    Error_globale = 0
    for i in range(N):
        gradient_local_result, erreur_locale = Error_gradient_local(X[i], E[i], w)
        gradient_globale += gradient_local_result
        Error_globale += erreur_locale
    return gradient_globale/N,Error_globale/N
def descente_gradient(w,gradient,i,delta0) :
    delta = delta0 / ((1e-4 * i )+1)   #en utilis keras
    w -= delta * gradient
    return w
def perceptron(X, E, delta0, n_iter):
    poids = []
    k = len(X[0])
    w = initialisation(k)
    erreur_minimale = float('inf')  # Initialiser l'erreur minimale à une valeur élevée

    for i in range(n_iter):
        print("=============================\npour itération :", i + 1)
        gradient, error = Error_gradient_Globale(X, E, w)
        poids.append(np.append(w.copy(), error))
        
        # Mettre à jour l'erreur minimale et le vecteur de poids optimal si une erreur plus basse est trouvée
        if error < erreur_minimale:
            poids_optimaux = w.copy()
            erreur_minimale = error

        w = descente_gradient(w, gradient, i, delta0)

    poids = np.array(poids)
    return w,error,erreur_minimale,poids_optimaux 

#old data
'''
def get_IO_data():
    #80% ENTREE 20 % TEST
    X = np.array([[1, 0, 1, 0],
                  [1, 0, 1, 1],
                  [0, 1, 0, 1]])
    E = np.array([1, 1, 0])
    return X, E'''
#data from file
def get_IO_data(file_path):
    # Charger les données à partir du fichier CSV
    data = pd.read_csv(file_path, delimiter=";")

    # Mélanger les données
    data_shuffled = data.sample(frac=1, random_state=42)  # Assure la même séquence aléatoire à chaque fois

    # Calculer la taille de l'ensemble de test (20%)
    test_size = int(0.20 * len(data_shuffled))

    # Diviser les données en ensembles d'entraînement (80%) et de test (20%)
    X = data_shuffled[["X1", "X2"]].to_numpy()
    E = data_shuffled["Y"].to_numpy()

    X_train, X_test = X[:-test_size], X[-test_size:]
    E_train, E_test = E[:-test_size], E[-test_size:]

    return X_train, X_test, E_train, E_test
def matrice_de_confusion(predictions, labels):
    tp, tn, fp, fn = 0, 0, 0, 0
    for prediction, label in zip(predictions, labels):
        if prediction == 1 and label == 1:
            tp += 1
        elif prediction == 0 and label == 0:
            tn += 1
        elif prediction == 1 and label == 0:
            fp += 1
        elif prediction == 0 and label == 1:
            fn += 1
    return tp, tn, fp, fn
#X, E = get_IO_data()
X_train, X_test, E_train, E_test = get_IO_data("Donnees_Numeriques.csv")
delta0 = np.random.rand() 
n_iter = 1000
w,error,erreur_minimale,poids_optimaux  = perceptron(X_train, E_train, delta0 , n_iter)
print("Poids finaux : ", w,"error est :",error)
print("Poids Optimaux :",poids_optimaux," error est:",erreur_minimale)
# Faites des prédictions sur l'ensemble de test
predictions_test = [sortie_Perceptron(exemple, w) for exemple in X_test]

# Appliquez un seuil pour attribuer des classes (par exemple, 0,5)
seuil = 0.5
classes_predites = [1 if prediction >= seuil else 0 for prediction in predictions_test]

# Calcul de la matrice de confusion
tp, tn, fp, fn = matrice_de_confusion(classes_predites, E_test)
# Afficher la matrice de confusion
print("Matrice de confusion :")
print("Vrais Positifs (VP):", tp)
print("Vrais Null(VN):", tn)
print("Faux Positifs (FP):", fp)
print("Faux Null (FN):", fn)
#exactitude
Exactitude=(tp+tn)/(tp + tn + fp + fn)
#precision de clase 1
precisionP = tp / (tp + fp)
#precision de clase 0
precisionN = tn / (tn + fn)
#precision  Globale:
precision_globale =(precisionP+precisionN)/2
#Rappel pour la classe 1
rappelp = tp / (tp + fn)
#Rappel pour la classe 0
rappeln = tn / (tn+ fp)
#rappel Globale:
rappel_globale =(rappelp + rappeln) / 2
#F-mesure :
f_mesure = (2 * precision_globale * rappel_globale ) / (precision_globale +rappel_globale )
print("La precision pour la classe 1 : ",precisionP, "la classe 0 : ",precisionN )
print("Le rappel  pour la classe 1 : ",rappelp , "la classe 0 : ",rappeln )
print("la precision globale :",precision_globale)
print("le rappel globale :",rappel_globale )
print("le F-mesure :",f_mesure)
print("L'exactitude :",Exactitude)