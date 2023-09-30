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
    print("result pour",x,"est :",sigmoïde(somme))
    return sigmoïde (somme)
def initialisation(k) :
    w = np.random.rand(k)  
    return w   
'''def error (x, e, w) :
    return np.power((sortie_Perceptron(x,w)-e),2)'''
def gradient_local(x, e, w)  :
    s=sortie_Perceptron(x, w)
    DErreur=2*(s-e)
    return DErreur*derivee_sigmoide(s)*x
def gradient_Globale(X, E, w)  :
    N = len(X)
    gradient_globale = 0
    for i in range(N):
        gradient_globale +=gradient_local(X[i], E[i], w)
    return gradient_globale/N
def descente_gradient(w,gradient,i,delta0) :
    delta = delta0 / ((1e-4 * i )+1)   #en utilis keras
    w -= delta * gradient
    return w
def perceptron(X, E, delta0, n_iter) :
    k = len(X[0]) 
    print (k)
    #ui
    w = initialisation(k)
    #100
    for i in range(n_iter):
        print("=============================\npour itiration :",i+1)
        gradient = gradient_Globale(X, E, w)
        #print("gradient globale est :",gradient)
        w= descente_gradient(w,gradient,i,delta0) 
    return w
def get_IO_data():
    X = np.array([[1, 0, 1, 0],
                  [1, 0, 1, 1],
                  [0, 1, 0, 1]])
    E = np.array([1, 1, 0])
    return X, E
X, E = get_IO_data()
delta0 = np.random.rand() 
n_iter = 190
poids = perceptron(X, E, delta0 , n_iter)
print("Poids finaux : ", poids)
