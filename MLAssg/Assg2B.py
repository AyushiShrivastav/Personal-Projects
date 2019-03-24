
import math
import random
import sys
'''Get feature data from file as a matrix with a row per data instance'''
def getFeatureData(featureFile):
    x=[]
    dFile = open(featureFile, 'r')
    for line in dFile:
        row = line.split()
        rVec = [float(item) for item in row]
        x.append(rVec)
    dFile.close()
    return x

'''Get label data from file as a dictionary with key as data instance index and value as the class index'''
def getLabelData(labelFile):
    lFile = open(labelFile, 'r')
    lDict = {}
    for line in lFile:
        row = line.split()
        lDict[int(row[1])] = int(row[0])
    lFile.close()
    return lDict

def sigmoid(y):
    if isinstance(y,list):
        return [sigmoid(item) for item in y]
    else:
        return 1.0/(1+math.exp((-y)))
    
def dotProduct(u,v):
    sum = 0
    for i in range(len(u)):
        sum += u[i] * v[i]
    return sum

def norm(v):
    return math.sqrt(dotProduct(v,v)) 

def matrixTranspose(a):
    return[[a[i][j] for i in range(len(a))] for j in range(len(a[0]))]

def prediction(X, w):
    y=[]
    sums=w[0]
    #As y =  sigmoid(w0+ Xw)
    for i in range(len(X)):
        y.append(dotProduct(w, X[i]))
    y=sigmoid(y)
    for i in range(len(y)):
        sums+=y[i]
        if(sums<=1):
            print 'Prediction ',i+1,': class0'
        else:
            print 'Prediction ',i+1,': class1'
    
def prog():

    X = getFeatureData(sys.argv[1])                 #"C:\\Users\\CZL\\Desktop\\data1.data"
    r = getLabelData(sys.argv[2])                 #C:\\Users\\CZL\\Desktop\\data1.data

    i=1
    pe=0.01
    w=[]
    N=0.001     #Value of eta, the learning rate
    d=[]
    dw=[]
    dE=[]
    e=0

    n=len(X)
    dn=len(X[0])

    #Adding bias term to the dataset
    for i in range(len(X)):
        col=X[i]
        col.insert(0, 1)
   
    Xt=matrixTranspose(X) 

    #initialising W = [w0, w1, ...w(d+1)] as values between 0 and 0.01
    for i in range(dn+1):
        w.append(random.random()*0.01)

    while(1):

        #As y =  sigmoid(w0+ Xw)
        y=[]
        for i in range(len(X)):
            y.append(dotProduct(w, X[i])+w[0])
        y=sigmoid(y)
        
        #Error is calculated by multiplying d and transpose of d, i.e. E = -Xt*(r-y)
        et=[]
        et = [r[i]*math.log(Decimal(y[i]))+(1-r[i])*math.log(Decimal(abs(1-y[i]))) for i in range(len(r))]
        
        for i in range(len(et)):
            e+=et[i]
        
        #Stopping condition is PreviousError - CurrentError <= 0.001 (theta), as specified in the assignment question
        #Using normalisation to for the stopping condition
            
        if(abs(abs(pe)-abs(e)/e)<=0.001):
            break

        #Calculating dE and dw to update w
        dE=[]
        temp=[(r[i]-y[i]) for i in range(len(r))]
        
        for j in range(dn):
            dE.append(dotProduct(Xt[j], temp))
        dw = [i*N for i in dE]
        
        #Updating w...
        i=0
        w=[w[i]+dw[i] for i in range(dn)]
        pe = e


    wt=[]
    for i in range(dn-1):
        wt.append(w[i+1])
    #Calculating Hyperplane distance from origin, i.e, (w0/||w||)
    distance = abs(w[0]/norm(wt))
    print "Distance of the Hyperplane from the Origin is ", distance

    prediction(X, w)
    
prog()
