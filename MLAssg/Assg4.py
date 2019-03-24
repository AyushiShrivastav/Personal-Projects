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

'''Sigmoid function'''
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

def matrixTranspose(a):
    return[[a[i][j] for i in range(len(a))] for j in range(len(a[0]))]

def norm(v):
    return math.sqrt(dotProduct(v,v)) 

def prog():

    X = getFeatureData("C:\\Users\\CZL\\Desktop\\ionosphere.data")               #"C:\\Users\\CZL\\Desktop\\breast_cancer.data"
    r = getLabelData("C:\\Users\\CZL\\Desktop\\ionosphere.trainlabels.0")                 #sys.argv[2]

    Xt=matrixTranspose(X)
    
    i=1
    prevE=0.000000001
    w=[]
    N=0.01     #Value of eta, the learning rate
    vect=[]
    dw=[]
    dE=[]
    E=[]
    lamb=0
    reg=0

    #Adding bias term to the dataset
    for i in range(len(X)):
        col=X[i]
        col.insert(0, 1)
    
    n=len(X)
    dn=len(X[0])

    for i in range(len(X[0])):
        w.append(1)
    
    #initialising W = [w0, w1, ...w(d+1)] as values between 0 and 0.01
    #for i in range(dn+1):
     #   w.append(random.random()*0.01)
    err=1
    
    while(abs(err-prevE)>=0.0000001):
        
        prevE = err
        
        vect=[]
        #As vect = X*wt
        for key,value in r.iteritems():
            dot_p = dotProduct(X[key], w)
            vect.append(sigmoid(dot_p))

        reg=0
        #Regularized term
        for i in range(len(w)):
            reg+= (w[i]**2)

        reg=reg*(lamb/2)

        err=reg
        i=0
        #Error is calculated by the formula
        for key,value in r.iteritems():
            E.append((-1*r[key]*math.log(vect[i]))-((1-r[key])*math.log(1-vect[i])))
            err+=E[i]
            i+=1
            
        #Stopping condition is PreviousError - CurrentError <= 0.0000001, as specified in the assignment question. 
        #if(abs(err-prevE)<=0.0000001):
        #    break
        
        delf=[]
        temp=0
        i=0
        
        #Calculating dE(0)/dw to update w
        for key,value in r.iteritems():
            temp+=(r[key]-vect[i])
            i+=0
        delf.append(-1*temp)
        
        #Calculating dE/dw to update w
        for j in range(1, dn):
            temp=0
            for key,value in r.iteritems():
                temp+= X[key][j]*(r[key]-vect[i])
            delf.append(-1*temp + lamb*w[j])
        
        #Updating w...
        for k in range(dn):
            w[k]=w[k]+(-1*delf[k]*N)
 
    wt=[]
    for i in range(dn-1):
        wt.append(w[i+1])
             
    #Calculating Hyperplane distance from origin, i.e, (w0/||w||)
    print "||w|| = ", norm(wt)
    distance = abs(w[0]/norm(wt))
    print "Distance of the Hyperplane from the Origin is ", distance

    for i in range(len(X)):
      if r.get(i,None)==None :
        temp=dotProduct(w,X[i])
        if (temp>0):
          print(": 1 " + str(i))
        else:
          print(": 0 " + str(i))
          
prog()
