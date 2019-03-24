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

def dotProduct(u,v):
    sum = 0
    for i in range(len(u)):
        sum += u[i] * v[i]
    return sum

def norm(v):
    return math.sqrt(dotProduct(v,v)) 

def matrixTranspose(a):
    return[[a[i][j] for i in range(len(a))] for j in range(len(a[0]))]

def prog():

    X = getFeatureData("C:\\Users\\CZL\\Desktop\\data1.data")               #"C:\\Users\\CZL\\Desktop\\breast_cancer.data"
    r = getLabelData("C:\\Users\\CZL\\Desktop\\data1.labels")                 #sys.argv[2]

    i=1
    pe=0.01
    w=[]
    N=0.001     #Value of eta, the learning rate
    d=[]
    dw=[]
    dE=[]
    e=0

    #Adding bias term to the dataset
    for i in range(len(X)):
        col=X[i]
        col.insert(0, 1)
    
    n=len(X)
    dn=len(X[0])
    Xt=matrixTranspose(X)
    #w.append(1)
    #w.append(1)
    #w.append(1)
    #initialising W = [w0, w1, ...w(d+1)] as values between 0 and 0.01
    for i in range(dn+1):
        w.append(random.random()*0.01)

    while(True):
        d=[]
        i=0
        #As d = r - Xw
        for i in range(n):
            print(dotProduct(X[i],w))
            d.append(r[i]*(dotProduct(X[i],w)+w[0]))
            if(d[i]>0):
                e+=(1-d[i])
                
        #Stopping condition is PreviousError - CurrentError <= 0.001, as specified in the assignment question. So, try adding normalisation/Scaling
        if(abs(e-pe)<=0.000000001):
            break
        dE=[]
        dw=[]
        #Calculating dE and dw to update w
        wd=[]
        for j in range(dn):
            if(d[i]<1):
                wd.append(r[i])
            else:
                wd.append(0)
        dE = [wd[i]*X[i] for i in dE]
        dw = [-1*N*dE for i in dE]

        i=0
        #Updating w...
        w=[w[i]+dw[i] for i in range(len(dw))]
        #print w
        pe = e

    wt=[]
    for i in range(dn-1):
        wt.append(w[i+1])
    print 'w[0]= ', w[0], 'wt=', wt
    #Calculating Hyperplane distance from origin, i.e, (w0/||w||)
    distance = abs(w[0]/norm(wt))
    print "Distance of the Hyperplane from the Origin is ", distance

prog()
