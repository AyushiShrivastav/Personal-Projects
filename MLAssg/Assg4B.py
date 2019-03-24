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

    X = getFeatureData("C:\\Users\\CZL\\Desktop\\climate.data")               #"C:\\Users\\CZL\\Desktop\\breast_cancer.data"
    r = getLabelData("C:\\Users\\CZL\\Desktop\\climate.trainlabels.0")                 #sys.argv[2]

    test=[]
    train=[]
    testlabels={}
    trainlabels={}
    
    #Adding bias term to the dataset
    for i in range(len(X)):
        col=X[i]
        col.insert(0, 1)

    #Dividing data & labels into testing and training 

    i = 0
    for key,value in r.items():
        if i>len(r):
            break
        elif i>len(r)/2:
            testlabels[i-1-len(r)/2]=value
            test.append(X[key])
            i+=1
        else:
            trainlabels[i]=value
            train.append(X[key])
            i+=1
     
    i=1
    prevE=0.000000001
    w=[]
    N=0.01     #Value of eta, the learning rate
    vect=[]
    E=[]
    lamb=0
    reg=0
    ewr=0
    err=1
    
    n=len(train)
    dn=len(train[0])

    for i in range(dn):
        w.append(1)
    
    #initialising W = [w0, w1, ...w(d+1)] as values between 0 and 0.01
    #for i in range(dn+1):
     #   w.append(random.random()*0.01)
    
    while(abs(err-prevE)>=0.0000001):
        
        prevE = err
        i=0
        vect=[]
        #As vect = X*wt
        for i in range(len(train)):
            dot_p = dotProduct(train[i], w)
            vect.append(sigmoid(dot_p))

        reg=0
        #Regularized term
        for i in range(len(w)):
            reg+= (w[i]**2)

        reg=reg*(lamb/2)
        
        err=0
        i=0
        #Error is calculated by the formula
        for i in range(len(trainlabels)):
            
            E.append((-1*trainlabels[i]*math.log(vect[i]))-((1-trainlabels[i])*math.log(1-vect[i])))
            err+=E[i]
        ewr=err
        err+=reg
        
        #Stopping condition is PreviousError - CurrentError <= 0.0000001, as specified in the assignment question. 
        #if(abs(err-prevE)<=0.0000001):
        #    break
        
        delf=[]
        temp=0
        
        i=0
        #Calculating dE(0)/dw to update w
        for key,value in trainlabels.items():
            temp+=(trainlabels[key]-vect[key])
        delf.append(-1*temp)
        
        #Calculating dE/dw to update w
        for j in range(1, dn):
            temp=0
            for key,value in trainlabels.iteritems():
                temp+= X[key][j]*(trainlabels[key]-vect[i])
            delf.append(-1*temp + lamb*w[j])
        
        #Updating w...
        for k in range(dn):
            w[k]=w[k]+(-1*delf[k]*N)

    print "Error = ", ewr

    wt=[]
    for i in range(dn-1):
        wt.append(w[i+1])
             
    #Calculating Hyperplane distance from origin, i.e, (w0/||w||)
    print("||w|| = ", norm(wt))
    
    distance = abs(w[0]/norm(wt))
    print("Distance of the Hyperplane from the Origin is ", distance)
    
    cmisses=0

    for i in range(len(train)):
        temp=sigmoid(dotProduct(w,train[i]))
        if (temp>0.5):
          #print(" 1 " + str(i), trainlabels[i])
          if trainlabels[i]!=1:
              cmisses+=1
        else:
          #print(" 0 " + str(i), trainlabels[i])
          if trainlabels[i]!=0:
              cmisses+=1
    print("Training Missclassifications = ", cmisses)

    tmisses=0
    err=0
    E=[]
    vect=[]
    err=0
    for i in range(len(test)):
        dot_p = dotProduct(test[i], w)
        vect.append(sigmoid(dot_p))
        E.append((-1*testlabels[i]*math.log(vect[i]))-((1-testlabels[i])*math.log(1-vect[i])))
        err+=E[i]
    print("Testing Error is ", err)
    print("Total error is", err+ewr)
    i=0
    for i in range(len(test)):
        temp=sigmoid(dotProduct(w,test[i]))
        if (temp>0.5):
          #print(" 1 ", str(i), testlabels[i])
          if testlabels[i]!=1:
              tmisses+=1
        else:
          #print(" 0 ", str(i), testlabels[i])
          if testlabels[i]!=0:
              tmisses+=1
    print("Cross validation missclassifications = ", tmisses)
    print("Total misclassifications", cmisses+tmisses)

prog()
