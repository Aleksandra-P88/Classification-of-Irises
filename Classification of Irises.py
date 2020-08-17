#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class Irys:
    
    def __init__(self,x,y):
        self.x=x
        self.y=y
                
    def split(self):
        xtrain, xtest, ytrain, ytest = train_test_split( 
        self.x, self.y, test_size = 0.25, random_state = 0) 
        return xtrain, xtest, ytrain, ytest 
        
    def scaler(self,xtrain, xtest, ytrain, ytest):
        sc = StandardScaler() 
        xtrain = sc.fit_transform(xtrain)  
        xtest = sc.transform(xtest) 
        return xtrain, xtest
    
    def LogReg(self,xtrain, xtest, ytrain):
        classifier = LogisticRegression(random_state = 0) 
        classifier.fit(xtrain, ytrain) 
        y_pred = classifier.predict(xtest) 
        return y_pred
    
    def Conf(self,ytest,y_pred):
        
        cm = confusion_matrix(ytest, y_pred) 
        
        #print ("Confusion Matrix : \n", cm) 
        
        A_flower=('Actual Setosa','Actual Veriscolor','Actual Virginica')
        P_flower=('Predict Setosa','Predict Veriscolor','Predict Virginica')
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(cm)
        ax.grid(False) 
        ax.set_yticklabels(A_flower, fontsize=12, color='black')
        ax.set_xticklabels(P_flower, fontsize=12, color='black')
        ax.xaxis.set(ticks=range(3))
        ax.yaxis.set(ticks=range(3))
        ax.set_ylim(2.5, -0.5)
         
        for i in range(3):
            
            for j in range(3):
                
                ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
                
        plt.show()      
        
        
    def Acc(self,y_pred, ytest):
        print ("Accuracy : ", accuracy_score(ytest,y_pred)) 
        
    def Report(self,y_pred, ytest):
        print('Report : ',classification_report(ytest,y_pred ) )
        
        
# loading the iris dataset 
iris = datasets.load_iris() 
  
# X -> features, y -> label 
X = iris.data 
y = iris.target       
        
I = Irys(iris.data,iris.target)
xtr,xte,ytr,yte=I.split()

xtr_2,xte_2=I.scaler(xtr,xte,ytr,yte)

ypre=I.LogReg(xtr_2,xte_2,ytr)  

I.Conf(yte,ypre)

I.Acc(ypre,yte)

I.Report(ypre,yte)
      

 



    
 


    


# In[ ]:





# In[ ]:




