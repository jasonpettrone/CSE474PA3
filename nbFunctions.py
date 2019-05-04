from sklearn.base import BaseEstimator
import numpy as np
import scipy.stats as stats

# For this assignment we will implement the Naive Bayes classifier as a
# a class, sklearn style. You only need to modify the fit and predict functions.
# Additionally, implement the Disparate Impact measure as the evaluateBias function.
class NBC(BaseEstimator):
    '''
    (a,b) - Beta prior parameters for the class random variable
    alpha - Symmetric Dirichlet parameter for the features
    '''

    def __init__(self, a=1, b=1, alpha=1):
        self.a = a
        self.b = b
        self.alpha = alpha
        self.params = None
        
    def get_a(self):
        return self.a

    def get_b(self):
        return self.b

    def get_alpha(self):
        return self.alpha

    # you need to implement this function

    def fit(self,X,y):
        '''
        This function does not return anything
        
        Inputs:
        X: Training data set (N x d numpy array)
        y: Labels (N length numpy array)
        '''
        a = self.get_a()
        b = self.get_b()
        alpha = self.get_alpha()
        self.__classes = np.unique(y)

        N1 = 0
        N2 = 0
        N = X.shape[0]
        for i in range(len(y)):
            if (y[i] == 1):
                N1 += 1
            else:
                N2 += 1

        P1 = (N1 + a) / (N + a + b) # Equation 5
        P2 = 1 - P1

        #numSubFeatures = [4, 1, 5, 11, 1, 5, 5, 1, 3, 1, 4, 3, 3, 1, 4, 1, 2, 2]
        numSubFeatures = [4, 5, 5, 11, 9, 5, 5, 4, 3, 4, 4, 3, 3, 4, 4, 2, 2, 2]
        p1 = P1 * np.ones(shape=(11, 18))
        p2 = P2 * np.ones(shape=(11, 18))
        for i in range(0, X.shape[1]):
            for j in range(0, numSubFeatures[i]):
                total = 0
                p1Total = 0
                for k in range(0, X.shape[0]):
                    if(X[k,i] == j+1):
                        total += 1
                        if(y[k] == 1):
                            p1Total += 1
                if(total == 0):
                    p1[j, i] = P1
                    p2[j, i] = P2
                else:
                    p1[j, i] = (p1Total + alpha) / (N1 + numSubFeatures[i] * alpha) # Equation 8
                    p2[j, i] = ((total - p1Total) + alpha) / (N2 + numSubFeatures[i] * alpha) # Equation 9

        # remove next line and implement from here
        # you are free to use any data structure for paramse
        params = p1, p2, P1, P2
        # do not change the line below
        self.params = params
    
    # you need to implement this function
    def predict(self,Xtest):
        '''
        This function returns the predicted class for a given data set
        
        Inputs:
        Xtest: Testing data set (N x d numpy array)
        
        Output:
        predictions: N length numpy array containing the predictions
        '''
        params = self.params
        a = self.get_a()
        b = self.get_b()
        alpha = self.get_alpha()
        #remove next line and implement from here
        predictions = np.random.choice(self.__classes, np.unique(Xtest.shape[0]))
        for i in range(0, Xtest.shape[0]): # Select customer
            p1 = params[0]
            p2 = params[1]
            p1SumProbability = 1
            p2SumProbability = 1
            for j in range(0, p1.shape[0]): # Go over each feature
                if(j==1):
                    customerSubfeatureSelector = Xtest[i][j]
                else:
                    customerSubfeatureSelector = Xtest[i][j] - 1
                p1SumProbability *= p1[customerSubfeatureSelector][j]
                p2SumProbability *= p2[customerSubfeatureSelector][j]
            p1Probability = (params[2] * p1SumProbability) / (params[2] * p1SumProbability + params[3] * p2SumProbability)
            p2Probability = (params[3] * p2SumProbability) / (params[2] * p1SumProbability + params[3] * p2SumProbability)
            if(p1Probability >= p2Probability):
                predictions[i] = 1
            else:
                predictions[i] = 2




        #do not change the line below
        return predictions
        
def evaluateBias(y_pred,y_sensitive):
    '''
    This function computes the Disparate Impact in the classification predictions (y_pred),
    with respect to a sensitive feature (y_sensitive).
    
    Inputs:
    y_pred: N length numpy array
    y_sensitive: N length numpy array
    
    Output:
    di (disparateimpact): scalar value
    '''
    #remove next line and implement from here
    di = 0
    
    #do not change the line below
    return di

def genBiasedSample(X,y,s,p,nsamples=1000):
    '''
    Oversamples instances belonging to the sensitive feature value (s != 1)
    
    Inputs:
    X - Data
    y - labels
    s - sensitive attribute
    p - probability of sampling unprivileged customer
    nsamples - size of the resulting data set (2*nsamples)
    
    Output:
    X_sample,y_sample,s_sample
    '''
    i1 = y == 1 # good
    i1 = i1[:,np.newaxis]
    i2 = y == 2 # bad
    i2 = i2[:,np.newaxis]
    
    sp = s == 1 #privileged
    sp = sp[:,np.newaxis]
    su = s != 1 #unprivileged
    su = su[:,np.newaxis]

    su1 = np.where(np.all(np.hstack([su,i1]),axis=1))[0]
    su2 = np.where(np.all(np.hstack([su,i2]),axis=1))[0]
    sp1 = np.where(np.all(np.hstack([sp,i1]),axis=1))[0]
    sp2 = np.where(np.all(np.hstack([sp,i2]),axis=1))[0]
    inds = []
    for i in range(nsamples):
        u = stats.bernoulli(p).rvs(1)
        if u == 1:
            #sample one bad instance with s != 1
            inds.append(np.random.choice(su2,1)[0])
            #sample one good instance with s == 1
            inds.append(np.random.choice(sp1,1)[0])
        else:
            #sample one good instance with s != 1
            inds.append(np.random.choice(su1,1)[0])
            #sample one bad instance with s == 1
            inds.append(np.random.choice(sp2,1)[0])
    X_sample = X[inds,:]
    y_sample = y[inds]
    s_sample = s[inds]
    
    return X_sample,y_sample,s_sample,inds
