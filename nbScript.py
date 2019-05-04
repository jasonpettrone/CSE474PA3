import pickle
from sklearn.model_selection import cross_val_predict,cross_val_score
from nbFunctions import *
# load data
data = pickle.load(open('german.data.pickle','rb'))
# measure performance in cross validation mode
model = NBC(a=1,b=1,alpha=1)
print('Average F-Measure = %.2f'%np.mean(cross_val_score(model,data['features'], data['labels'].flatten(),cv=10,scoring='f1_macro')))
# measure disparate impact
si = 0
y_pred = cross_val_predict(model,data['features'], data['labels'].flatten(),cv=10)
print('Disparate Impact Score for Sensitive Feature %d = %.2f'%(si,evaluateBias(y_pred,data['sensitive'][:,si])))

# artifically induce bias in training data
p = 0.5
X_sample,y_sample,s_sample,inds = genBiasedSample(data['features'], data['labels'].flatten(),data['sensitive'][:,si],p)
model = NBC(a=1,b=1,alpha=1)
y_pred = cross_val_predict(model,X_sample,y_sample,cv=10)
print('Disparate Impact Score for Sensitive Feature %d = %.2f'%(si,evaluateBias(y_pred,s_sample)))
