from Dataprepare import*
from Logistic import*
from SVM import*
from NN import*

# cross validation
from sklearn.model_selection import cross_val_score
# Compute three score vectores
scores1 = cross_val_score(logreg, X_train, Y_train, cv=5)
scores2 = cross_val_score(lin_clf, X_train, Y_train, cv=5)
scores3 = cross_val_score(mlp, X_train, Y_train, cv=5)

# Compute two mean scores and print out 
print("Logistic Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))
print("SVM Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))
print("MLP Accuracy: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std() * 2))
