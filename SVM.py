from Dataprepare import*
#train a SVM
lin_clf = svm.LinearSVC(C=1)
#Split dateset into random train and test subsets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.001, random_state=0)

lin_clf.fit(X_train, Y_train)
#Print out the accuracy of the train and test subsets
print('SVM result: ',lin_clf.score(X_train,Y_train))
print('SVM testsets result: ',lin_clf.score(X_test,Y_test))

# predict over training data 
# test-image set pre-prepare
X_testSet=[]
Y_testSet=[]
for i in range(0,len(testLabels)):
    X_testSet.append(testImages[i].flatten())
    Y_testSet.append(int(testLabels[i]))
X_testSet=np.array(X_testSet)
Y_testSet=np.array(Y_testSet)
lin_clf_Ypred=lin_clf.predict(X_testSet)

# print classification result
plt.figure(figsize=(15,70))
for i in range(0,len(testLabels)):
    plt.subplot(20,5,i+1)
    plt.title('class '+str(lin_clf_Ypred[i]))
    plt.imshow(testImages[i])
plt.show()

# check the accuracy
print('SVM final Classification Accuracy:')
accuracy_score(Y_testSet,lin_clf_Ypred)

