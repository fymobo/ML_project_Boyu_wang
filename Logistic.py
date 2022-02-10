from Dataprepare import*
#logistics regression
#Create an instance of Logistic Regression Classifier
logreg = LogisticRegression(C=1, solver="sag")
#Split dateset into random train and test subsets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.001, random_state=0)
#Fit the data.
logreg.fit(X_train, Y_train)
#Print out the accuracy of the train and test subsets
print('Logistic result: ',logreg.score(X_train,Y_train))
print('Logistic testsets result: ',logreg.score(X_test,Y_test))


# predict over training data 
# test-image set pre-prepare
X_testSet=[]
Y_testSet=[]
for i in range(0,len(testLabels)):
    X_testSet.append(testImages[i].flatten())
    Y_testSet.append(int(testLabels[i]))
X_testSet=np.array(X_testSet)
Y_testSet=np.array(Y_testSet)
Ypred=logreg.predict(X_testSet)

# print classification result
plt.figure(figsize=(15,70))
for i in range(0,len(testLabels)):
    plt.subplot(20,5,i+1)
    plt.title('class '+str(Ypred[i]))
    plt.imshow(testImages[i])
plt.show()

# check the accuracy
print('Logistic final Classification Accuracy:')
accuracy_score(Y_testSet,Ypred)
