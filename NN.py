from Dataprepare import*

from sklearn.neural_network import MLPClassifier

# train a MLP
mlp = MLPClassifier(hidden_layer_sizes=(50, ), max_iter=10, alpha=1e-4, 
                    solver='sgd', verbose=10, random_state=1, learning_rate_init=.1)


X = X/500.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.001, random_state=0)

mlp.fit(X_train, Y_train)


print(f"MLP Training set score: {mlp.score(X_train, Y_train):.3f}")
print(f"MLP Test set score: {mlp.score(X_test, Y_test):.3f}")

# predict over training data 
# test-image set pre-prepare
X_testSet=[]
Y_testSet=[]
for i in range(0,len(testLabels)):
    X_testSet.append(testImages[i].flatten())
    Y_testSet.append(int(testLabels[i]))
X_testSet=np.array(X_testSet)
Y_testSet=np.array(Y_testSet)
Ypred=mlp.predict(X_testSet)

# print classification result
plt.figure(figsize=(15,70))
for i in range(0,len(testLabels)):
    plt.subplot(20,5,i+1)
    plt.title('class '+str(Ypred[i]))
    plt.imshow(testImages[i])
plt.show()

# check the accuracy
print('MLP final Classification Accuracy:')

accuracy_score(Y_testSet,Ypred)


# check the accuracy
accuracy_score(Y_testSet,Ypred)
