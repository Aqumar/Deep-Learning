import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values

y = dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X_1 = LabelEncoder()

X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()

X[:,2] = labelencoder_X_1.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features=[1])

X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]

from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

from sklearn.preprocessing import  StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)

import keras

from keras.models import Sequential  ##To initialize ANN
from keras.layers import Dense  ###Creating layers in Neural Network



##Evaulating the ANN
##Create a Keras wrapper to include cross validation from scikit-learn.

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential() ##Initialize ANN
    #init initializes the weight in uniform distribution close to zero
    #output dim is choosen as average of number of inputs and output
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu',input_dim=11))#Adding first input layer and hidden layer
    
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))#Adding second hidden layer
    
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))#Adding output layer

    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)



mean_accuracies = mean(accuracies)
variance = std(accuracies)




