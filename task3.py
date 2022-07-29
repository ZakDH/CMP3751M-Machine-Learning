import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

#Section 3.1#
#imports dataset for section 3
df = pd.read_csv("Task3 - dataset - HIV RVG.csv")
#displays statistical summary of dataframe excluding the percentiles
print(df.describe(percentiles=[]))

#plots a boxplot using seaborn against the alpha values
sns.boxplot(x="Participant Condition", y="Alpha",data=df)
plt.show()

#plots a density plot against the beta values
betadensity=df.pivot(columns='Participant Condition',values='Beta')
betadensity.plot.density(linewidth=2)
plt.show()

#Section 3.2#
#shuffles the dataset
df_shuffle = df.sample(frac=1).reset_index(drop=True)
#assigns the 'control' and 'patient' values to 0 and 1
d = {'Control': 0, 'Patient': 1}
#maps the new control and patient values to the 'df_shuffle variable
df_shuffle['Participant Condition'] = df_shuffle['Participant Condition'].map(d)
x_data = (df_shuffle[df.columns[:-1]])#assigns 'x_data' variable with all the columns except the last one
y_data = df_shuffle['Participant Condition']#assigns 'y_data' with the label column for the dataset
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=0) #90% training and 10% test

# scaler = StandardScaler() 
# X_train_transform = scaler.fit_transform(X_train) #transform
# X_test_transform = scaler.transform(X_test) #transform

#configures a neural network with 2 hidden layers with 500 neurons each, a logistic activation function and 500 iterations
mlp = MLPClassifier(hidden_layer_sizes=(500,2),activation='logistic',max_iter=500)
#training data is fitted into the neural network
mlp.fit(X_train, y_train)
#predictions are made based on the 'X_test' variable values
prediction = mlp.predict(X_test)
#mean accuracy of the model is displayed with 2dp formatting
print("Mean accuracy of the ANN: %.2f" % accuracy_score(y_test,prediction))

#random forest classfier is configured with 1000 trees and a minimum of 5 leaves per tree
rfc = RandomForestClassifier(n_estimators = 1000,min_samples_leaf=5) 
rfc.fit(X_train, y_train)
prediction = rfc.predict(X_test)
print("Mean accuracy of the RFC: %.2f" % accuracy_score(y_test,prediction))

#random forest classfier is configured with 1000 trees and a minimum of 10 leaves per tree
rfc = RandomForestClassifier(n_estimators = 1000,min_samples_leaf=10) 
rfc.fit(X_train, y_train)
prediction = rfc.predict(X_test)
print("Mean accuracy of the RFC: %.2f" % accuracy_score(y_test,prediction))

#Section 3.3#
#function for a neural network with different neurons being passed through
def cv_neural_network(x_train, y_train, neurons):
    #neural network configured based on number of neurons between 2 layers
    mlp = MLPClassifier(hidden_layer_sizes=(neurons,2),activation='logistic')
    #score is computed using 10-fold cv 
    scores = cross_val_score(estimator = mlp, X = x_train, y = y_train, cv = 10)
    return scores 

def cv_random_forest(x_train, y_train, trees):
    #random forest is configured based on the different number of trees and a minimum of 10 leaves
    rfc = RandomForestClassifier(n_estimators = trees, min_samples_leaf=10) 
    #score is computed using 10-fold cv 
    scores = cross_val_score(estimator = rfc, X = x_train, y = y_train, cv = 10)
    return scores

neurons_list = [50,500,1000]
trees_list = [50,500,10000]

#loops through neurons list and each element in the list is passed to the neural network function
for neurons in neurons_list:
    scores = cv_neural_network(X_train,y_train,neurons)
    #collects score for each fold
    print("CV accuracy scores: %s " % scores)
    #computes the mean score for all the folds and formats it to 2dp
    print("CV accuracy scores for " +str(neurons)+ " neuron neural network : %.2f" % (np.mean(scores)))

#loops through trees list and each element in the list is passed to the random forest function
for trees in trees_list:
    scores = cv_random_forest(X_train,y_train,trees)
    #collects score for each fold
    print("CV accuracy scores: %s " % scores)
    #computes the mean score for all the folds and formats it to 2dp
    print("CV accuracy scores for " +str(trees)+ " tree random forest classifier : %.2f" % (np.mean(scores)))