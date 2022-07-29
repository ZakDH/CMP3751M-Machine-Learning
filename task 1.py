import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Section 1.1#
def getPolynomialDataMatrix(x, degree):
    #assigns 'X' variable with matrix the same shape as 'x' parameter full of 1's
    X = np.ones(x.shape) 
    for i in range(1,degree + 1):#loops through each degree + 1
        #takes sequence of the 1D 'X' matrix and stacks them as a column to make a 2D matrix based on the exponents/power of 'X' and 'i' 
        X = np.column_stack((X, x ** i))
    return X

def pol_regression(x,y,degree):
    #if degree is equal to 0 then a straight line or constant is returned
    if degree == 0:
        return 0
    X = getPolynomialDataMatrix(x, degree)
    #transposes the new 'X' variable based on the dot function of 'X'
    XX = X.transpose().dot(X)
    #sets 'parameter' to solve the linear matrix equation of the transpose of 'X' and the dot function of 'y'
    parameter = np.linalg.solve(XX, X.transpose().dot(y))
    return parameter

#Section 1.3#
def eval_pol_regression(parameters,x,y,degree):
    #splits the data into training and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3)
    Xtrain = getPolynomialDataMatrix(X_train, degree) 
    Xtest = getPolynomialDataMatrix(X_test, degree)
    #calculates the RMSE for the training and testing variables based on the root mean squared error equation
    RMSEtrain[degree - 1] = np.sqrt(np.mean((Xtrain.dot(parameters) - Y_train)**2))
    RMSEtest[degree - 1] = np.sqrt(np.mean((Xtest.dot(parameters) - Y_test)**2))
    return RMSEtrain, RMSEtest

#imports the dataset for section 1
df = pd.read_csv('Task1_data.csv')
#sort the values by 'x' to get a linear regression graph
df.sort_values(by=['x'], inplace=True)
#assign training variables from dataset features
x_train = df['x']
y_train = df['y']
plt.figure()
#plot the training points
plt.plot(x_train,y_train, 'bo')

#Section 1.2#
degrees = [0,1,2,3,6,10]
colours = ['red','purple','cyan','orange','black','yellow']
counter = 0
for i in degrees: #for each degree in the degrees list
    #weights is equal to the result of 'pol_regression function'
    weights = pol_regression(x_train,y_train,i)
    #xtest1 variable instantiated from the 'getPolynomialDataMatrix'
    Xtest1 = getPolynomialDataMatrix(x_train,i)
    #ytest1 is equal to dot function of xtest1
    ytest1 = Xtest1.dot(weights)
    #plot the 'x_train' and new 'ytest1' variables to graph with unique colour
    plt.plot(x_train, ytest1, colours[counter])
    counter+=1
plt.xlim((-5, 5))#limits the x axis to the range of -5 to 5
plt.legend(('training points', '$x^{0}$', '$x^{1}$', '$x^{2}$', '$x^{3}$', '$x^{6}$', '$x^{10}$'), loc = 'lower right')
plt.show()

RMSEtrain = np.zeros((10,1)) #'RMSEtrain' equals a 10x1 array of zeros
RMSEtest = np.zeros((10,1))
for i in range(0,10):
    w = pol_regression(x_train, y_train, i) #'pol_regression' is called again to evaluate the polynomial regression
    #sets RMSEtrain and test to the product of 'eval_pol_regression' function
    RMSEtrain, RMSEtest = eval_pol_regression(w, x_train, y_train, i)
plt.figure()
plt.semilogy(range(0,10), RMSEtrain)
plt.semilogy(range(0,10), RMSEtest)
plt.legend(('RMSE on training set', 'RMSE on test set'))
plt.show()