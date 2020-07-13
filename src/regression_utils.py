import numpy as np 
import matplotlib.pyplot as plt 
import random 

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


# extract dataframes corresponding to countries
def countryDF(country, dataframe):
    countryData = dataframe[dataframe['Country'] == country]
    return countryData


# extract single column
def columnExtractor(dataframe, columnName):
    values = dataframe[columnName]
    return values 


# extract variable number of columns 
def variableColumnExtractor(dataframe, headers):
    if len(headers) > 0 and type(headers) == list:
        featureDF = dataframe[headers]
        return featureDF 


# plot values for dataset columns    
def plotQuantities(qty1, qty2, xlabel, ylabel, label, title):
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.scatter(qty1, qty2, label=label)
    #plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


# Plot the function, the prediction and the 95% confidence interval based on the MSE
def plotFinal(countryQuantity2, countryQuantity, countryQuantity2Train, countryQuantityTrain, countryQuantity2Test, countryQuantityTest, countryQuantity2Predict, countryQuantityPredict, xlabel, ylabel, sigma, regression_type):
    plt.figure()
    
    #actual data 
    plt.scatter(countryQuantity2, countryQuantity,label='Observations')
    plt.grid()
    #estimate
    plt.plot(countryQuantity2Predict, countryQuantityPredict, 'r--', label='Prediction')
    plt.scatter(countryQuantity2Test,countryQuantityTest,label='Missing values')

    #plt.fill(np.concatenate([x, x[::-1]]),
    #         np.concatenate([y_pred - 1.9600 * sigma,
    #                        (y_pred + 1.9600 * sigma)[::-1]])[:,6],
    #         alpha=1, fc='b', ec='None', label='95% confidence interval')
    
    if regression_type == 'Gaussian': 
        plt.fill_between(countryQuantity2Predict.flat, (countryQuantityPredict.flat-2*sigma), (countryQuantityPredict.flat+2*sigma), 
                     color='green',alpha=0.5,label='95% confidence interval')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()


# error calculation between predicted value and ground truth
def errorComputation(countryDF, countryQuantityPredict, quantity,regression_type):
    #countryQuantityPredict = countryQuantityPredict[::-1]
    countryQuantityActual = columnExtractor(countryDF,str(quantity)).tolist()
    countryQuantityActual = countryQuantityActual[::-1]
    print(regression_type,'Prediction \n', countryQuantityPredict.T)
    print('Actual \n', countryQuantityActual)
    error = (np.absolute((countryQuantityPredict.T - countryQuantityActual))/countryQuantityActual)*100
    return error


#def errorRMSE(countryDF, countryQuantityPredict, quantity,regression_type):
    

# plotting error values as computed by errorComputation() 
def errorPlot(qty1, error, xlabel, ylabel,regression_type,color):
    
    plt.plot(qty1[::-1], np.ones((len(qty1),1))*np.mean(error.T), '--', c=color, label=regression_type+' mean')
    print('mean absolute percentage error',regression_type,': ',np.mean(error.T))
    plt.plot(qty1[::-1],error.T, '-', c=color, label=regression_type)
    #plt.plot(qty1[::-1],np.zeros((len(qty1),1)),'k--')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()


# matrix randomizer
def randomizer(countryQuantity, countryQuantity2, split):
    countryQuantity = countryQuantity.tolist()
    countryQuantity2 = countryQuantity2.tolist()
     
    print('Train:Test split is: ',split,':',16-split)

    #combine both the lists and randomize while maintaining the mapping 
    combinedZip = list(zip(countryQuantity2,countryQuantity))
    random.shuffle(combinedZip) 

    #unzip the combination 
    countryQuantity2,countryQuantity = zip(*combinedZip) 

    countryQuantity = list(countryQuantity)
    countryQuantityTrain = countryQuantity[:split]

    countryQuantity2 = list(countryQuantity2)
    countryQuantity2Train = countryQuantity2[:split]

    countryQuantityTest = countryQuantity[split:]
    countryQuantity2Test = countryQuantity2[split:]

    #countryQuantity = countryQuantityTrain 
    #countryQuantity2 = countryQuantity2Train

    countryQuantityTrain = np.asarray(countryQuantityTrain).reshape(-1,1)
    countryQuantity2Train = np.asarray(countryQuantity2Train).reshape(-1,1)

    countryQuantityTest = np.asarray(countryQuantityTest).reshape(-1,1)
    countryQuantity2Test = np.asarray(countryQuantity2Test).reshape(-1,1)

    return countryQuantityTrain, countryQuantity2Train, countryQuantityTest, countryQuantity2Test


#################GAUSSIAN REGRESSION#################
# convention followed in relation to scikit documentation 

def gaussianRegression(xtrain, ytrain, xtest, ytest, x, y, countryQuantityName, countryQuantity2Name):
    # Instantiate a Gaussian Process model
    lengthScale = np.random.randint(50)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(lengthScale, (1e-2, 1e2))
    print('length scale is: ',lengthScale)
    #print(kernel)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    
    # Mesh the input space for evaluations of the real function, the prediction and its MSE
    #x = np.atleast_2d(np.linspace(0, 10, 1000)).T
    countryQuantity2Predict = np.array(np.linspace(2000, 2015, 16)).reshape(-1,1)

    countryQuantity2 = x 
    countryQuantity = y 
    
    countryQuantity2Train = xtrain 
    countryQuantityTrain = ytrain
    
    countryQuantity2Test = xtest
    countryQuantityTest = ytest 
    
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(countryQuantity2Train, countryQuantityTrain)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    countryQuantityPredict, sigma = gp.predict(countryQuantity2Predict, return_std=True)

    #change here 
    plotFinal(countryQuantity2, countryQuantity, countryQuantity2Train, countryQuantityTrain, countryQuantity2Test, countryQuantityTest, countryQuantity2Predict, countryQuantityPredict, countryQuantity2Name, countryQuantityName, sigma, regression_type = 'Gaussian')
    return countryQuantityPredict, sigma


#################LINEAR REGRESSION#################
# convention followed in relation to scikit documentation 

def linearRegression(xtrain, ytrain, xtest, ytest, x, y, countryQuantityName, countryQuantity2Name):
    countryQuantity2 = x 
    countryQuantity = y 
    
    countryQuantity2Train = xtrain 
    countryQuantityTrain = ytrain
    
    countryQuantity2Test = xtest
    countryQuantityTest = ytest    
    
    countryQuantity2Predict = np.array(np.linspace(2000, 2015, 16)).reshape(-1,1)

    # # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(countryQuantity2Train, countryQuantityTrain)

    # Make predictions using the testing set
    countryQuantityPredictLR = regr.predict(countryQuantity2Predict)

    # Plot outputs
    plotFinal(countryQuantity2, countryQuantity, countryQuantity2Train, countryQuantityTrain, countryQuantity2Test, countryQuantityTest, countryQuantity2Predict, countryQuantityPredictLR, countryQuantity2Name, countryQuantityName ,0,regression_type='Linear')
    return countryQuantityPredictLR

