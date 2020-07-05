import numpy as np 
import matplotlib.pyplot as plt 
import random 

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


#extract dataframes corresponding to countries
def countryDF(country, dataframe):
    countryData = dataframe[dataframe['Country'] == country]
    return countryData


#extract columns
def columnExtractor(dataframe, columnName):
    values = dataframe[columnName]
    return values 


#extract variable number of columns 
def variableColumnExtractor(dataframe, headers):
    if len(headers) > 0 and type(headers) == list:
        featureDF = dataframe[headers]
        return featureDF 


# plot values for dataset columns    
def plotQuantities(qty1, qty2, xlabel, ylabel, label, title):
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.scatter(qty1, qty2, label=label)
    plt.title(title)
    plt.legend()
    plt.show()


# Plot the function, the prediction and the 95% confidence interval based on the MSE
def plotFinal(years, countryQuantity, yearsTrain, countryQuantityTrain, yearsTest, countryQuantityTest, yearsPredict, countryQuantityPredict, ylabel, sigma, regression_type):
    plt.figure()
    
    #actual data 
    plt.scatter(years, countryQuantity,label='Observations')

    #estimate
    plt.plot(yearsPredict, countryQuantityPredict, 'r--', label='Prediction')
    plt.scatter(yearsTest,countryQuantityTest,label='Missing values')

    #plt.fill(np.concatenate([x, x[::-1]]),
    #         np.concatenate([y_pred - 1.9600 * sigma,
    #                        (y_pred + 1.9600 * sigma)[::-1]])[:,6],
    #         alpha=1, fc='b', ec='None', label='95% confidence interval')
    
    if regression_type == 'Gaussian': 
        plt.fill_between(yearsPredict.flat, (countryQuantityPredict.flat-2*sigma), (countryQuantityPredict.flat+2*sigma), 
                     color='green',alpha=0.5,label='95% confidence interval')
    plt.ylabel(ylabel)
    plt.xlabel('Year')
    plt.legend()


def errorComputation(countryDF, countryQuantityPredict, quantity,regression_type):
    #countryQuantityPredict = countryQuantityPredict[::-1]
    countryQuantityActual = columnExtractor(countryDF,str(quantity)).tolist()
    countryQuantityActual = countryQuantityActual[::-1]
    print(regression_type,'Prediction \n', countryQuantityPredict.T)
    print('Actual \n', countryQuantityActual)
    error = (np.absolute((countryQuantityPredict.T - countryQuantityActual))/countryQuantityActual)*100
    return error


#def errorRMSE(countryDF, countryQuantityPredict, quantity,regression_type):
    

#error calculation between predicted value and ground truth
def errorPlot(qty1, error, xlabel, ylabel,regression_type,color):
    
    plt.plot(qty1[::-1], np.ones((len(qty1),1))*np.mean(error.T), '--', c=color, label=regression_type+' mean')
    print('mean absolute percentage error',regression_type,': ',np.mean(error.T))
    plt.plot(qty1[::-1],error.T, '-', c=color, label=regression_type)
    #plt.plot(qty1[::-1],np.zeros((len(qty1),1)),'k--')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()


#matrix randomizer
def randomizer(countryQuantity, years, split):
    countryQuantity = countryQuantity.tolist()
    years = years.tolist()
     
    print('Train:Test split is: ',split,':',16-split)

    #combine both the lists and randomize while maintaining the mapping 
    combinedZip = list(zip(years,countryQuantity))
    random.shuffle(combinedZip) 

    #unzip the combination 
    years,countryQuantity = zip(*combinedZip) 

    countryQuantity = list(countryQuantity)
    countryQuantityTrain = countryQuantity[:split]

    years = list(years)
    yearsTrain = years[:split]

    countryQuantityTest = countryQuantity[split:]
    yearsTest = years[split:]

    #countryQuantity = countryQuantityTrain 
    #years = yearsTrain

    countryQuantityTrain = np.asarray(countryQuantityTrain).reshape(-1,1)
    yearsTrain = np.asarray(yearsTrain).reshape(-1,1)

    countryQuantityTest = np.asarray(countryQuantityTest).reshape(-1,1)
    yearsTest = np.asarray(yearsTest).reshape(-1,1)

    return countryQuantityTrain, yearsTrain, countryQuantityTest, yearsTest


#################GAUSSIAN REGRESSION#################
# convention followed in relation to scikit documentation 

#def linearRegression(xtrain, ytrain, xtest, ytest, x, y):
def gaussianRegression(xtrain, ytrain, xtest, ytest, x, y):
    # Instantiate a Gaussian Process model
    lengthScale = np.random.randint(50)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(lengthScale, (1e-2, 1e2))
    print('length scale is: ',lengthScale)
    #print(kernel)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    
    # Mesh the input space for evaluations of the real function, the prediction and its MSE
    #x = np.atleast_2d(np.linspace(0, 10, 1000)).T
    yearsPredict = np.array(np.linspace(2000, 2015, 16)).reshape(-1,1)

    years = x 
    countryQuantity = y 
    
    yearsTrain = xtrain 
    countryQuantityTrain = ytrain
    
    yearsTest = xtest
    countryQuantityTest = ytest 
    
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(yearsTrain, countryQuantityTrain)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    countryQuantityPredict, sigma = gp.predict(yearsPredict, return_std=True)

    #change here 
    plotFinal(years, countryQuantity, yearsTrain, countryQuantityTrain, yearsTest, countryQuantityTest, yearsPredict, countryQuantityPredict, 'Life Expectancy', sigma, regression_type = 'Gaussian')
    return countryQuantityPredict, sigma


#################LINEAR REGRESSION#################
# convention followed in relation to scikit documentation 

def linearRegression(xtrain, ytrain, xtest, ytest, x, y):
    years = x 
    countryQuantity = y 
    
    yearsTrain = xtrain 
    countryQuantityTrain = ytrain
    
    yearsTest = xtest
    countryQuantityTest = ytest    
    
    yearsPredict = np.array(np.linspace(2000, 2015, 16)).reshape(-1,1)

    # # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(yearsTrain, countryQuantityTrain)

    # Make predictions using the testing set
    countryQuantityPredictLR = regr.predict(yearsPredict)

    # Plot outputs
    plotFinal(years, countryQuantity, yearsTrain, countryQuantityTrain, yearsTest, countryQuantityTest, yearsPredict, countryQuantityPredictLR, 'Life Expectancy',0,regression_type='Linear')
    return countryQuantityPredictLR

