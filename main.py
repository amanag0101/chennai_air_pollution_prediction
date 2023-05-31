import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# visualize correlation between pollutant and aqi using scatter plot
def scatter_plot(x, y, x_label, y_label):
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

# hexbin plot
def hexbin_plot(x, y, x_label, y_label):
    plt.hexbin(x, y, gridsize=30, cmap="Blues")
    plt.colorbar(label="Count")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

# 2D Density Plot
def density_plot_2d(x, y):
    # remove nan values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    
    plt.hist2d(x, y, bins=30, cmap='Blues')
    plt.colorbar(label='Count')
    plt.show()

# correlation values between pollutant and aqi 
def correlation(x, y):
    return x.corr(y)

# calculate missing values
def impute_missing_values(X, strategy='mean'):
    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(X)
    X_imputed = imputer.transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
    return X_imputed

# encode the labels
def encode_labels(y):
    encoder = LabelEncoder()
    encoder.fit(y)
    y_encoded = encoder.transform(y)
    return y_encoded

# train the model
def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def test_linear_regression(X_test, y_test, model):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def predict_linear_regression(X, model):
    y_pred = model.predict(X)
    return y_pred   

if __name__ == "__main__":
    # read csv file
    csvFile = pd.read_csv("data.csv")
    
    # get column wise values from the csv file
    pm2_5 = csvFile["PM2.5"]
    # no = csvFile["NO"]  
    no2 = csvFile["NO2"]
    # nox = csvFile["NOx"]
    nh3 = csvFile["NH3"]
    # co = csvFile["CO"]
    so2 = csvFile["SO2"]
    o3 = csvFile["O3"]
    # benzene = csvFile["Benzene"]
    # toluene = csvFile["Toluene"]
    aqi = csvFile["AQI"]
    aqi_bucket = csvFile["AQI_Bucket"]
    aqi_bucket_class = encode_labels(aqi_bucket)

    # create a dataframe with independent variables
    X = pd.DataFrame({'PM2.5': pm2_5, 'NO2': no2, 'NH3': nh3, 'SO2': so2, 'O3': o3})
    # X = pd.DataFrame({'PM2.5': pm2_5})

    # impute missing values in X
    X_imputed = impute_missing_values(X)

    # create a series with dependent variable
    y = aqi

    y_dropped = y.dropna()
    X_dropped = X_imputed.loc[y_dropped.index]

    # get the coefficients
    model = train_linear_regression(X_dropped, y_dropped)
    print(model.coef_)

    # calculate the mean squared error on the test data
    mse = test_linear_regression(X_dropped, y_dropped, model)
    print(mse)

    # make predictions on the test data
    y_pred = predict_linear_regression(X_dropped, model)
    y_pred = pd.DataFrame({'Y_Pred': y_pred})

    data_dropped = pd.concat([X_dropped, y_dropped, y_pred], axis=1)
    data_dropped.to_csv("predicted.csv", index=False)