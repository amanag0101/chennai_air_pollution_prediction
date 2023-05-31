import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


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

    plt.hist2d(x, y, bins=30, cmap="Blues")
    plt.colorbar(label="Count")
    plt.show()


# correlation values between pollutant and aqi
def correlation(x, y):
    return x.corr(y)


# calculate missing values
def impute_missing_values(X, strategy="mean"):
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


def get_class_output(y, y_pred):
    y_class = []
    y_pred_class = []

    for i in y:
        y_class.append(get_class(i))

    for i in range(len(y_pred)):
        y_pred_class.append(get_class(i))

    return pd.DataFrame({"y_class": y_class}), pd.DataFrame(
        {"y_pred_class": y_pred_class}
    )


def get_class(value):
    if value >= 0 and value <= 50:
        return "Good"
    elif value >= 51 and value <= 100:
        return "Satisfactory"
    elif value >= 101 and value <= 200:
        return "Moderate"
    elif value >= 201 and value <= 300:
        return "Fair"
    elif value >= 301 and value <= 400:
        return "Poor"
    else:
        return "Very Poor"


if __name__ == "__main__":
    # read csv file
    csvFile = pd.read_csv("data.csv")

    # get column wise values from the csv file
    pm2_5 = csvFile["PM2.5"]
    no2 = csvFile["NO2"]
    # nh3 = csvFile["NH3"]
    # so2 = csvFile["SO2"]
    o3 = csvFile["O3"]
    aqi = csvFile["AQI"]
    aqi_bucket = csvFile["AQI_Bucket"]
    aqi_bucket_class = encode_labels(aqi_bucket)

    # create a dataframe with independent variables
    X = pd.DataFrame({"PM2.5": pm2_5, "NO2": no2, "NH3": nh3, "SO2": so2, "O3": o3})

    # impute missing values in X
    X_imputed = impute_missing_values(X)
    # create a series with dependent variable
    y = aqi

    # drop the empty data rows
    y_dropped = y.dropna()
    X_dropped = X_imputed.loc[y_dropped.index]

    # split the dataset into train set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_dropped, y_dropped, test_size=0.30
    )

    # get the coefficients
    model = train_linear_regression(X_train, y_train)
    print(model.coef_)

    # calculate the mean squared error on the test data
    mse = test_linear_regression(X_test, y_test, model)
    print(mse)

    # make predictions on the test data
    y_pred = predict_linear_regression(X_test, model)
    y_pred = pd.DataFrame({"Y_Pred": y_pred})

    # y_class, y_pred_class = get_class_output(y_test, y_pred)

    save_data = pd.concat(
        [
            X_test.reset_index(drop=True),
            y_test.reset_index(drop=True),
            y_pred,
            # y_class,
            # y_pred_class,
        ],
        axis=1,
    )
    save_data.to_csv("predicted.csv", index=False)
