import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

if __name__ == "__main__":
    # read csv file
    csvFile = pd.read_csv("data.csv")
    
    # get column wise values from the csv file
    pm2_5 = csvFile["PM2.5"]
    no = csvFile["NO"]  
    no2 = csvFile["NO2"]
    nox = csvFile["NOx"]
    nh3 = csvFile["NH3"]
    co = csvFile["CO"]
    so2 = csvFile["SO2"]
    o3 = csvFile["O3"]
    benzene = csvFile["Benzene"]
    toluene = csvFile["Toluene"]
    aqi = csvFile["AQI"]

    # hexbin_plot(pm2_5, aqi, "PM2.5", "AQI")
    # hexbin_plot(no, aqi, "NO", "AQI")
    # hexbin_plot(no2, aqi, "NO2", "AQI")
    # hexbin_plot(nox, aqi, "NOx", "AQI")
    # hexbin_plot(nh3, aqi, "NH3", "AQI")
    # hexbin_plot(co, aqi, "CO", "AQI")
    # hexbin_plot(so2, aqi, "SO2", "AQI")
    # hexbin_plot(o3, aqi, "O3", "AQI")
    # hexbin_plot(benzene, aqi, "Benzene", "AQI")
    # hexbin_plot(toluene, aqi, "Toluene", "AQI")
