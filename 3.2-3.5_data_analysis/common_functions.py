# Imports

import time
from typing import Union, Any, get_origin, get_args

## Math

import math
import numpy as np
from numpy import ndarray

## Data

import csv
import pandas as pd
from pandas.core.frame import DataFrame

pd.set_option("display.max_colwidth",100)
pd.set_option("display.max_rows",100)

## Plots

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf 

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.style.use('ggplot')

plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

plt.rcParams['text.color'] = '#000000'
plt.rcParams['axes.labelcolor'] = '#000000'
plt.rcParams['xtick.color'] = '#000000'
plt.rcParams['ytick.color'] = '#000000'

a, b, c = 18, 20, 24
plt.rcParams['font.size'] = b            # sets the default font size
plt.rcParams['axes.labelsize'] = c       # for x and y labels 
plt.rcParams['axes.titlesize'] = c       # for subplot titles
plt.rcParams['xtick.labelsize'] = a      # for x-axis tick labels
plt.rcParams['ytick.labelsize'] = a      # for y-axis tick labels
plt.rcParams['legend.fontsize'] = b      # for legend text

plt.rcParams['axes.labelpad'] = 15

## Models

from fitter import Fitter

from scipy import fft
from scipy.optimize import curve_fit, minimize
from scipy.stats import kstest, norm, laplace, entropy, cauchy, entropy
from sklearn.feature_selection import mutual_info_regression

# Common functions

## Error handling

def assert_type(obj: Any, expected: Union[type, list[type]]) -> None:
    # Type checking function. Raises error if False.
    # Parameters: 
    #     obj: object to type-check,
    #     expected: expected type(s) (type or list[type])
    
    assert generic_isinstance(obj, expected), f"Object is not of the expected type {expected}."

def generic_isinstance(obj: Any, expected: Union[type, list[type]]) -> bool:    
    # Generic inclusive wrapper for isinstance (recursive).
    # Parameters: 
    #     obj: object to type-check,
    #     expected: expected type(s) (type or list[type])
   
    # Any or None
    if expected is Any or obj is None: 
        return True
    
    # List of types
    if isinstance(expected, list):
        return any(generic_isinstance(obj, t) for t in expected) #Check if any type in list returns True
    
    origin = get_origin(expected)
    args = get_args(expected)
    
    # Single un-parametrized type
    if origin is None:
        return isinstance(obj, expected)
    
    # Single Parametrized type (union, tuple, list)
    ## Union
    if origin is Union: 
        return any(generic_isinstance(obj, t) for t in args)  #Check if any type in Union returns True
    ## Tuple
    if origin is tuple:
        if not isinstance(obj, tuple) or len(obj) != len(args): #Check if obj is tuple and correct length
            return False
        return all(generic_isinstance(obj[i], args[i]) for i in range(len(args))) #Check if all items in obj are correct type
    ## List
    if origin is list: 
        if not isinstance(obj, list) or len(args) != 1: #Check if obj is tuple and correct length
            return False
        return all(generic_isinstance(item, args[0]) for item in obj) #Check if all items in obj are correct type
    
    assert False, f"Type {expected} not handled."

## Printing
    
def printif(verbose: bool, string: str) -> None:
    # Conditional printing.
    # Parameters: 
    #     verbose: condition for printing (boolean),
    #     string: item to print (string)
    
    if verbose:
        print(string)
    
def tup_string(selection: list[tuple[str, Any]]) -> str:
    # Returns a formatted string for a list of 2-element tuples (e.g. [('a',2),('b',1)] as "a = 2, b = 1").
    # Parameters: 
    #     selection: item to format (list[tuple[str,Any]])
    # Returns: 
    #     tup_str: formatted output (string)
    
    assert_type(selection, list[tuple[str,Any]])
    
    tup_str = ""
    for tup in selection:
        if not tup_str == "":
            tup_str += ", "
        tup_str += str(tup[0]) + " = " + str(tup[1])
        
    return tup_str

def get_stats(data: DataFrame, col: str) -> None:
    # Formatted print of the stats of a column from a dataframe.
    # Parameters: 
    #     data: data source (pandas dataframe), 
    #     col: column name from data to apply stats to (string)
    
    assert col in data.columns, f"Column id {col} not in dataframe."
    
    print("    Stats for " + col + " :")
    stats = data[col].value_counts()
    stats_perc = data[col].value_counts(normalize = True)
    
    total = 0
    for i in range(len(stats)):
        total = stats[i]
    
    for i in range(len(stats)):
        print("        " + str(stats.index[i]) + "\t\t", str(stats[stats.index[i]]) + 
              '\t\t', str(round(stats_perc[stats.index[i]] * 100)) + '%')
    print()
    
## Time-Series Operations

def diff(df: DataFrame, columns: list[str]) -> None:
    # Apply time series differencing in place to multiple columns of a dataframe.
    # Parameters:
    #     df: data source (pandas dataframe), 
    #     columns: list of column names containing time series, in time order (list of strings)
    
    for col in columns:
        y = df[col].to_numpy()
        y =  y - np.roll(y.copy(), 1)
        y[0] = 0
        df.loc[:, col] = y

def remove_extremes(x: ndarray) -> ndarray: 
    # Remove values outside of the boxplot [Q1 - 1.5*IRQ, Q3 + 1.5*IRQ].
    # Parameters:
    #     x: time series (numpy array)
    # Returns:
    #     y: resulting time series (numpy array)
    
    Q1, Q3 = np.quantile(x, [0.25, 0.75])
    IQR = Q3 - Q1
    a, b = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    
    y = x[(a <= x) & (x <= b)]
    return y

def fft_autocorr(x: ndarray) -> ndarray:
    # Return the autocorrelation coefficients of a time series using the Fast Fourier Transform.
    # Parameters:
    #     x: time series, in time order (numpy array)
    # Returns:
    #     result: discrete autocorrelation function of x (numpy array)
    
    x = np.asarray(x)
    N = len(x)
    s = fft.fft(x, N*2-1)
    
    coeffs = np.real(fft.ifft(s * np.conjugate(s), N*2-1))[:N]
    coeffs /= coeffs[0]
    result = np.array(coeffs.astype(float))
    return result

def data_freq(filename: str, column: str, diff_order: int = 0, selection: list[tuple[str, Any]] = [],
              crop: tuple[float, float] = (None, None)) -> DataFrame:
    # Outputs the frequencies of unique values of a series from a csv data file.
    # Parameters: 
    #     filename: name of the csv data file (str), 
    #     column: name of the column to plot (str),
    #     diff_order: order of differencing for time series (int in [0,2]),
    #     selection: to select rows by value (list[tuple[str, Any]], e.g. '[('idle', True)]'),
    #     crop: to cull head or tail of data (tuple[float, float])
    
    # Import
    data_path = path + "\\csv\\" + filename + ".csv"
    data = pd.read_csv(data_path)
    
    # Selection
    assert_type(selection, list[tuple[str,Any]])
    for tup in selection: 
        data = data[data[tup[0]] == tup[1]] 

    # Cropping
    assert_type(crop, tuple[float,float])
    n = data.shape[0]
    if crop[0] != None and int(crop[0]*n) > 0: 
        data = data.tail(-int( n*crop[0] ))
    if crop[1] != None and int(crop[1]*n) > 0: 
        data = data.head(-int( n*crop[1] ))
        
    # Differencing
    assert isinstance(diff_order, int) and diff_order >= 0, f"Order of differencing must be positive."
    for i in range(diff_order):
        diff(data, [column])
    
    # Frequencies
    value_counts = data[column].value_counts()
    value_counts_df = value_counts.reset_index()
    value_counts_df.columns = ['Unique Value', 'Count']
    value_counts_df['Ratio'] = value_counts_df['Count'] / data.shape[0]
    value_counts_df['Percentage'] = value_counts_df['Ratio'] * 100
    return value_counts_df

def conditional_data_freq(filename: str, column: str, diff_order: int = 0, selection: list[tuple[str, Any]] = [],
              crop: tuple[float, float] = (None, None), output: str = None):
    # Outputs the conditional frequencies of the values of a time series given the previous value. 
    # Reads from a csv data file and optionally outputs to a csv data file.
    # Parameters: 
    #     filename: name of the csv data file (str), 
    #     column: name of the column to plot (str),
    #     diff_order: order of differencing for time series (int in [0,2]),
    #     selection: to select rows by value (list[tuple[str, Any]], e.g. '[('idle', True)]'),
    #     crop: to cull head or tail of data (tuple[float, float])
        
    # Import
    data_path = path + "\\csv\\" + filename + ".csv"
    data = pd.read_csv(data_path)
    
    # Selection
    assert_type(selection, list[tuple[str,Any]])
    for tup in selection: 
        data = data[data[tup[0]] == tup[1]] 

    # Cropping
    assert_type(crop, tuple[float,float])
    n = data.shape[0]
    if crop[0] != None and int(crop[0]*n) > 0: 
        data = data.tail(-int( n*crop[0] ))
    if crop[1] != None and int(crop[1]*n) > 0: 
        data = data.head(-int( n*crop[1] ))
        
    # Differencing
    assert isinstance(diff_order, int) and diff_order >= 0, f"Order of differencing must be positive."
    for i in range(diff_order):
        diff(data, [column])
    
    # Frequencies
    data['prev'] = data[column].shift(1)
    data = data.dropna(subset=['prev'])
    frequency_table = data.groupby(['prev', column]).size().unstack(fill_value=0)
    
    if output:
        csvfile = open("models/" + output + ".csv", "w")
        csvfile.write(','.join(["prev", "freqs"]))
        csvfile.write('\n')
    result = []
        
    for prev_val, row in frequency_table.iterrows():
        total_count = row.sum()
        if total_count > 0:
            freqs = [(col, count / total_count) for col, count in row.items() if count > 0]
            if output:
                csvfile.write(str(prev_val) + ', ' + str(freqs))
                csvfile.write('\n')
            result.append([prev_val, freqs])
         
    return result

## Model Fitting

def kstest_norm(x: ndarray, bins: int = 1) -> tuple[ndarray, ndarray, float, float]:
    # Fit the normal distribution to a discrete distribution using kstest.
    # Parameters:
    #     x: discrete distribution (numpy array), 
    #     bins: number of bins for hist (int, default is sqrt(len(x))).
    # Returns:
    #     pdf_x: PDF of the distribution normalized over bins (numpy array),
    #     pdf_y: PDF of the best fitting model normalized over bins (numpy array),
    #     mu: mean parameter of the model (float),
    #     sigma: standard deviation parameter of the model (float)
    
    # Fitting
    mu, sigma = np.mean(x), np.std(x)
    pdf_x = np.linspace(np.min(x),np.max(x), bins*10)
    pdf_y = 1.0 / np.sqrt(2*np.pi*sigma**2) * np.exp(-0.5*(pdf_x - mu)**2/sigma**2)
    
    # KS Test
    ks_statistic, p_value = kstest(x, 'norm')
    print("\t KS test: statistic="+str(ks_statistic)+", pvalue="+str(p_value))
    print("\t Gauss model: mu="+str(mu)+", sigma="+str(sigma))
    
    return (pdf_x, pdf_y, mu, sigma)

def kstest_cauchy(x: ndarray, bins: int = 1) -> tuple[ndarray, ndarray, float, float]:
    # Fit the Cauchy distribution to a discrete distribution using kstest.
    # Parameters:
    #     x: discrete distribution (numpy array), 
    #     bins: number of bins for hist (int, default is sqrt(len(x))).
    # Returns:
    #     pdf_x: PDF of the distribution normalized over bins (numpy array),
    #     pdf_y: PDF of the best fitting model normalized over bins (numpy array),
    #     loc: location parameter of the model (float),
    #     scale: scale parameter of the model (float)
    
    def cdf(x, loc, scale):
        return  1/np.pi * np.arctan((x-loc)/scale) + 0.5
    
    def pdf(x, loc, scale):
        return  1/(np.pi * scale * (1 + ((x-loc)/scale)**2))
    
    # Fitting
    pdf_x = np.linspace(np.min(x),np.max(x), bins*10)
    fitter = Fitter(x, distributions=['cauchy'])
    fitter.fit()
    loc, scale = fitter.fitted_param["cauchy"]
    
    # KS Test
    ks_statistic, p_value = kstest(x, cdf, args=(loc, scale))
    print("\t Model parameters: scale="+str(scale), "location="+str(loc))
    print("\t KS test: statistic="+str(ks_statistic)+", pvalue="+str(p_value))
    
    return (pdf_x, pdf(pdf_x, loc, scale), loc, scale)

def kstest_laplace(x: ndarray, bins: int = 1) -> tuple[ndarray, ndarray, float, float]:
    # Fit the Cauchy distribution to a discrete distribution using kstest.
    # Parameters:
    #     x: discrete distribution (numpy array), 
    #     bins: number of bins for hist (int, default is sqrt(len(x))).
    # Returns:
    #     pdf_x: PDF of the distribution normalized over bins (numpy array),
    #     pdf_y: PDF of the best fitting model normalized over bins (numpy array),
    #     loc: location parameter of the model (float),
    #     scale: scale parameter of the model (float)
    
    def cdf(x, loc, scale):
        return 1/2 * (1 + np.sign(x-loc) * (1 - np.exp(-(np.abs(x-loc)/scale))))
    
    # Fitting
    pdf_x = np.linspace(np.min(x),np.max(x), bins*10)
    fitter = Fitter(x, distributions=['laplace'])
    fitter.fit()
    loc, scale = fitter.fitted_param["laplace"]
    
    # Fitting
    ks_statistic, p_value = kstest(x, cdf, args=(loc, scale))
    print("\t Exponential model: scale="+str(scale))
    print("\t KS test: statistic="+str(ks_statistic)+", pvalue="+str(p_value))
    return (pdf_x, cdf(pdf_x, loc, scale), loc, scale)

## Data Visualization
    
def data_plot(filename: str, column: str, diff_order: int = 0, selection: list[tuple[str, Any]] = [],
              crop: tuple[float, float] = (None, None), xlim: tuple[int, int] = (None, None),
              log: bool = False, model: str = None, ylabel: str = None) -> None:
    # Plot a series from a csv data file.
    # Parameters: 
    #     filename: name of the csv data file (str), 
    #     column: name of the column to plot (str),
    #     diff_order: order of differencing for time series (int in [0,2]),
    #     selection: to select rows by value (list[tuple[str, Any]], e.g. '[('idle', True)]'),
    #     crop: to cull head or tail of data (tuple[float, float]),
    #     xlim: to set the limits of the x-axis (tuple[int, int]),
    #     log: to use a logarithmic scale for y-axis (boolean),
    #     ylabel: label for the y-axis (str)
    
    # Import
    data_path = path + "\\csv\\" + filename + ".csv"
    data = pd.read_csv(data_path)
    
    # Selection
    assert_type(selection, list[tuple[str,Any]])
    for tup in selection: 
        data = data[data[tup[0]] == tup[1]] 

    # Cropping
    assert_type(crop, tuple[float,float])
    n = data.shape[0]
    if crop[0] != None and int(crop[0]*n) > 0: 
        data = data.tail(-int( n*crop[0] ))
    if crop[1] != None and int(crop[1]*n) > 0: 
        data = data.head(-int( n*crop[1] ))
        
    # Differencing
    assert isinstance(diff_order, int) and diff_order >= 0, f"Order of differencing must be positive."
    for i in range(diff_order):
        diff(data, [column])
    
    # Plotting
    fig, ax = plt.subplots()
    x = np.array(data[column])
    ax.plot(x)
    plt.xlabel(r'Time order $t$')
    plt.ylabel(ylabel if ylabel else r'Variable value')
    
    # Zoom
    if xlim != (None, None):
        plt.xlim(np.min(x) if xlim[0]==None else xlim[0], np.max(x) if xlim[1]==None else xlim[1])
    
    # Print
    print("Plot of", filename, ":")
    print("\t differencing order:", str(diff_order))
    print("\t selection:", str(selection))
    print("\t crop values:", str(0 if not crop[0] else int(crop[0]*n))+",", str(0 if not crop[1] else int(crop[1]*n)))
    print("\t # of elements:", str(data.shape[0]))
    print("\t log =", str(log))
    plt.show()
    
def data_autocorr(filename: str, column: str, mode: str = 'fft', diff_order: int = 0, selection: list[tuple[str, Any]] = [],
                  crop: tuple[float, float] = (None, None), xlim: tuple[int, int] = (None, None)) -> None:
    # Plot the correlogram of a time series from a csv data file.
    # Parameters:
    #     filename: name of the csv data file (str), 
    #     column: name of the column to plot (str),
    #     mode: acf computation method (str, i.e. 'fft' or 'acf')
    #     diff_order: order of differencing for time series (int in [0,2]),
    #     selection: to select rows by value (list[tuple[str, Any]], e.g. '[('idle', True)]'),
    #     crop: to cull head or tail of data (tuple[float, float]),
    #     xlim: to set the limits of the x-axis (tuple[int, int])
    
    # Import
    data_path = path + "\\csv\\" + filename + ".csv"
    data = pd.read_csv(data_path)
    
    # Selection
    for tup in selection: 
        data = data[data[tup[0]] == tup[1]] 
    
    # Cropping
    n = data.shape[0]
    if crop[0] != None and int(crop[0]*n) > 0: 
        data = data.tail(-int( n*crop[0] ))
    if crop[1] != None and int(crop[1]*n) > 0: 
        data = data.head(-int( n*crop[1] ))
        
    # Differencing
    assert (diff_order >= 0 & diff_order < 3), f"Differencing order {diff_order} not supported."
    for i in range(diff_order):
        diff(data, [column])
        
    # Print
    print("Correlogram of", filename, ":")
    print("\t differencing order:", str(diff_order))
    print("\t selection:", str(selection))
    print("\t crop values:", str(0 if not crop[0] else int(crop[0]*n))+",", str(0 if not crop[1] else int(crop[1]*n)))
    print("\t # of elements:", str(data.shape[0]))
        
    # Plotting
    x = np.array(data[column])
    if mode == 'acf':
        plot_acf(x, fft = True, title=None)
    elif mode == 'fft':
        autocorr = fft_autocorr(x)
        plt.plot(autocorr)    
    else: 
        raise Exception(f"Mode {mode} not found.") 

    # Zoom
    if xlim != (None, None):
        plt.xlim(np.min(x) if xlim[0]==None else xlim[0], np.max(x) if xlim[1]==None else xlim[1])
        
    plt.xlabel(r'Lag $\tau$')
    plt.ylabel(r'Autocorrelation coefficient $\rho$')
    plt.show()

def data_hist(filename: str, column: str, bins: int = 0, diff_order: int = 0, selection: list[tuple[str, Any]] = [],
              crop: tuple[float, float] = (None, None), xlim: tuple[int, int] = (None, None),
              log: bool = False, model: str = None, xlabel: str = None, boxplot: bool = False) -> None:
    # Plot the histogram of a series from a csv data file.
    # Parameters: 
    #     filename: name of the csv data file (str), 
    #     column: name of the column to plot (str),
    #     bins: number of bins for histogram (int, default is sqrt(len(x))),
    #     diff_order: order of differencing for time series (int in [0,2]),
    #     selection: to select rows by value (list[tuple[str, Any]], e.g. '[('idle', True)]'),
    #     crop: to cull head or tail of data (tuple[float, float]),
    #     xlim: to set the limits of the x-axis (tuple[int, int]),
    #     log: to use a logarithmic scale for y-axis (boolean),
    #     model: to fit a model to the histogram (str),
    #     xlabel: label for the xaxis (str)
    
    # Import
    data_path = path + "\\csv\\" + filename + ".csv"
    data = pd.read_csv(data_path)
    
    # Selection
    assert_type(selection, list[tuple[str,Any]])
    for tup in selection: 
        data = data[data[tup[0]] == tup[1]] 

    # Cropping
    assert_type(crop, tuple[float,float])
    n = data.shape[0]
    if crop[0] != None and int(crop[0]*n) > 0: 
        data = data.tail(-int( n*crop[0] ))
    if crop[1] != None and int(crop[1]*n) > 0: 
        data = data.head(-int( n*crop[1] ))
        
    # Differencing
    assert isinstance(diff_order, int) and diff_order >= 0, f"Order of differencing must be positive."
    for i in range(diff_order):
        diff(data, [column])
    
    # Boxplot
    if boxplot:
        x = remove_extremes(data[column])
    else:
        x = np.array(data[column])
    
    # Plotting
    bins = max(int(np.sqrt(data.shape[0])), 1) if bins < 1 else bins
    plt.hist(x, density=True,  log=log, bins=bins)
    plt.xlabel(xlabel if xlabel else r'Variable value')
    plt.ylabel(r'Frequency $f$')
    
    # Zoom
    if xlim != (None, None):
        plt.xlim(np.min(x) if xlim[0]==None else xlim[0], np.max(x) if xlim[1]==None else xlim[1])
        
    # Print
    print("Histogram of", filename, ":")
    print("\t differencing order:", str(diff_order))
    print("\t selection:", str(selection))
    print("\t crop values:", str(0 if not crop[0] else int(crop[0]*n))+",", str(0 if not crop[1] else int(crop[1]*n)))
    print("\t # of elements:", str(data.shape[0]))
    print("\t # of bins:", str(bins))
    print("\t log =", str(log))
    
    # Model
    if model=="norm": 
        pdf_x, pdf_y, mu, sigma = kstest_norm(x, bins)
        model, = plt.plot(pdf_x, pdf_y, 'b--', label="pdf") 
        plt.legend([model], ['model'])
    elif model=='laplace':
        pdf_x, pdf_y, loc, scale = kstest_laplace(x, bins)
        model, = plt.plot(pdf_x, pdf_y, 'b--', label="pdf") 
        plt.legend([model], ['model'])
    elif model=='cauchy':
        pdf_x, pdf_y, loc, scale = kstest_cauchy(x, bins)
        model, = plt.plot(pdf_x, pdf_y, 'b--', label="pdf")
        plt.legend([model], ['model'])
    elif model=="test": 
        fitter = Fitter(x, distributions=['norm', 'laplace', 'cauchy'])
        fitter.fit()
        print(fitter.summary())
    elif model=='export':
        x.to_csv("csv/fit_results.csv")
    plt.show()