import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet2, butter, sosfilt, find_peaks
# importing libraries necessary to plot
#Read data from file
from pyscript import display
from js import document
from io import BytesIO
import asyncio
import pyodide




# Data and datetime split
def data_time_split(df, signal_for_analysis):
        # The format of the date and time columns is: dd mm yy HH:MM:SS
        datetime = pd.to_datetime(df.iloc[:,0]).to_numpy()
        activity = df[signal_for_analysis].astype(np.float64).to_numpy()
        return datetime, activity





# Transform scales to frequencies
def f (s, w, fs = 1, max_power_indices = None):
        return (w*fs / (2*s[max_power_indices]*np.pi))

def peak_finder(array, total_peaks):
        peak_ind = find_peaks(array, height=0)
        peak_height = np.argsort(peak_ind[1]['peak_heights'])[::-1]
        peak_ind = peak_ind[0][peak_height]
        if np.all(np.isnan(peak_ind)) == False and np.all(peak_ind != None) and total_peaks < len(peak_ind):
                peaks = peak_ind[:total_peaks].astype(int)
               
        else:
                max_datapoint = np.argsort(np.abs(array), axis = 0)[::-1][:total_peaks]
                peaks = max_datapoint.astype(int)
        return peaks

# Compute the CWT
def cwt_compute(signal, sampling_rate = 60, min_period = 18, max_period = 32, components = 1):

        # Compute the frequencies of interest and adjust by period
        frequencies_of_interest = 1/(np.array([min_period, max_period])*sampling_rate)

        # Compute the scales
        w = sampling_rate*max_period # Width of the wavelet in time domain (max_period)
        scales = w / (2 * frequencies_of_interest * np.pi)
        scales = np.arange(scales.min(), scales.max(), w*0.05)

        # Perform the wavelet transform
        cwtmatr = cwt(signal, morlet2, scales, dtype=np.complex128, w = w)

        # Find the top k powers
        top_power_indices = np.apply_along_axis(peak_finder, 0, np.abs(cwtmatr), total_peaks = components)  

        # Find the top k powers, frequencies, and phases
        # powers = np.abs(cwtmatr[top_power_indices, np.arange(len(signal))])
        freqs = f(scales, w, 1, top_power_indices)
        # phases = np.angle(cwtmatr[top_power_indices, np.arange(len(signal))])

        return freqs

# Cosinor analysis
def multicomponent_cosinor(freqs):
        # Compute the cosinor
        linear_space = np.repeat(np.arange(0,freqs.shape[1]).reshape(-1,1).T, repeats = freqs.shape[0], axis=0)

        cosine_component = np.cos(linear_space * (np.pi*2*freqs))
        sine_component = np.sin(linear_space * (np.pi*2*freqs))

        synthetic_signal = np.vstack((cosine_component, sine_component))
        
        return synthetic_signal

def signal_smoother(signal, sampling_rate = 60, min_period = 18, max_period = 32, order = 2):
        sos = butter(order, [1/max_period, 1/min_period], 'bandpass', fs=sampling_rate, output='sos')
        smooth_signal = sosfilt(sos, signal)
        return smooth_signal
        

# Linar Regression
def cosinor_lm(signal, synthetic_signal):
        # Create the model
        X = sm.add_constant(synthetic_signal.T)
        model = sm.OLS(signal, X, hasconst=True)
        # Fit the model to the data
        results = model.fit()
        # Evaluate the model
        return results

# Rolling Regression
def rolling_cosinor_lm(signal, synthetic_signal, sampling_rate = 60, max_period = 32):
        # Create the model
        # Add a paddding of zeroes to the signal and the synthetic signal the size of the window
        # pad_size = np.int64(np.round(sampling_rate*max_period))

        
        # signal = np.pad(signal, (pad_size,0), 'constant', constant_values=(np.nan))

        # synthetic_signal = np.apply_along_axis(np.pad, 1, synthetic_signal, pad_width = (pad_size,0), mode = 'constant', constant_values = (np.nan))

        synthetic_signal = sm.add_constant(synthetic_signal.T)
  
        model = RollingOLS(signal, synthetic_signal, window = sampling_rate*max_period, min_nobs = (sampling_rate*max_period))
        # Fit the model to the data
        results = model.fit()
        # Evaluate the model
        return results.rsquared_adj

# Adjustment of Synthetic Signal
def adjust_synthetic_signal(synthetic_signal, model):
        MESOR = model.params[0]
        coeffs = model.params[1:]
        pvalues = model.pvalues[1:] <= 0.05
        dominant_signal = MESOR + (coeffs[pvalues] * synthetic_signal[pvalues].T)
        return dominant_signal.T

# report the results
def convenience_reporter(signal, smooth_signal, synthetic_signal, freqs, cosinor_model, rolling_cosinor_model, sampling_rate = 60,  path = None):
        
        times = np.arange(0, len(signal)) / sampling_rate

        fig, ax = plt.subplots(5, 1)
        fig.set_size_inches(10, 20)

        # Raw Signal
        ax[0].plot(times, signal)
        ax[0].set_ylabel('Raw Signal')
        ax[0].grid(True)




        # Rolling R-Squared
        ax[1].scatter(times, rolling_cosinor_model)
        ax[1].hlines(0.5, xmin=0, xmax=times[-1], linestyles='--', color='black')
        ax[1].set_ylim(0,1)
        ax[1].set_title('Rolling R-Squared')
        ax[1].set_ylabel('R-Squared')
        ax[1].grid(True)


        # Cosinor Fit with Smoothed Data
    
        # Set figure size
        ax[2].plot(times, synthetic_signal.sum(axis=0), label='Cosinor', linestyle='--', color='red')
        # Create the second plot
        axtwin = ax[2].twinx()
        axtwin.plot(times, smooth_signal, label='Signal')
        # Set the labels for the two y axes
        ax[2].set_ylabel('Cosinor')
        axtwin.set_ylabel('Smooth Signal')

        ax[2].set_title(f'Cosinor Fit with Adj. R-Squared: {cosinor_model.rsquared_adj}')
        ax[2].grid(True)
        ax[2].legend()


        # Cosinor Fit with Raw Data
  
        # Set figure size
        ax[3].plot(times, synthetic_signal.sum(axis=0), label='Cosinor', linestyle='--', color='red')
        # Create the second plot
        axtwin = ax[3].twinx()
        axtwin.plot(times, signal, label='Signal')
        # Set the labels for the two y axes
        ax[3].set_ylabel('Cosinor')
        axtwin.set_ylabel('Raw Signal')
        ax[3].set_title(f'Cosinor Fit with Adj. R-Squared: {cosinor_model.rsquared_adj}')
        ax[3].grid(True)
        ax[3].legend()

        # Periodogram
        for i in range(freqs.shape[0]):
            ax[4].plot(times, (1/freqs[i, :])/sampling_rate, label= f'Component {i+1}', alpha = 0.5)
            ax[4].set_xlabel('Time (hours)')
            ax[4].set_ylabel('Period (hours)')
            ax[4].set_title('Periodogram')
            ax[4].legend()
        
        display(fig.show())

    
# Whenever any of the search paramters are updated, update all search paramters
# We will need to create a listener for all of the search paramters
async def update_search_paramters(event):
        global samples_per_hour
        global max_period
        global min_period
        global number_of_components
        global signal_for_analysis
        samples_per_hour = document.getElementById('samples_per_hour').value
        max_period = document.getElementById('max_period').value
        min_period = document.getElementById('min_period').value
        number_of_components = document.getElementById('number_of_components').value
        signal_for_analysis = document.getElementById('signal').value
        document.getElementById('params').innerHTML = "Parameters have been set!"
        return



# Move data from local storage to python through the web worker 


async def upload(event):
        myfile = event.target.files.item(0)
        arrayBuffer = await myfile.arrayBuffer()
        file_bytes = arrayBuffer.to_bytes()
        global df
        df = pd.read_csv(BytesIO(file_bytes))
        cols = df.columns.to_numpy().astype(str)
        df.columns = cols
        display(df.head(), target = 'csv-content')
      

get_file_proxy = pyodide.ffi.create_proxy(upload)
document.getElementById('myfile').addEventListener('change', get_file_proxy)


async def sample_data(event):
        global df
        url = "https://raw.githubusercontent.com/edpclau/Client-Side-Circadian-Dynamics/main/Monitor4%20copy.csv"
        df = pd.read_csv(url)
        cols = df.columns.to_numpy().astype(str)
        df.columns = cols
        display(df.head(), target = 'csv-content')


# Function to run the analysis

async def run_analysis(event):
        #Get global variables
        global df
        global samples_per_hour
        global max_period
        global min_period
        global number_of_components
        global signal_for_analysis
        # Turn into integers
        samples_per_hour = np.int64(samples_per_hour)
        max_period = np.int64(max_period)
        min_period = np.int64(min_period)
        number_of_components = np.int64(number_of_components)

        # Split the data into datetime and activity
        datetime, signal = data_time_split(df, signal_for_analysis)
        # Smooth the signal

        smooth_signal = signal_smoother(signal, sampling_rate = samples_per_hour, min_period = min_period, max_period = max_period, order = 2)
    
        # Extract the frequency components of the signal
        freqs = cwt_compute(smooth_signal, sampling_rate = samples_per_hour, min_period = min_period, max_period = max_period, components = number_of_components)

        print('The frequency components have been extracted.')
        #Generate a synthetic signal with the same frequency components
        synthetic_signal = multicomponent_cosinor(freqs)
 
        print('The synthetic signal has been generated.')
        #Fit a cosinor model to the signal
        cosinor_model = cosinor_lm(smooth_signal, synthetic_signal)

        print('The cosinor model has been fit.')
        #Fit a rolling cosinor model to the signal
        rolling_cosinor_model = rolling_cosinor_lm(smooth_signal, synthetic_signal, sampling_rate = samples_per_hour, max_period = max_period)
    
        print('The rolling cosinor model has been fit.')
        #Adjust the synthetic signal to the cosinor model
        synthetic_signal = adjust_synthetic_signal(synthetic_signal, cosinor_model)
        
        print('The synthetic signal has been adjusted.')
        #Plot the results
        convenience_reporter(signal, smooth_signal, synthetic_signal, freqs, cosinor_model, rolling_cosinor_model, sampling_rate = samples_per_hour,  path = None)
        return