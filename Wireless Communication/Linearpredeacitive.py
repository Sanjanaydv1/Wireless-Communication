import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.signal import lfilter

# LPC function to compute the LPC coefficients
def lpc(signal, order):
    # Step 1: Autocorrelation
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Positive lags only

    # Step 2: Build the Toeplitz matrix for the autocorrelation
    R = toeplitz(autocorr[:order])

    # Step 3: Compute the LPC coefficients using Levinson-Durbin recursion
    a = np.zeros(order)
    e = np.zeros(order + 1)
    e[0] = autocorr[0]
    for i in range(1, order + 1):
        lambda_val = np.dot(autocorr[1:i+1], a[i-1::-1]) / e[i-1]
        a[i-1] = lambda_val
        e[i] = e[i-1] * (1 - lambda_val**2)

    return a, e[order]

# Function to generate residual (error signal)
def lpc_residual(signal, order, coeffs):
    prediction = np.zeros_like(signal)
    for n in range(order, len(signal)):
        prediction[n] = -np.dot(coeffs, signal[n-order:n][::-1])

    residual = signal - prediction
    return residual

# Generate a sample signal (e.g., a sine wave) to simulate
fs = 16000  # Sampling frequency
t = np.linspace(0, 1, fs)  # Time vector (1 second)
signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

# Set LPC order (number of previous samples for prediction)
order = 10

# Calculate LPC coefficients
coeffs, error = lpc(signal, order)

# Compute the residual signal (prediction error)
residual = lpc_residual(signal, order, coeffs)

# Plotting the original signal and the residual
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal (440 Hz Sine Wave)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(t, residual)
plt.title('Residual Signal (Prediction Error)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

print(f"LPC Coefficients (order {order}):")
print(coeffs)
print(f"Prediction Error (e): {error}")
