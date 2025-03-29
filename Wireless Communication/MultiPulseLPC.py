import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.signal import lfilter

# LPC Function
def lpc(signal, order):
    # Compute autocorrelation
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags
    
    # Build the Toeplitz matrix
    R = toeplitz(autocorr[:order])
    r = autocorr[1:order+1]
    
    # Solve for LPC coefficients
    a = np.linalg.solve(R, r)
    
    # Compute residual energy
    error = autocorr[0] - np.dot(a, autocorr[1:order+1])
    
    return np.concatenate(([1], -a)), error

# Compute LPC residual (error signal)
def lpc_residual(signal, lpc_coeffs):
    return lfilter(lpc_coeffs, [1], signal)

# Multi-Pulse Excitation (MPE) selection
def mpe_selection(residual, num_pulses):
    frame_size = len(residual)
    pulses = np.zeros(frame_size)
    
    for _ in range(num_pulses):
        # Find the index of the highest energy peak in residual
        idx = np.argmax(np.abs(residual))
        
        # Assign a pulse at that location
        pulses[idx] = residual[idx]
        
        # Remove the used peak to avoid duplicates
        residual[idx] = 0

    return pulses

# LPC Synthesis (Reconstructing the signal)
def lpc_synthesis(mpe_pulses, lpc_coeffs):
    return lfilter([1], lpc_coeffs, mpe_pulses)

# Generate a sample speech-like signal (a sine wave + noise)
fs = 8000  # 8 kHz sampling rate (common in speech)
t = np.linspace(0, 1, fs, endpoint=False)
signal = np.sin(2 * np.pi * 200 * t) + 0.5 * np.random.randn(len(t))

# Set LPC order and number of pulses
lpc_order = 10
num_pulses = 5  # Number of excitation pulses per frame

# Divide signal into frames for LPC analysis
frame_size = 160  # 20ms frames for 8kHz speech
num_frames = len(signal) // frame_size
reconstructed_signal = np.zeros_like(signal)

for i in range(num_frames):
    # Extract frame
    frame = signal[i * frame_size : (i + 1) * frame_size]
    
    # Compute LPC coefficients
    lpc_coeffs, _ = lpc(frame, lpc_order)
    
    # Compute residual
    residual = lpc_residual(frame, lpc_coeffs)
    
    # Select multi-pulse excitation
    mpe_pulses = mpe_selection(residual, num_pulses)
    
    # Synthesize speech from pulses
    reconstructed_frame = lpc_synthesis(mpe_pulses, lpc_coeffs)
    
    # Store reconstructed frame
    reconstructed_signal[i * frame_size : (i + 1) * frame_size] = reconstructed_frame

# Plot original and reconstructed signals
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(signal[:1000])
plt.title("Original Speech Signal")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(reconstructed_signal[:1000])
plt.title("Reconstructed Speech Signal using MPE-LPC")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
