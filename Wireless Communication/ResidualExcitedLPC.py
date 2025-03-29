import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.signal import lfilter

# LPC Function (Computes LPC coefficients)
def lpc(signal, order):
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # Keep only positive lags

    # Build Toeplitz matrix
    R = toeplitz(autocorr[:order])
    r = autocorr[1:order + 1]

    # Solve for LPC coefficients
    a = np.linalg.solve(R, r)
    
    # Compute residual energy
    error = autocorr[0] - np.dot(a, autocorr[1:order + 1])
    
    return np.concatenate(([1], -a)), error

# Compute LPC residual (excitation signal)
def lpc_residual(signal, lpc_coeffs):
    return lfilter(lpc_coeffs, [1], signal)

# Quantize residual (Simple uniform quantization)
def quantize_residual(residual, num_levels):
    min_val = np.min(residual)
    max_val = np.max(residual)
    step = (max_val - min_val) / num_levels
    quantized_residual = np.round((residual - min_val) / step) * step + min_val
    return quantized_residual

# LPC Synthesis (Reconstructs speech signal)
def lpc_synthesis(residual, lpc_coeffs):
    return lfilter([1], lpc_coeffs, residual)

# Generate a sample speech-like signal (a sine wave + noise)
fs = 8000  # Sampling rate (8 kHz, common for speech)
t = np.linspace(0, 1, fs, endpoint=False)
signal = np.sin(2 * np.pi * 200 * t) + 0.5 * np.random.randn(len(t))

# LPC Order and Frame Size
lpc_order = 10
frame_size = 160  # 20ms frames
num_frames = len(signal) // frame_size

# Number of quantization levels for residual
num_levels = 16  

# Store reconstructed signal
reconstructed_signal = np.zeros_like(signal)

for i in range(num_frames):
    # Extract frame
    frame = signal[i * frame_size : (i + 1) * frame_size]

    # Compute LPC coefficients
    lpc_coeffs, _ = lpc(frame, lpc_order)

    # Compute residual (excitation signal)
    residual = lpc_residual(frame, lpc_coeffs)

    # Quantize residual
    quantized_residual = quantize_residual(residual, num_levels)

    # Synthesize speech from quantized residual
    reconstructed_frame = lpc_synthesis(quantized_residual, lpc_coeffs)

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
plt.title("Reconstructed Speech Signal using RE-LPC")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
