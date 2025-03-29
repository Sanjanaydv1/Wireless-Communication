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

# Function to compute LPC residual (excitation signal)
def lpc_residual(signal, lpc_coeffs):
    return lfilter(lpc_coeffs, [1], signal)

# Generate Stochastic (Fixed) Codebook
def generate_codebook(size, length):
    return np.random.randn(size, length)

# CELP Encoding: Find the best excitation vector from the codebook
def celp_encode(residual, codebook):
    best_index = 0
    best_gain = 0
    best_error = np.inf
    frame_size = len(residual)

    for i, codeword in enumerate(codebook):
        gain = np.dot(residual, codeword) / np.dot(codeword, codeword)
        reconstructed = gain * codeword
        error = np.sum((residual - reconstructed) ** 2)
        
        if error < best_error:
            best_error = error
            best_index = i
            best_gain = gain

    return best_index, best_gain

# CELP Decoding: Retrieve excitation from codebook
def celp_decode(index, gain, codebook):
    return gain * codebook[index]

# LPC Synthesis (Reconstructs speech signal)
def lpc_synthesis(excitation, lpc_coeffs):
    return lfilter([1], lpc_coeffs, excitation)

# Generate a synthetic speech-like signal (sine wave + noise)
fs = 8000  # Sampling rate (8 kHz, common for speech)
t = np.linspace(0, 1, fs, endpoint=False)
signal = np.sin(2 * np.pi * 200 * t) + 0.5 * np.random.randn(len(t))

# LPC Order and Frame Size
lpc_order = 10
frame_size = 160  # 20ms frames
num_frames = len(signal) // frame_size

# Generate a random codebook
codebook_size = 64  # Number of codebook entries
codebook = generate_codebook(codebook_size, frame_size)

# Store reconstructed signal
reconstructed_signal = np.zeros_like(signal)

for i in range(num_frames):
    # Extract frame
    frame = signal[i * frame_size : (i + 1) * frame_size]

    # Compute LPC coefficients
    lpc_coeffs, _ = lpc(frame, lpc_order)

    # Compute residual (excitation signal)
    residual = lpc_residual(frame, lpc_coeffs)

    # Find best excitation codeword
    best_index, best_gain = celp_encode(residual, codebook)

    # Decode excitation
    excitation = celp_decode(best_index, best_gain, codebook)

    # Synthesize speech from excitation
    reconstructed_frame = lpc_synthesis(excitation, lpc_coeffs)

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
plt.title("Reconstructed Speech Signal using CELP")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
