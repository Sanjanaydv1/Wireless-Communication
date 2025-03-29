import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.signal.windows import gaussian  # Updated import for gaussian

# Parameters
num_symbols = 10000  # Number of symbols
bit_rate = 1e3  # Bit rate (1 kbit/s)
symbol_rate = bit_rate  # Symbol rate (same as bit rate in GMSK)
snr_dB = 10  # Signal-to-noise ratio in dB
sampling_rate = 10 * bit_rate  # Sampling rate (to avoid aliasing)
time_symbol = 1 / symbol_rate  # Time duration of each symbol
t_symbol = np.linspace(0, num_symbols * time_symbol, int(num_symbols * sampling_rate * time_symbol), endpoint=False)

# Gaussian filter parameters
alpha = 0.3  # Excess bandwidth factor (controls the bandwidth of the Gaussian filter)
filter_length = 5  # Length of the Gaussian filter (in symbols)
gaussian_filter = gaussian(filter_length * int(sampling_rate * time_symbol), std=alpha * time_symbol * sampling_rate)

# Generate random bit stream
bits = np.random.randint(0, 2, num_symbols)

# GMSK Modulation
def gmsk_modulate(bits, symbol_rate, sampling_rate, alpha, gaussian_filter):
    # Gaussian filtering to shape the input bit stream
    # Each bit is mapped to 1 or -1 (for GMSK, typically using 2-phase modulation like MSK)
    symbols = 2 * bits - 1  # Map bits to {+1, -1}
    
    # Filter the symbols using Gaussian filter
    # Upsample the symbols (interpolation to match the sampling rate)
    upsampled_symbols = np.repeat(symbols, int(sampling_rate / symbol_rate))
    
    # Apply Gaussian filter (shaping filter)
    shaped_symbols = convolve(upsampled_symbols, gaussian_filter, mode='same')
    
    # Perform GMSK modulation (integrate the shaped signal)
    phase = np.cumsum(shaped_symbols) / sampling_rate  # Normalize by sampling rate for proper integration
    
    # GMSK signal is exp(j * phase) with real part for I and imaginary part for Q
    gmsk_signal = np.cos(2 * np.pi * symbol_rate * np.arange(len(phase)) / sampling_rate + phase)  # Real part
    return gmsk_signal

# Modulate the bit stream using GMSK
gmsk_signal = gmsk_modulate(bits, symbol_rate, sampling_rate, alpha, gaussian_filter)

# Add AWGN (Additive White Gaussian Noise)
def awgn_noise(signal, snr_dB):
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_dB / 10.0)
    
    # Calculate the signal power
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / snr_linear
    
    # Generate Gaussian noise
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    
    # Add noise to the signal
    return signal + noise

# Transmit the GMSK signal over an AWGN channel
received_signal = awgn_noise(gmsk_signal, snr_dB)

# GMSK Demodulation (Using coherent detection)
def gmsk_demodulate(received_signal, symbol_rate, sampling_rate, alpha, gaussian_filter):
    # Correct phase unwrapping and alignment
    phase = np.unwrap(np.angle(received_signal))  # Unwrap phase to avoid discontinuities
    diff_signal = np.diff(phase) * sampling_rate / (2 * np.pi * symbol_rate)  # Normalize by symbol rate
    demodulated_bits = (diff_signal > 0).astype(int)  # Threshold decision
    return demodulated_bits[:len(bits)]  # Ensure output length matches input bits

# Demodulate the received signal
demodulated_bits = gmsk_demodulate(received_signal, symbol_rate, sampling_rate, alpha, gaussian_filter)

# Calculate Bit Error Rate (BER)
bit_errors = np.sum(bits != demodulated_bits)
ber = bit_errors / num_symbols

# Output the results
print(f"Bit Error Rate (BER): {ber:.6f}")

# Plot the transmitted and received signals
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(np.real(gmsk_signal[:1000]), label="Transmitted GMSK Signal (Real Part)")
plt.title("Transmitted GMSK Signal (Real Part)")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(np.real(received_signal[:1000]), label="Received GMSK Signal (Real Part)", color='orange')
plt.title("Received GMSK Signal (Real Part) with AWGN")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()
