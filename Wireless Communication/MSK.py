import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_symbols = 10000  # Number of symbols
bit_rate = 1e3  # Bit rate (1 kbit/s)
symbol_rate = bit_rate  # Symbol rate (same as bit rate in MSK)
snr_dB = 10  # Signal-to-noise ratio in dB
sampling_rate = 10 * bit_rate  # Sampling rate (to avoid aliasing)
time_symbol = 1 / symbol_rate  # Time duration of each symbol
t_symbol = np.linspace(0, time_symbol, int(sampling_rate * time_symbol), endpoint=False)

# Generate random bit stream
bits = np.random.randint(0, 2, num_symbols)

# MSK Modulation (Continuous Phase Frequency Shift Keying)
def msk_modulate(bits, symbol_rate, sampling_rate):
    # MSK is based on the following formula:
    # s(t) = sqrt(2/T) * cos(2 * pi * f1 * t + phi) for bit 0
    # s(t) = sqrt(2/T) * cos(2 * pi * f2 * t + phi) for bit 1
    
    # Frequency separation: f1 = bit_rate/2, f2 = -bit_rate/2
    f1 = symbol_rate / 2
    f2 = -symbol_rate / 2
    
    # Initialize the signal
    signal = np.array([])
    
    # Generate signal for each bit
    for bit in bits:
        if bit == 0:
            signal = np.concatenate((signal, np.sqrt(2/time_symbol) * np.cos(2 * np.pi * f1 * t_symbol)))
        else:
            signal = np.concatenate((signal, np.sqrt(2/time_symbol) * np.cos(2 * np.pi * f2 * t_symbol)))
    
    return signal

# Modulate the bit stream using MSK
msk_signal = msk_modulate(bits, symbol_rate, sampling_rate)

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

# Transmit the MSK signal over an AWGN channel
received_signal = awgn_noise(msk_signal, snr_dB)

# MSK Demodulation
def msk_demodulate(received_signal, symbol_rate, sampling_rate):
    # The received signal should be demodulated by performing an integral (or coherent detection).
    # This involves mixing with both sinusoids corresponding to f1 and f2, and then low-pass filtering.
    
    f1 = symbol_rate / 2
    f2 = -symbol_rate / 2
    
    # Time vector for one symbol period
    t_symbol = np.linspace(0, 1 / symbol_rate, int(sampling_rate / symbol_rate), endpoint=False)
    
    # Initialize demodulated bits
    demodulated_bits = []
    
    # Split the received signal into chunks corresponding to each symbol
    num_chunks = len(received_signal) // len(t_symbol)
    
    for i in range(num_chunks):
        chunk = received_signal[i * len(t_symbol): (i + 1) * len(t_symbol)]
        
        # Coherent detection: Mix the received signal with the reference carrier (f1 and f2)
        in_phase = np.cos(2 * np.pi * f1 * t_symbol) * chunk
        quadrature = np.cos(2 * np.pi * f2 * t_symbol) * chunk
        
        # Integrate over one symbol period (low-pass filter operation)
        in_phase_integral = np.sum(in_phase)
        quadrature_integral = np.sum(quadrature)
        
        # Decision rule: Choose the bit based on the higher integral value
        if in_phase_integral > quadrature_integral:
            demodulated_bits.append(0)
        else:
            demodulated_bits.append(1)
    
    return np.array(demodulated_bits)

# Demodulate the received signal
demodulated_bits = msk_demodulate(received_signal, symbol_rate, sampling_rate)

# Calculate Bit Error Rate (BER)
bit_errors = np.sum(bits != demodulated_bits)
ber = bit_errors / num_symbols

# Output the results
print(f"Bit Error Rate (BER): {ber:.6f}")

# Plot the transmitted and received signals
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(np.real(msk_signal[:1000]), label="Transmitted Signal (Real Part)")
plt.title("Transmitted MSK Signal (Real Part)")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(np.real(received_signal[:1000]), label="Received Signal (Real Part)", color='orange')
plt.title("Received MSK Signal (Real Part) with AWGN")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()  # Corrected from plt.tight to plt.tight_layout()
plt.show()  # Add plt.show() to display the plots
