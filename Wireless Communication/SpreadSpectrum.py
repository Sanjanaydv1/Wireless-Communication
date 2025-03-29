import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_bits = 1000  # Number of bits to transmit
bit_rate = 1e3  # Bit rate (1 kbps)
chip_rate = 10 * bit_rate  # Chip rate (higher than bit rate for spreading)
snr_dB = 10  # Signal-to-noise ratio in dB
sampling_rate = 10 * bit_rate  # Sampling rate for simulation

# Generate random bit stream
bits = np.random.randint(0, 2, num_bits)

# Generate Pseudo-random sequence (PN sequence) for DSSS spreading
# Ensure spreading_factor is an integer
spreading_factor = int(chip_rate // bit_rate)
pn_sequence = np.random.randint(0, 2, spreading_factor)

# DSSS Modulation
def dsss_modulate(bits, pn_sequence, chip_rate, bit_rate, sampling_rate):
    modulated_signal = []
    
    # Map bits to +1/-1 (BPSK modulation)
    for bit in bits:
        bit_symbol = 2 * bit - 1
        
        # Repeat the bit using the PN sequence to spread the signal
        chip_signal = np.repeat(bit_symbol * pn_sequence, chip_rate // bit_rate)
        
        # Upsample and add the spread signal to the modulated signal
        modulated_signal.extend(chip_signal)
        
    return np.array(modulated_signal)

# Modulate the bit stream using DSSS
dsss_signal = dsss_modulate(bits, pn_sequence, chip_rate, bit_rate, sampling_rate)

# Add AWGN (Additive White Gaussian Noise)
def awgn_noise(signal, snr_dB):
    snr_linear = 10 ** (snr_dB / 10.0)
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise

# Transmit the DSSS signal over an AWGN channel
received_signal = awgn_noise(dsss_signal, snr_dB)

# DSSS Demodulation
def dsss_demodulate(received_signal, pn_sequence, chip_rate, bit_rate, sampling_rate):
    # Perform correlation with the PN sequence to despread the signal
    correlated_signal = np.correlate(received_signal, pn_sequence, mode='same')
    
    # Downsample the signal to match the bit rate
    demodulated_signal = correlated_signal[::spreading_factor]
    
    # Make decisions based on the sign of the correlated signal
    demodulated_bits = (demodulated_signal > 0).astype(int)
    
    return demodulated_bits

# Demodulate the received signal
demodulated_bits = dsss_demodulate(received_signal, pn_sequence, chip_rate, bit_rate, sampling_rate)

# Trim the demodulated signal to match the length of the original bit stream
demodulated_bits = demodulated_bits[:num_bits]

# Calculate Bit Error Rate (BER)
bit_errors = np.sum(bits != demodulated_bits)
ber = bit_errors / num_bits

# Output the results
print(f"Bit Error Rate (BER): {ber:.6f}")

# Plot the transmitted and received signals
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(dsss_signal[:1000], label="Transmitted DSSS Signal")
plt.title("Transmitted DSSS Signal")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(np.real(received_signal[:1000]), label="Received DSSS Signal with AWGN", color='orange')
plt.title("Received DSSS Signal with AWGN")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()
