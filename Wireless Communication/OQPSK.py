import numpy as np
import matplotlib.pyplot as plt

# Parameters
M = 4  # OQPSK (4 symbols)
k = int(np.log2(M))  # Number of bits per symbol (2 bits per symbol for OQPSK)
num_symbols = 10000  # Number of symbols to simulate
snr_dB = 10  # Signal-to-noise ratio in dB

# Generate random bit stream
bits = np.random.randint(0, 2, num_symbols * k)

# OQPSK Modulation
def bits_to_symbols(bits):
    # Reshape bits into two separate streams (I and Q components)
    bits_I = bits[::2]  # In-phase components (I)
    bits_Q = bits[1::2]  # Quadrature components (Q)
    
    # Define the OQPSK constellation points
    symbols_I = 2 * bits_I - 1  # Map bits {0, 1} to {1, -1} for I channel
    symbols_Q = 2 * bits_Q - 1  # Map bits {0, 1} to {1, -1} for Q channel
    
    return symbols_I, symbols_Q

# Convert bits to OQPSK symbols
symbols_I, symbols_Q = bits_to_symbols(bits)

# Apply OQPSK offset (delay the Q channel)
def apply_offset(symbols_Q):
    # Apply a half symbol delay to the Q channel for OQPSK
    return np.roll(symbols_Q, 1)  # Delay Q by 1 sample (half-symbol delay)

# Apply offset to the Q channel
symbols_Q_offset = apply_offset(symbols_Q)

# Combine the I and offset Q channels
symbols = symbols_I + 1j * symbols_Q_offset

# Add noise (AWGN)
def awgn_noise(symbols, snr_dB):
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_dB / 10.0)
    
    # Calculate noise power
    symbol_power = np.mean(np.abs(symbols) ** 2)
    noise_power = symbol_power / snr_linear
    
    # Generate complex Gaussian noise
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols)))
    
    # Add noise to symbols
    return symbols + noise

# Transmit the symbols over the noisy channel
received_symbols = awgn_noise(symbols, snr_dB)

# OQPSK Demodulation
def symbols_to_bits(received_symbols):
    # Define QPSK constellation points (I and Q separately)
    symbols_I_received = received_symbols.real
    symbols_Q_received = received_symbols.imag
    
    # Demodulate the symbols back into bits
    bits_I_received = (symbols_I_received > 0).astype(int)
    bits_Q_received = (symbols_Q_received > 0).astype(int)
    
    # Combine bits
    received_bits = np.empty(2 * len(bits_I_received), dtype=int)
    received_bits[::2] = bits_I_received
    received_bits[1::2] = bits_Q_received
    
    return received_bits

# Demodulate the received symbols back to bits
received_bits = symbols_to_bits(received_symbols)

# Calculate Bit Error Rate (BER)
bit_errors = np.sum(bits != received_bits)
ber = bit_errors / (num_symbols * k)

# Output the results
print(f"Bit Error Rate (BER): {ber:.6f}")

# Plot the constellation diagram
plt.figure(figsize=(8, 6))
plt.scatter(received_symbols.real, received_symbols.imag, color='blue', alpha=0.5, label="Received symbols")
plt.scatter(symbols.real, symbols.imag, color='red', alpha=0.5, label="Transmitted symbols")
plt.title(f"OQPSK Constellation Diagram (SNR = {snr_dB} dB)")
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.legend()
plt.grid(True)
plt.show()
