import numpy as np
import matplotlib.pyplot as plt

# Parameters
M = 4  # QPSK (4 symbols)
k = int(np.log2(M))  # Number of bits per symbol (2 bits per symbol for QPSK)
num_symbols = 10000  # Number of symbols to simulate
snr_dB = 10  # Signal-to-noise ratio in dB

# Generate random bit stream
bits = np.random.randint(0, 2, num_symbols * k)

# QPSK Modulation
def bits_to_symbols(bits):
    # Reshape bits into groups of 2 bits per symbol
    bits = bits.reshape(-1, 2)
    symbols = np.zeros(bits.shape[0], dtype=complex)
    
    # QPSK constellation points: (1+1j), (-1+1j), (-1-1j), (1-1j)
    for i in range(bits.shape[0]):
        # Map bits to QPSK symbols
        if bits[i, 0] == 0 and bits[i, 1] == 0:
            symbols[i] = 1 + 1j  # (00)
        elif bits[i, 0] == 0 and bits[i, 1] == 1:
            symbols[i] = -1 + 1j  # (01)
        elif bits[i, 0] == 1 and bits[i, 1] == 0:
            symbols[i] = -1 - 1j  # (10)
        else:
            symbols[i] = 1 - 1j  # (11)
    
    return symbols

# Convert bits to symbols
symbols = bits_to_symbols(bits)

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

# QPSK Demodulation
def symbols_to_bits(received_symbols):
    # Define QPSK constellation points
    constellation = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    
    # Initialize bit array
    bits = np.zeros((len(received_symbols), k), dtype=int)
    
    for i, symbol in enumerate(received_symbols):
        # Find the closest constellation point
        distances = np.abs(np.array(constellation) - symbol)
        closest_symbol_index = np.argmin(distances)
        
        # Map the constellation point to bits
        if closest_symbol_index == 0:
            bits[i] = [0, 0]
        elif closest_symbol_index == 1:
            bits[i] = [0, 1]
        elif closest_symbol_index == 2:
            bits[i] = [1, 0]
        else:
            bits[i] = [1, 1]
    
    return bits.flatten()

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
plt.title(f"QPSK Constellation Diagram (SNR = {snr_dB} dB)")
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.legend()
plt.grid(True)
plt.show()
