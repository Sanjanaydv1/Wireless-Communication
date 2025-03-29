import numpy as np
import matplotlib.pyplot as plt

# Parameters
M = 16  # QAM order (16-QAM)
k = int(np.log2(M))  # Number of bits per symbol
num_symbols = 10000  # Number of symbols to simulate
snr_dB = 10  # Signal-to-noise ratio in dB

# Generate random bit stream
bits = np.random.randint(0, 2, num_symbols * k)

# Map bits to symbols (16-QAM)
def bits_to_symbols(bits, M):
    # Reshape bits into groups of 'k' bits per symbol
    bits = bits.reshape(-1, k)
    symbols = np.zeros(bits.shape[0], dtype=complex)
    
    # Define the 16-QAM constellation points (real and imaginary parts)
    x = np.array([-3, -1, 1, 3])  # Real part values
    y = np.array([-3, -1, 1, 3])  # Imaginary part values
    
    for i in range(bits.shape[0]):
        real_idx = int(bits[i, 0] * 2 + bits[i, 1])  # Mapping for real part
        imag_idx = int(bits[i, 2] * 2 + bits[i, 3])  # Mapping for imaginary part
        symbols[i] = x[real_idx] + 1j * y[imag_idx]
    
    return symbols

# Convert bits to symbols
symbols = bits_to_symbols(bits, M)

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

# Demodulate received symbols
def symbols_to_bits(received_symbols, M):
    # Define the 16-QAM constellation points (real and imaginary parts)
    x = np.array([-3, -1, 1, 3])
    y = np.array([-3, -1, 1, 3])
    
    # Initialize bit array
    bits = np.zeros((len(received_symbols), k), dtype=int)
    
    for i, symbol in enumerate(received_symbols):
        real_idx = np.argmin(np.abs(x - symbol.real))  # Find closest constellation point
        imag_idx = np.argmin(np.abs(y - symbol.imag))  # Find closest constellation point
        
        bits[i, 0] = real_idx // 2
        bits[i, 1] = real_idx % 2
        bits[i, 2] = imag_idx // 2
        bits[i, 3] = imag_idx % 2
    
    return bits.flatten()

# Demodulate the received symbols back to bits
received_bits = symbols_to_bits(received_symbols, M)

# Calculate Bit Error Rate (BER)
bit_errors = np.sum(bits != received_bits)
ber = bit_errors / (num_symbols * k)

# Output the results
print(f"Bit Error Rate (BER): {ber:.6f}")

# Plot the constellation diagram
plt.figure(figsize=(8, 6))
plt.scatter(received_symbols.real, received_symbols.imag, color='blue', alpha=0.5, label="Received symbols")
plt.scatter(symbols.real, symbols.imag, color='red', alpha=0.5, label="Transmitted symbols")
plt.title(f"16-QAM Constellation Diagram (SNR = {snr_dB} dB)")
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.legend()
plt.grid(True)
plt.show()
