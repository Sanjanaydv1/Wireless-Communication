import numpy as np
import random

# Hamming (7,4) Generator Matrix (G) and Parity Check Matrix (H)
G = np.array([[1, 1, 0, 1],  # Generator matrix
              [1, 0, 1, 1],
              [1, 0, 0, 0],
              [0, 1, 1, 1],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

H = np.array([[1, 0, 1, 0, 1, 0, 1],  # Parity check matrix
              [0, 1, 1, 0, 0, 1, 1],
              [0, 0, 0, 1, 1, 1, 1]])

# Encoding function
def hamming_encode(data_bits):
    return np.mod(np.dot(G, data_bits), 2)

# Decoding function (Error Correction)
def hamming_decode(received_codeword):
    syndrome = np.mod(np.dot(H, received_codeword), 2)
    error_pos = int("".join(str(x) for x in syndrome), 2) - 1
    
    if error_pos >= 0:
        received_codeword[error_pos] ^= 1  # Correct the bit

    return received_codeword[2], received_codeword[4], received_codeword[5], received_codeword[6]  # Extract data bits

# Simulate wireless communication with random bit errors
def simulate_wireless_communication(data_bits, error_probability=0.1):
    # Encode the data bits
    encoded_bits = hamming_encode(data_bits)
    print("Encoded Bits: ", encoded_bits)

    # Introduce random bit errors based on error probability
    transmitted_bits = encoded_bits.copy()
    for i in range(len(transmitted_bits)):
        if random.random() < error_probability:
            transmitted_bits[i] ^= 1  # Flip the bit

    print("Transmitted Bits (With Noise): ", transmitted_bits)

    # Decode the received bits
    decoded_bits = hamming_decode(transmitted_bits)
    print("Decoded Bits (After Error Correction): ", decoded_bits)

    return decoded_bits

# Example: Simulate wireless communication
data_bits = np.array([1, 0, 1, 1])  # Input 4-bit data
decoded_bits = simulate_wireless_communication(data_bits, error_probability=0.2)
