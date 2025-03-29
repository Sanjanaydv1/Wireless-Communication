import numpy as np
import matplotlib.pyplot as plt

# Define parameters
fs = 10000  # Sampling frequency
Tb = 1e-3   # Bit duration (1 ms per bit)
fc = 10e3   # Carrier frequency for ASK, PSK
f1 = 15e3   # Frequency for FSK (bit 1)
f0 = 5e3    # Frequency for FSK (bit 0)
Ac = 1      # Carrier amplitude

# Generate a random binary sequence
data_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
num_bits = len(data_bits)

# Time vector for one bit duration
t_bit = np.linspace(0, Tb, int(fs * Tb), endpoint=False)
t = np.linspace(0, num_bits * Tb, num_bits * len(t_bit), endpoint=False)

# Generate signals
ask_signal = np.array([])
fsk_signal = np.array([])
psk_signal = np.array([])

for bit in data_bits:
    if bit == 1:
        ask_wave = Ac * np.cos(2 * np.pi * fc * t_bit)  # ASK (Bit = 1)
        fsk_wave = Ac * np.cos(2 * np.pi * f1 * t_bit)  # FSK (Bit = 1)
        psk_wave = Ac * np.cos(2 * np.pi * fc * t_bit)  # PSK (0 phase shift)
    else:
        ask_wave = np.zeros_like(t_bit)                # ASK (Bit = 0)
        fsk_wave = Ac * np.cos(2 * np.pi * f0 * t_bit)  # FSK (Bit = 0)
        psk_wave = Ac * np.cos(2 * np.pi * fc * t_bit + np.pi)  # PSK (180° shift)
    
    ask_signal = np.concatenate((ask_signal, ask_wave))
    fsk_signal = np.concatenate((fsk_signal, fsk_wave))
    psk_signal = np.concatenate((psk_signal, psk_wave))

# Plot the signals
plt.figure(figsize=(12, 9))

# Plot binary data
plt.subplot(4, 1, 1)
plt.step(np.arange(num_bits), data_bits, where='post', color='k', linewidth=2)
plt.title("Binary Data")
plt.ylim(-0.2, 1.2)
plt.ylabel("Bit")
plt.grid(True)

# Plot ASK signal
plt.subplot(4, 1, 2)
plt.plot(t, ask_signal, 'r')
plt.title("Amplitude Shift Keying (ASK)")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot FSK signal
plt.subplot(4, 1, 3)
plt.plot(t, fsk_signal, 'b')
plt.title("Frequency Shift Keying (FSK)")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot PSK signal
plt.subplot(4, 1, 4)
plt.plot(t, psk_signal, 'g')
plt.title("Phase Shift Keying (PSK)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()
