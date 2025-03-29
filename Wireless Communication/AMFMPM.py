import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
fs = 10000  # Sampling frequency (Hz)
t = np.linspace(0, 0.05, fs)  # Time vector (0 to 50ms)
Am = 1  # Message signal amplitude
fm = 50  # Message signal frequency (Hz)
Ac = 2  # Carrier amplitude
fc = 500  # Carrier frequency (Hz)
kf = 100  # Frequency deviation constant
kp = np.pi/2  # Phase deviation constant

# Message signal (modulating signal)
message = Am * np.cos(2 * np.pi * fm * t)

# Carrier signal
carrier = Ac * np.cos(2 * np.pi * fc * t)

# Amplitude Modulation (AM) Signal
am_signal = (1 + message) * Ac * np.cos(2 * np.pi * fc * t)

# Frequency Modulation (FM) Signal
fm_signal = Ac * np.cos(2 * np.pi * fc * t + kf * np.sin(2 * np.pi * fm * t))

# Phase Modulation (PM) Signal
pm_signal = Ac * np.cos(2 * np.pi * fc * t + kp * message)

# Plot results
plt.figure(figsize=(12, 11))

# Plot message signal
plt.subplot(5, 1, 1)
plt.plot(t, message, 'g')
plt.title("Message Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

# Plot carrier signal
plt.subplot(5, 1, 2)
plt.plot(t, carrier, 'r')
plt.title("Carrier Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

# Plot AM signal
plt.subplot(5, 1, 3)
plt.plot(t, am_signal, 'c')
plt.title("Amplitude Modulated Signal (AM)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

# Plot FM signal
plt.subplot(5, 1, 4)
plt.plot(t, fm_signal, 'b')
plt.title("Frequency Modulated Signal (FM)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

# Plot PM signal
plt.subplot(5, 1, 5)
plt.plot(t, pm_signal, 'm')
plt.title("Phase Modulated Signal (PM)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()
