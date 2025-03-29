import numpy as np
import matplotlib.pyplot as plt

# Parameters for the sine wave
frequency = 5  # in Hz
amplitude = 1  # in units
sampling_rate = 1000  # in Hz
duration = 1  # in seconds

# Generate time values
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Generate sine wave
sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)

# Generate a square wave
square_wave = amplitude * np.sign(sine_wave)

#Generete triangular wave
triangular_wave = amplitude * (2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1)

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Plot the sine wave
axs[0].plot(t, sine_wave)
axs[0].set_title("Sine Wave")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Amplitude")
axs[0].grid()

# Plot the square wave
axs[1].plot(t, square_wave)
axs[1].set_title("Square Wave")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Amplitude")
axs[1].grid()

# Plot the triangular wave
axs[2].plot(t, triangular_wave, color='orange')
axs[2].set_title("Triangular Wave")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Amplitude")
axs[2].grid()



# Adjust layout and show the plots
plt.tight_layout()
plt.show()