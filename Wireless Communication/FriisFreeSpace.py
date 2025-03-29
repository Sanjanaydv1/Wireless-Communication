import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 3e8  # Speed of light in m/s
f = 2.4e9  # Frequency in Hz (e.g., WiFi 2.4 GHz)
Pt = 1  # Transmitted power in Watts
Gt = 1  # Transmitter antenna gain
Gr = 1  # Receiver antenna gain

# Compute wavelength
lambda_ = c / f

# Define distance range (1m to 1000m)
d = np.linspace(1, 1000, 1000)

# Calculate received power using Friis equation
Pr = Pt * (Gt * Gr * lambda_**2) / ((4 * np.pi * d) ** 2)

# Convert power to dBm
Pr_dBm = 10 * np.log10(Pr * 1000)  # Convert W to mW and then to dBm

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(d, Pr_dBm, label="Received Power (dBm)", color='b')
plt.xlabel("Distance (m)")
plt.ylabel("Received Power (dBm)")
plt.title("Friis Free Space Propagation Model")
plt.grid(True)
plt.legend()
plt.show()
