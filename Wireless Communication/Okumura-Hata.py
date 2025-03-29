import numpy as np
import matplotlib.pyplot as plt

def okumura_hata_path_loss(f, h_t, h_r, d, area="urban"):
    """
    Calculate path loss using Okumura-Hata model.

    Parameters:
        f  : Frequency in MHz (150 MHz to 1500 MHz)
        h_t: Transmitter antenna height in meters (30m to 200m)
        h_r: Receiver antenna height in meters (1m to 10m)
        d  : Distance in km (1 km to 20 km)
        area: "urban", "suburban", or "rural"
    
    Returns:
        Path loss in dB.
    """
    if f < 150 or f > 1500:
        raise ValueError("Frequency should be between 150 MHz and 1500 MHz")

    if h_t < 30 or h_t > 200:
        raise ValueError("Transmitter height should be between 30m and 200m")

    if h_r < 1 or h_r > 10:
        raise ValueError("Receiver height should be between 1m and 10m")

    if d < 1 or d > 20:
        raise ValueError("Distance should be between 1 km and 20 km")

    # Mobile station antenna height correction factor
    if f >= 300:  # Large city
        a_hr = 3.2 * (np.log10(11.75 * h_r))**2 - 4.97
    else:  # Small/Medium city
        a_hr = (1.1 * np.log10(f) - 0.7) * h_r - (1.56 * np.log10(f) - 0.8)

    # Urban path loss
    PL_urban = 69.55 + 26.16 * np.log10(f) - 13.82 * np.log10(h_t) - a_hr + \
               (44.9 - 6.55 * np.log10(h_t)) * np.log10(d)

    if area == "urban":
        return PL_urban
    elif area == "suburban":
        return PL_urban - 2 * (np.log10(f / 28))**2 - 5.4
    elif area == "rural":
        return PL_urban - 4.78 * (np.log10(f))**2 + 18.33 * np.log10(f) - 40.94
    else:
        raise ValueError("Invalid area type. Choose from 'urban', 'suburban', or 'rural'.")

# Simulation parameters
f = 900  # MHz
h_t = 50  # m
h_r = 5  # m
d = np.linspace(1, 20, 100)  # Distance from 1 to 20 km

# Calculate path loss for different environments
PL_urban = [okumura_hata_path_loss(f, h_t, h_r, di, "urban") for di in d]
PL_suburban = [okumura_hata_path_loss(f, h_t, h_r, di, "suburban") for di in d]
PL_rural = [okumura_hata_path_loss(f, h_t, h_r, di, "rural") for di in d]

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(d, PL_urban, label="Urban", color="r")
plt.plot(d, PL_suburban, label="Suburban", color="g")
plt.plot(d, PL_rural, label="Rural", color="b")
plt.xlabel("Distance (km)")
plt.ylabel("Path Loss (dB)")
plt.title("Okumura-Hata Path Loss Model")
plt.legend()
plt.grid(True)
plt.show()
