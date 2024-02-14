import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cheb2ord, cheby2, bilinear, freqs, freqz

# Passband specifications given in discrete-time:
Wp = np.pi * np.array([0.2, 0.4])  # Passband frequencies (rad)
Wr = np.pi * np.array([0.1, 0.5])  # Stopband (rejection) frequencies (rad)
Apass = 3  # Maximum passband attenuation (ripple) (dB)
Astop = 30  # Minimum stopband attenuation (dB)

# Choose convenient sampling frequency
Fs = 0.5  # in Hertz

# Map from discrete to continuous-time using pre-warping
wp = 2 * Fs * np.tan(Wp / 2)  # in rad/s
wr = 2 * Fs * np.tan(Wr / 2)  # in rad/s

# Design analog filter
N, w_design = cheb2ord(wp, wr, Apass, Astop, analog=True)  # Find the filter order
print(f"N = \n {N}")
print(f"w_design = \n {w_design}")
Bs, As = cheby2(
    N, Astop, w_design, "bandpass", analog=True, output="ba"
)  # Obtain H(s) = Bs/As
print(f"Bs = \n {Bs}")
print(f"As = \n {As}")

plt.figure(1)
w, h = freqs(Bs, As)  # Compute the frequency response of H(s)
magnitude = np.abs(h)
phase_deg = np.angle(h) * 180 / np.pi

plt.subplot(2, 1, 1)
plt.loglog(w, magnitude)  # Plot the magnitude response
plt.xlabel("Frequency (rad/s)")
plt.ylabel("Magnitude")
plt.xlim(0.01, 10)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.semilogx(w, phase_deg)  # Plot the phase response in degrees with logarithmic x-axis
plt.xlabel("Frequency (rad/s)")
plt.ylabel("Phase (degrees)")
plt.xlim(0.01, 10)
plt.ylim(-200, 200)
plt.yticks(np.linspace(-200, 200, 5))
plt.grid(True)

plt.subplots_adjust(hspace=0.5)  # Adjust the spacing between subplots

plt.show(block=False)

# Convert to discrete-time using bilinear
Bz, Az = bilinear(Bs, As, Fs)  # Obtain H(z) from H(s) with given Fs
print(f"Bz = \n {Bz}")
print(f"Az = \n {Az}")

plt.figure(2)
w, h = freqz(Bz, Az)  # Compute the frequency response of H(z)
magnitude_dB = 20 * np.log10(np.abs(h))
phase_deg = np.unwrap(np.angle(h)) * 180 / np.pi
w_normalized = w / np.pi  # Normalize the frequency

plt.subplot(2, 1, 1)
plt.plot(w_normalized, magnitude_dB)  # Plot the magnitude response in dB
plt.xticks(np.linspace(0, 1, 11))
plt.yticks(np.arange(-80, 1, 20))
plt.xlim(0, 1)
plt.xlabel("Normalized Frequency (x pi rad/sample)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(w_normalized, phase_deg)  # Plot the phase response in degrees
plt.xticks(np.linspace(0, 1, 11))
plt.yticks(np.linspace(-200, 200, 5))
plt.xlim(0, 1)
plt.ylim(-270, 270)
plt.xlabel("Normalized Frequency (x pi rad/sample)")
plt.ylabel("Phase (degrees)")
plt.grid(True)

plt.subplots_adjust(hspace=0.5)  # Adjust the spacing between subplots

plt.show()

s = 1j * wp
Hs_at_wp = 20 * np.log10(np.abs(np.polyval(Bs, s) / np.polyval(As, s)))  # Cutoff
print(f"Hs_at_wp = \n {Hs_at_wp}")

z = np.exp(1j * Wp)
H_at_Wp = 20 * np.log10(np.abs(np.polyval(Bz, z) / np.polyval(Az, z)))
print(f"H_at_Wp = \n {H_at_Wp}")

z = np.exp(1j * Wr)
H_at_Wr = 20 * np.log10(np.abs(np.polyval(Bz, z) / np.polyval(Az, z)))
print(f"H_at_Wr = \n {H_at_Wr}")
