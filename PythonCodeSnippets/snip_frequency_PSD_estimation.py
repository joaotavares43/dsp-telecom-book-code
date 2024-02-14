import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz, welch, dlti, dimpulse
from spectrum import aryule

## generate some H(z) - case 1 - AR model (IIR)
# defining the poles as complex conjugates
p1 = 0.5
p2 = 0.3 + 0.2j
p3 = 0.3 - 0.2j
p4 = 0.9 * np.exp(1j * 2)
p5 = 0.9 * np.exp(-1j * 2)

# find H(z) denominator
Asystem = np.poly([p1, p2, p3, p4, p5])
# get rid of numerical errors
Asystem = np.real(Asystem)
# H(z) numerator given that H(z) is all-poles
Bsystem = np.array([1])
# H(z) impulse response
_, y_out = dimpulse(dlti(Bsystem, Asystem))
h = np.squeeze(y_out)

## generate x[n] and y[n]
# sampling frequency
Fs = 8000
# power in Watts for the random signal y[n]
Py_desired = 3
# number of samples for y[n]
S = 100000

# energy of the systems impulse response
Eh = np.sum(h**2)
print(f"Eh = \n {Eh}")

# white Gaussian with given power
x = np.sqrt(Py_desired / Eh) * np.random.randn(S)
# finally, generate y[n]
y = lfilter(Bsystem, Asystem, x)
# get power, to check if close to Py_desired/Eh
Px = np.mean(x**2)
print(f"Px = \n {Px}")
# get power, to check if close to Py_desired
Py = np.mean(y**2)
print(f"Py = \n {Py}")

## LPC analysis for estimating the PSD of y[n]
# assume we know the correct order of A(z) (matched condition)
P = 5
# solves Yule-Walker to estimate H(z)
# note that Perror is approximately Py_desired/Eh, the power of x
A, Perror, _ = aryule(y, P)
A = np.concatenate(([1], A))

# value for the bilateral PSD Sx(w)
N0over2 = Perror / Fs
# assumes a unilateral PSD Sy(w)=N0/|A(z)|^2
N0 = 2 * N0over2
# number of FFT points for all spectra
Nfft = 1024
# frequency response H(w) from 1/A(z)
f, Hw = freqz([1], A, Nfft, fs=Fs)
# unilateral PSD estimated via AR modeling
Sy = N0 * (np.abs(Hw) ** 2)
# Welchs PSD
f2, Swelch = welch(y, window="hamming", nperseg=Nfft, noverlap=0, nfft=Nfft, fs=Fs)
# DTFT of assumed system
f3, Hsystem = freqz(Bsystem, Asystem, Nfft, fs=Fs)
# theoretical PSD
Sy_theoretical = (Px / (Fs / 2)) * (np.abs(Hsystem) ** 2)

# compare PSD from Welch, AR and theoretical
plt.plot(f2, 10 * np.log10(Swelch), label="Welch")
plt.plot(f3, 10 * np.log10(Sy_theoretical), label="Theoretical")
plt.plot(f, 10 * np.log10(Sy), label="AR")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (dB/Hz)")
plt.legend()
plt.show()
