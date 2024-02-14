import numpy as np

N = 3000  # total number of signal samples
L = 4  # number of non-zero samples of h[n]
power_x = 600  # noise power in Watts
x = np.sqrt(power_x) * np.ones(N) / 2
h = np.ones(L)  # FIR moving average filter impulse response
y = np.convolve(h, x)  # filtered signal y[n] = x[n] * h[n]
H = np.fft.fft(h, 4 * N)  # DTFT (sampled) of the impulse response
Sy_th = power_x * np.abs(H) ** 2  # theoretical PSD via sampling DTFT
M = 256  # maximum lag chosen as M < N
Ry = np.correlate(y, y, mode="full")[len(y) - M - 1 : len(y) + M] / len(y)
Sy_corr = np.abs(np.fft.fft(Ry))  # PSD estimate via autocorrelation
