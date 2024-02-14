import numpy as np
from scipy import signal

a = 1  # zeros in the s-plane
b, c, d = -2, -1 + 1j * 4, -0.1 + 1j * 20  # poles in the s-plane

Hs_num = np.poly([a])  # numerator of H_s(s)
Hs_den = np.poly([b, c, np.conj(c), d, np.conj(d)])  # H_s(s) denominator
print(f"Hs_den = \n {Hs_den}")

k = Hs_den[-1] / Hs_num[-1]  # calculate factor
Hs_num = k * Hs_num  # force a gain=1 at DC (s=0)
print(f"Hs_num = \n {Hs_num}")

Fs = 10  # sampling frequency (Hz)

# apply the bilinear transformation
Hz_num, Hz_den = signal.bilinear(Hs_num, Hs_den, fs=Fs)

print(f"Hz_num = \n {Hz_num}")

print(f"Hz_den = \n {Hz_den}")
