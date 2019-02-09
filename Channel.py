import numpy as np
from scipy.stats import levy_stable

def AWGN(k,N,sedcodes,SNRs):
    for SNRi in SNRs:
        Es = SNRi + 10*np.log10(k/N)
        sigma = np.sqrt(1/(2*10**(Es/10)))
        for i in range(2**k):
            sedcodes[i] = sedcodes[i] + sigma*np.random.standard_normal(sedcodes[0].shape)
    return sedcodes

def calscale(GSNR,alpha,R):
    GSNR=10**(GSNR/10)       # Eb/No conversion from dB to decimal
    S0=1/(np.sqrt(7.12*GSNR*R))        #3.56
    gamma=((1.78*S0)**alpha)/1.78
    scale=gamma**(1/alpha)
    return scale

def ImpulChannel(k,N,sedcodes,alpha,GSNR):
    scale = calscale(GSNR, alpha, k/N)
    revcodes = sedcodes + levy_stable.rvs(alpha, 0, 0, scale, (2**k, N))
    return revcodes, scale


