import numpy as np
import pyldpc
import Codes
from time import time
from scipy.stats import levy_stable


# for single word, can we find it?
def MLSinCodeWord(revcode,k,N,codeword,alpha,scale):
    res = revcode
    maxlog = -float('inf')
    for i in range(2**k):
        codei = codeword[i]
        t = time()
        sumpdf = np.sum(levy_stable.logpdf(revcode-codei,alpha,0,0,scale))
        #print('one codeword:',time()-t)
        maxlog = max(maxlog, sumpdf)
        if(maxlog == sumpdf):
            res = codei
    return res, maxlog

# return the soft Maximum Likelihood decision
def MLDecoder(revcodes,k,N,codeword,alpha,scale):
    res = revcodes
    for i in range(2**k):
        resi, _ = MLSinCodeWord(revcodes[i],k,N,codeword,alpha,scale)
        res[i] = resi
        #print("Finish:", i, "total:", 2**k)
    return res

def DecMessage(Dec_Codes,tG,k):
    Dec_Mes = np.zeros((2**k, k))
    for i in range(2**k):
        datai = ((-Dec_Codes[i])+1) / 2
        Dec_Mes[i] = pyldpc.DecodedMessage(tG, datai)

    return Dec_Mes

if __name__ == '__main__':
    revcode = [1, -1]
    codeword = [[-1,1],[1,-1]]
    codei, maxlog = MLSinCodeWord(revcode,1,2,codeword,1.8,0.6)
    #print(codei,maxlog)