import numpy as np
import pyldpc
import Codes
from time import time
from scipy.stats import levy_stable
np.seterr(divide = 'ignore')


# for single word, can we find it?
N = 32
indexnon0 = 10**-10
list1 = np.zeros((N), dtype=int) + 1
list_1 = np.zeros((N), dtype=int) - 1
def MLSinCodeWord(revcode,k,N,codeword,alpha,scale):
    res = revcode
    maxlog = -float('inf')
    #print("list1:",list1, "list_1", list_1)
    val1 = levy_stable.logpdf(revcode-list1+indexnon0, alpha, 0, 0, scale)
    val_1 = levy_stable.logpdf(revcode-list_1+indexnon0, alpha, 0, 0, scale)
    codeword1 = (codeword+1)/2
    codeword_1 = (codeword-1)/(-2)
    # val1.shape = (N, 1)
    # val_1.shape = (N, 1)
    val1 = np.transpose(val1)
    val_1 = np.transpose(val_1)
    res1 = np.dot(codeword1, val1)
    res2 = np.dot(codeword_1, val_1)
    res3 = res1 + res2
    # print("cw1",codeword1,"cw_1",codeword_1)
    maxpos = np.where(res3 == np.amax(res3))
    # print("maxpos",maxpos)
    res = codeword[maxpos[0]]
    # ########################
    # ## 硬判决
    # res = np.zeros(N, dtype=int)
    # for j in range(N):
    #     if val1[j] > val_1[j]:
    #         res[j] = 1
    #     else:
    #         res[j] = -1
    # return res
    # ########################

    # for i in range(2**k):
    #     sumpdf = 0
    #     codei = codeword[i]
    #     t = time()
    #     for j in range(N):
    #         if codei[j] == 1:
    #             sumpdf += val1[j]
    #         else:
    #             sumpdf += val_1[j]
    #
    #     #sumpdf = np.sum(levy_stable.logpdf(revcode-codei+10**-10, alpha, 0, 0, scale))
    #     #print('one codeword:',time()-t)
    #     maxlog = max(maxlog, sumpdf)
    #     if(maxlog == sumpdf):
    #         res = codei
    return res

# return the soft Maximum Likelihood decision
def MLDecoder(revcodes,k,N,codeword,alpha,scale):
    res = revcodes
    for i in range(2**k):
        t = time()
        resi = MLSinCodeWord(revcodes[i],k,N,codeword,alpha,scale)
        res[i] = resi
        print("Finish:", i, "total:", 2**k, "time use:", time() - t)
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