'''
define the code and generate the test codes
'''
import pyldpc
import numpy as np


def encoding(k=8,N=16,H=None):
    if (k == 2 and N == 4):
        H = [[0, 0, 0, 1],
             [0, 1, 0, 0]]

    elif(k==8 and N == 16):

        H = [[0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
             [0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]]

    elif(k==16 and N == 32):

        H = [[0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
             [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
             [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
             [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
             [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]]

    tG = pyldpc.CodingMatrix(H)
    print(tG)
    return tG


# Create all possible information words
def dec2bin(num,k):
    l = np.zeros((k), dtype='int64')
    i = k-1
    while True:
        num, remainder = divmod(num, 2)
        l[i] = int(remainder)
        i = i - 1
        if num == 0:
            return l

# promise all the words are possible equally
#  gen 2**k data
def genData(k,N,shuffle=True):
    tG = encoding(k, N, [])
    label = np.zeros((2**k,k),dtype=int)
    for s in range(0, 2**k):
        label[s] = dec2bin(s,k)

    # Create sets of all possible codewords (codebook)
    data = np.zeros((2**k, N), dtype=int)
    for i in range(0, 2**k):
        data[i] = (pyldpc.Coding(tG, label[i], 0) )  # no Noise! HY：修改了pyLDPC源文件，pyLDPC生产码字不加噪声，统一用noise_layers层加噪
    if shuffle == True:
        np.random.shuffle(data)
    return data

def genCodeword(k, N):
    tG = encoding(k, N, [])
    label = np.zeros((2 ** k, k), dtype=int)
    for s in range(0, 2 ** k):
        label[s] = dec2bin(s, k)

    # Create sets of all possible codewords (codebook)
    codeword = np.zeros((2 ** k, N), dtype=int)
    for i in range(0, 2 ** k):
        codeword[i] = (pyldpc.Coding(tG, label[i], 0))  # no Noise! HY：修改了pyLDPC源文件，pyLDPC生产码字不加噪声，统一用noise_layers层加噪
    return codeword

if __name__ == '__main__':
    k = 16
    N = 32
    tG = encoding(2, 4, [])
    #data = genData(k, N, shuffle=False)
    #print('Data:', data.shape, data[0:5])


