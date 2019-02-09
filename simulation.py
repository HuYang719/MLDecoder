import numpy as np
import Codes
import Channel
import MLDecoder
from time import time

def calerrors(dec_code, dec_true):
    errors = np.sum(np.not_equal(dec_code,dec_true), dtype=np.int32)
    num = dec_code.size
    return errors, num

def simML(k,N,alpha,GSNR):
    tG = Codes.encoding(k, N, [])
    data = Codes.genData(k, N, shuffle=False)
    codeword = Codes.genCodeword(k,N)
    revcodes, scale = Channel.ImpulChannel(k,N,data,alpha,GSNR)
    dec_ML_N = MLDecoder.MLDecoder(revcodes,k,N,codeword,alpha,scale)
    dec_ML = MLDecoder.DecMessage(dec_ML_N,tG,k)
    dec_true = MLDecoder.DecMessage(data,tG,k)
    errors, nums = calerrors(dec_ML, dec_true)
    # print('true_data:', data)
    # print("dec_true:", dec_true)
    # print("dec_ML_N", dec_ML_N,"dec_ML", dec_ML)
    return errors, nums





if __name__ == '__main__':
    k = 8
    N = 16
    alpha = 1.5
    GSNRs = np.linspace(0, 9, 15)
    t = time()
    sim_nums = 1
    for i in range(15):
        errors = 0
        nums = 0
        for j in range(sim_nums):
            error, num = simML(k, N, alpha, GSNRs[i])
            errors += error
            nums += num
        with open('./results.txt', 'a') as f:
            f.write('GSNRS:{},errors:{},nums:{},BER:{}\n'.format(GSNRs[i], errors, nums, errors/nums))
    t = time() - t
    print(errors, nums)
    print("Use time:", t)