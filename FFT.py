import numpy as np
import cmath
from cmath import sin, cos, pi
from numfi import numfi


def _reverse_pos(num, stage_N):
    out = 0
    bits = 0
    _i = stage_N
    data = num
    while (_i != 0):
        _i = _i // 2
        bits += 1
    for i in range(bits - 1):
        out = out << 1
        out |= (data >> i) & 1
    return out
  
class FFT():
    def __init__(self , N = 0):
        self.N = N
        self.stage_num = int(cmath.log(N, 2).real)                                             # FFT stage num
        self.wn = np.array([(cos(2 * pi / N) - sin(2 * pi / N) * 1j) ** i for i in range(N)])  # FFT twiddle factors
        self.wn = np.expand_dims(self.wn,1)
        self._index_gen()                                                                      # FFT stages butterfly index generator
 
    def _index_gen(self):
        self.p_stage_index = []
        self.q_stage_index = []
        self.w_stage_index = []
        self.w_stage_iter = []
        for i in range(self.stage_num):
            self.group_num = 2 ** i
            self.group_iter_num = self.N // self.group_num
            self.half_group_iter_num = (self.group_iter_num) // 2
            p_stage_index_tmp_0 = []
            q_stage_index_tmp_0 = []
            w_stage_index_tmp_0 = []
            w_stage_iter_tmp_0 = []
            for g in range(self.group_num):
                for x in range(self.half_group_iter_num):
                    p_stage_index_tmp_0.append(x + self.group_iter_num * g)
                    q_stage_index_tmp_0.append(x + self.group_iter_num * g + self.half_group_iter_num)
                    w_stage_index_tmp_0.append(_reverse_pos(g, 2 ** i) << (self.stage_num - 1 - i))
                    w_stage_iter_tmp_0.append(self.wn[_reverse_pos(g, 2 ** i) << (self.stage_num - 1 - i)])
            self.p_stage_index.append(p_stage_index_tmp_0)
            self.q_stage_index.append(q_stage_index_tmp_0)
            self.w_stage_index.append(w_stage_index_tmp_0)
            self.w_stage_iter.append(w_stage_iter_tmp_0)

        self.final_reverse = np.array([_reverse_pos(x, self.N) for x in range(self.N)])   # final reverse code
        # print(self.p_stage_index)
        # print(self.q_stage_index)
        # print(self.w_stage_index)
        self.p_stage_index = np.array(self.p_stage_index)   # faster than list
        self.q_stage_index = np.array(self.q_stage_index)
        self.w_stage_index = np.array(self.w_stage_index)

    def _butterfly(self, xp, xq, wn):
        xw = xq * wn
        yp = xp + xw
        yq = xp - xw
        return yp, yq

    def cal_fft(self, data, rfft = False):
        raw_data = np.copy(data)
        if(data.ndim == 1):
            raw_data = np.expand_dims(raw_data,1)
        for i in range(self.stage_num):
            xp = raw_data[self.p_stage_index[i]]
            xq = raw_data[self.q_stage_index[i]]
            w = self.wn[self.w_stage_index[i]]
            # w = w_stage_iter[i]
            raw_data[self.p_stage_index[i]], raw_data[self.q_stage_index[i]] = self._butterfly(xp, xq, w)
            # stages.append(data[final_reverse])
        raw_data = raw_data[self.final_reverse]
        if(data.ndim == 1):
            raw_data = np.squeeze(raw_data)
        if(rfft):
            raw_data = raw_data[:int(data.shape[0]/2)+1]
        return raw_data

    def cal_fft_with_stage(self, data):
        stages = []
        raw_data = np.copy(data)
        if(data.ndim == 1):
            raw_data = np.expand_dims(raw_data,1)
        for i in range(self.stage_num):
            xp = raw_data[self.p_stage_index[i]]
            xq = raw_data[self.q_stage_index[i]]
            w = self.wn[self.w_stage_index[i]]
            # w = w_stage_iter[i]
            raw_data[self.p_stage_index[i]], raw_data[self.q_stage_index[i]] = self._butterfly(xp, xq, w)
            stages.append(np.copy(raw_data))
        raw_data = raw_data[self.final_reverse]
        if(data.ndim == 1):
            raw_data = np.squeeze(raw_data)
        return raw_data,np.array(stages)

class qFFT():
    def __init__(self , N = 0, qlist = None, wq = (1,16,15)):
        self.N = N
        self.qlist = qlist
        self.wq = wq
        self.stage_num = np.log2(self.N).astype(np.int32)                                             # FFT stage num
        wn = np.array([(cos(2 * pi / N) - sin(2 * pi / N) * 1j) ** i for i in range(int(N/2))])       # FFT twiddle factors
        self.wn_real = np.array(numfi(wn.real,wq[0],wq[1],wq[2]))
        self.wn_real = np.expand_dims(self.wn_real, 1)
        self.wn_imag = np.array(numfi(wn.imag,wq[0],wq[1],wq[2]))
        self.wn_imag = np.expand_dims(self.wn_imag, 1)
        self.wn_complex = self.wn_real + self.wn_imag * 1j
        self._index_gen()                                                                      # FFT stages butterfly index generator

    def _index_gen(self):
        self.p_stage_index = []
        self.q_stage_index = []
        self.w_stage_index = []
        self.w_stage_iter = []
        for i in range(self.stage_num):
            self.group_num = 2 ** i
            self.group_iter_num = self.N // self.group_num
            self.half_group_iter_num = (self.group_iter_num) // 2
            p_stage_index_tmp_0 = []
            q_stage_index_tmp_0 = []
            w_stage_index_tmp_0 = []
            w_stage_iter_tmp_0 = []
            for g in range(self.group_num):
                for x in range(self.half_group_iter_num):
                    p_stage_index_tmp_0.append(x + self.group_iter_num * g)
                    q_stage_index_tmp_0.append(x + self.group_iter_num * g + self.half_group_iter_num)
                    w_stage_index_tmp_0.append(self._reverse_pos(g, 2 ** i) << (self.stage_num - 1 - i))
                    #w_stage_iter_tmp_0.append(self.wn[self._reverse_pos(g, 2 ** i) << (self.stage_num - 1 - i)])
            self.p_stage_index.append(p_stage_index_tmp_0)
            self.q_stage_index.append(q_stage_index_tmp_0)
            self.w_stage_index.append(w_stage_index_tmp_0)
            #self.w_stage_iter.append(w_stage_iter_tmp_0)

        self.final_reverse = np.array([self._reverse_pos(x, self.N) for x in range(self.N)])   # final reverse code
        # print(self.p_stage_index)
        # print(self.q_stage_index)
        # print(self.w_stage_index)
        self.p_stage_index = np.array(self.p_stage_index)   # faster than list
        self.q_stage_index = np.array(self.q_stage_index)
        self.w_stage_index = np.array(self.w_stage_index)

    def _reverse_pos(self , num, stage_N):
        out = 0
        bits = 0
        _i = stage_N
        data = num
        while (_i != 0):
            _i = _i // 2
            bits += 1
        for i in range(bits - 1):
            out = out << 1
            out |= (data >> i) & 1
        return out

    def _butterfly(self, xp, xq, wn):
        xw = xq * wn
        yp = xp + xw
        yq = xp - xw
        # ac = xq.real * wn.real
        # bd = xq.imag * wn.imag
        # ad = xq.real * wn.imag
        # bc = xq.imag * wn.real
        # xw_real = ac - bd
        # xw_imag = ad + bc
        # yp_real = xp.real + xw_real
        # yp_imag = xp.imag + xw_imag
        # yq_real = xp.real - xw_real
        # yq_imag = xp.imag - xw_imag
        return yp, yq
        #return yp_real + yp_imag*1j, yq_real + yq_imag*1j


    def _butterfly_golden(self , xp_real, xp_imag, xq_real, xq_imag, wn_real, wn_imag):
        ac = xq_real * wn_real
        bd = xq_imag * wn_imag
        ad = xq_real * wn_imag
        bc = xq_imag * wn_real
        xw_real = ac - bd
        xw_imag = ad + bc
        yp_real = xp_real + xw_real
        yp_imag = xp_imag + xw_imag
        yq_real = xp_real - xw_real
        yq_imag = xp_imag - xw_imag
        return yp_real, yp_imag, yq_real, yq_imag

    def cal_fft_with_stage(self, data, rfft = False):
        stages = []
        raw_data = np.copy(data)
        if(data.ndim == 1):
            raw_data = np.expand_dims(raw_data,1)
        raw_data_real = np.array(numfi(raw_data.real,self.qlist[0][0],(self.qlist[0][1]),(self.qlist[0][2])))
        raw_data_imag = np.array(numfi(raw_data.imag,self.qlist[0][0],(self.qlist[0][1]),(self.qlist[0][2])))
        for i in range(self.stage_num):
            xp_real = raw_data_real[self.p_stage_index[i]]
            xp_imag = raw_data_imag[self.p_stage_index[i]]
            xq_real = raw_data_real[self.q_stage_index[i]]
            xq_imag = raw_data_imag[self.q_stage_index[i]]
            w_real = self.wn_real[self.w_stage_index[i]]
            w_imag = self.wn_imag[self.w_stage_index[i]]
            raw_data_real[self.p_stage_index[i]],raw_data_imag[self.p_stage_index[i]],\
            raw_data_real[self.q_stage_index[i]],raw_data_imag[self.q_stage_index[i]] = \
                self._butterfly_golden(xp_real, xp_imag, xq_real, xq_imag, w_real, w_imag)

            raw_data_real = np.array(numfi(raw_data_real, (self.qlist[i+1][0]),(self.qlist[i+1][1]),(self.qlist[i+1][2]) ))
            raw_data_imag = np.array(numfi(raw_data_imag, (self.qlist[i+1][0]),(self.qlist[i+1][1]),(self.qlist[i+1][2]) ))
            if (data.ndim == 1):
                stages.append(np.array([np.squeeze(np.copy(raw_data_real)), np.squeeze(np.copy(raw_data_imag))]))
            else :
                stages.append(np.array([np.copy(raw_data_real),np.copy(raw_data_imag)]))
        raw_data_real = raw_data_real[self.final_reverse]
        raw_data_imag = raw_data_imag[self.final_reverse]
        if(data.ndim == 1):
            raw_data_real = np.squeeze(raw_data_real)
            raw_data_imag = np.squeeze(raw_data_imag)

        if(rfft):
            raw_data_real = raw_data_real[:int(data.shape[0]/2)+1]
            raw_data_imag = raw_data_imag[:int(data.shape[0]/2)+1]

        #return raw_data_real + raw_data_imag * 1j
        return np.array([raw_data_real, raw_data_imag]), np.array(stages)

    def cal_fft(self, data, rfft=False):
        raw_data = np.copy(data)
        if (data.ndim == 1):
            raw_data = np.expand_dims(raw_data, 1)
        raw_data_real = np.array(numfi(raw_data.real, self.qlist[0][0], (self.qlist[0][1]), (self.qlist[0][2])))
        raw_data_imag = np.array(numfi(raw_data.imag, self.qlist[0][0], (self.qlist[0][1]), (self.qlist[0][2])))
        raw_data = raw_data_real + raw_data_imag * 1j
        for i in range(self.stage_num):
            xp = raw_data[self.p_stage_index[i]]
            xq = raw_data[self.q_stage_index[i]]
            w = self.wn_complex[self.w_stage_index[i]]
            raw_data[self.p_stage_index[i]], raw_data[self.q_stage_index[i]] = self._butterfly(xp, xq, w)
            raw_data_real = np.array(
                numfi(raw_data.real, (self.qlist[i + 1][0]), (self.qlist[i + 1][1]), (self.qlist[i + 1][2])))
            raw_data_imag = np.array(
                numfi(raw_data.imag, (self.qlist[i + 1][0]), (self.qlist[i + 1][1]), (self.qlist[i + 1][2])))
            raw_data = raw_data_real + raw_data_imag * 1j

        raw_data_real = raw_data_real[self.final_reverse]
        raw_data_imag = raw_data_imag[self.final_reverse]
        if (data.ndim == 1):
            raw_data_real = np.squeeze(raw_data_real)
            raw_data_imag = np.squeeze(raw_data_imag)

        qfft_out_all = raw_data_real + raw_data_imag * 1j
        if (rfft):
            qfft_out_all = qfft_out_all[:int(data.shape[0] / 2) + 1]
        return qfft_out_all

    # def cal_fft(self, data, rfft = False):
    #     raw_data = np.copy(data)
    #     if(data.ndim == 1):
    #         raw_data = np.expand_dims(raw_data,1)
    #     raw_data_real = np.array(numfi(raw_data.real,self.qlist[0][0],(self.qlist[0][1]),(self.qlist[0][2])))
    #     raw_data_imag = np.array(numfi(raw_data.imag,self.qlist[0][0],(self.qlist[0][1]),(self.qlist[0][2])))
    #     for i in range(self.stage_num):
    #         xp_real = raw_data_real[self.p_stage_index[i]]
    #         xp_imag = raw_data_imag[self.p_stage_index[i]]
    #         xq_real = raw_data_real[self.q_stage_index[i]]
    #         xq_imag = raw_data_imag[self.q_stage_index[i]]
    #         w_real = self.wn_real[self.w_stage_index[i]]
    #         w_imag = self.wn_imag[self.w_stage_index[i]]
    #         raw_data_real[self.p_stage_index[i]],raw_data_imag[self.p_stage_index[i]],\
    #         raw_data_real[self.q_stage_index[i]],raw_data_imag[self.q_stage_index[i]] = \
    #             self._butterfly_golden(xp_real, xp_imag, xq_real, xq_imag, w_real, w_imag)
    #
    #         raw_data_real = np.array(numfi(raw_data_real, (self.qlist[i+1][0]),(self.qlist[i+1][1]),(self.qlist[i+1][2]) ))
    #         raw_data_imag = np.array(numfi(raw_data_imag, (self.qlist[i+1][0]),(self.qlist[i+1][1]),(self.qlist[i+1][2]) ))
    #
    #     raw_data_real = raw_data_real[self.final_reverse]
    #     raw_data_imag = raw_data_imag[self.final_reverse]
    #     if(data.ndim == 1):
    #         raw_data_real = np.squeeze(raw_data_real)
    #         raw_data_imag = np.squeeze(raw_data_imag)
    #
    #     qfft_out_all = raw_data_real + raw_data_imag * 1j
    #     if(rfft):
    #         qfft_out_all = qfft_out_all[:int(data.shape[0]/2)+1]
    #     return qfft_out_all

def figure_show_difference(true_fft , my_fft_out):
    dist = true_fft - my_fft_out
    plt.figure(1)
    plt.hist2d(dist.real, dist.imag)
    plt.figure(2)
    plt.plot(my_fft_out.real,'g')
    plt.figure(3)
    plt.plot(np.abs(true_fft.real-my_fft_out.real), 'r')
    plt.show()

def figure_show_stage_dist(stages):
    fig, ax = plt.subplots(1, stages.shape[0])
    for i,s in enumerate(stages):
        ax[i].hist2d(s.real, s.imag)
    plt.show()

def cal_stages_max_value(stages):
    max_value = np.zeros(stages.shape[0])
    for i,s in enumerate(stages):
        max_value[i] = (np.max([np.max(s.real), np.max(s.imag)]))
    return max_value

def cal_diff_sum(true_fft, my_fft):
    return (np.sum(np.abs((true_fft - my_fft).real)) + np.sum(np.abs((true_fft - my_fft).imag)) * 1j)

def cal_diff_more_sample_avg(iter_num= 1000, fft_cmp=None, fft_my=None, input_w =16,input_f=15):
    fft_error_value = 0+0j
    for i in range(iter_num):
        data = np.random.uniform(low=-1.0, high=1.0, size=N)
        data = numfi(data, 1, input_w, input_f)
        data = np.array(data)
        data = np.zeros(N, dtype=complex) + data
        true_fft = fft_cmp(data)
        qfft_out, stages = fft_my(data)
        qfft_out_all = qfft_out[0, :] + qfft_out[1, :] * 1j
        diff_num = cal_diff_sum(true_fft, qfft_out_all)
        fft_error_value += diff_num

    avg_fft_error_value = fft_error_value / iter_num
    return avg_fft_error_value

def cal_stage_diff_more_sample_avg(iter_num= 1000, N = 256, fft_cmp=None, fft_my=None, input_w =16,input_f=15):
    fft_error_value = np.array([0+0j for x in range(int(np.log2(N)))])
    for i in range(iter_num):
        data = np.random.uniform(low=-1.0, high=1.0, size=N)
        data = numfi(data, 1, input_w, input_f)
        data = np.array(data)
        data = np.zeros(N, dtype=complex) + data
        true_fft, true_stages = fft_cmp(data)
        qfft_out, my_stages = fft_my(data)
        qfft_stages_all = my_stages[:, 0, :] + my_stages[:, 1, :] * 1j
        diff_num = (np.sum(np.abs((true_stages - qfft_stages_all).real), axis=1) + np.sum(np.abs((true_stages - qfft_stages_all).imag), axis=1) * 1j)
        #print(diff_num.shape)
        #raise('K')
        fft_error_value += diff_num

    avg_fft_error_value = fft_error_value / iter_num
    return avg_fft_error_value

def plot_diff_of_two_fft(plt,true_fft,my_fft):
    plt.figure()
    plt.plot(true_fft,'r')
    plt.plot(my_fft,'g')
    plt.show()

def check_scala_np_wn(real_path, imag_path, qlist, N=256,width=16):
    my_q_fft = qFFT(N, qlist)
    my_wn_real = (np.array(np.squeeze(my_q_fft.wn_real))*(2 ** (width-1))).astype(np.int32)
    my_wn_imag = (np.array(np.squeeze(my_q_fft.wn_imag))*(2 ** (width-1))).astype(np.int32)
    with open(real_path,'r') as f:
        scala_wn_real = f.readlines()
    with open(imag_path, 'r') as f:
        scala_wn_imag = f.readlines()
    scala_wn_real = np.array(scala_wn_real).astype(np.int32)
    scala_wn_imag = np.array(scala_wn_imag).astype(np.int32)
    if np.sum(scala_wn_real == my_wn_real) != scala_wn_real.shape :
        raise(scala_wn_real == my_wn_real)
    if np.sum(scala_wn_imag == my_wn_imag) != scala_wn_real.shape :
        raise(scala_wn_imag == my_wn_imag)
    print('scala wn equals np wn!')

if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    print('************** FFT test ****************')
    N = 256
    bits = 16

    data = np.random.uniform(low=-1.0, high=1.0, size=N)
    data = np.zeros(N, dtype=complex) + data
   # print(data)
    np_fft_time = time.time()
    true_fft = np.fft.fft(data)
    print('Execute time of Numpy.fft.fft = ',(time.time() - np_fft_time) * 100000)
    my_fft = FFT(N)
    my_start_time = time.time()
    my_fft_out = my_fft.cal_fft(data)
    my_finish_time = time.time()
    print('Execute time of my fft        = ',(my_finish_time - my_start_time) * 100000)
    print('Error of np.fft and my fft : ', sum(true_fft - my_fft_out))
    #figure_show_difference(true_fft , my_fft_out)

    iter_num = 1000
    max_value = np.zeros(int(np.log2(N)))
    for i in range(iter_num):
        data = np.random.uniform(low=-1.0, high=1.0, size=N)
        data = np.zeros(N, dtype=complex) + data
        my_fft_out, stages = my_fft.cal_fft_with_stage(data)
        max_value += cal_stages_max_value(stages)
    avg_max = max_value / iter_num
    print(avg_max)

    ##############  test quantized FFT  ################
    #max_bits = np.ceil(np.log2(avg_max)).astype(np.int32)
    #max_bits = max_bits + np.ones(int(np.log2(N)))*1
    #max_bits = np.array([4,4,4,4,4,4,4,4])
    max_bits = np.array([5,5,5,5,5,5,5,5])

    print(max_bits)
    round_bits = 15 - max_bits
    round_list = [[1,16,10]] # input round
    round_list += [[1,16,int(x)] for x in round_bits]
    qlist = round_list
    print(qlist)
    my_q_fft = qFFT(N, qlist)

    data = np.random.uniform(low=-1.0, high=1.0, size=N)
    data = numfi(data,1,16,15)
    data = np.array(data)
    data = np.zeros(N, dtype=complex) + data

    my = np.random.uniform(low=-1.0, high=1.0, size=N)
    a = my.copy()
    data_test = numfi(a,1,16,15)
    data_test = numfi(data_test,1,16,10)
    print(np.array([data_test[x].bin[0] for x in range(data_test.shape[0])]))
    data_test = np.array(data_test)
    #print((data_test * (2**15)).astype(np.int32))
    my_data = numfi(my,1,16,15)
    print(np.array([my_data[x].bin[0] for x in range(my_data.shape[0])]))
    my_data = np.array(my_data)
    #print((my_data * (2**15)).astype(np.int32))

    print(qlist[0][1])
    start_time = time.time()
    true_fft = np.fft.fft(data)
    print('Execute time of Numpy.fft.fft = ',(time.time() - start_time) * 100000)
    start_time = time.time()
    my_fft_out,true_stages = my_fft.cal_fft_with_stage(data)
    print('Execute time of my fft        = ',(time.time() - start_time) * 100000)

    start_time = time.time()
    qout = my_q_fft.cal_fft(data,rfft=False)
    print('Execute time of quantize fft  = ',(time.time() - start_time) * 100000)
    qfft_out, my_stages = my_q_fft.cal_fft_with_stage(data)

    qfft_out_all = qfft_out[0,:] + qfft_out[1,:]*1j
    print('diff between np fft and my fft: '+ str(cal_diff_sum(true_fft,my_fft_out)))
    print('diff between np fft and my fft quantize:'+ str(cal_diff_sum(true_fft,qfft_out_all)))

    plot_diff_of_two_fft(plt, true_fft,qfft_out_all)
    plt.show()
    # = cal_diff_more_sample_avg(iter_num=1000, fft_cmp=np.fft.fft, fft_my=my_q_fft.cal_fft_with_stage, input_w=16, input_f=15)
    #print(avg_fft_error_value)
    avg_fft_error_value = cal_stage_diff_more_sample_avg(iter_num=1000, N=N, fft_cmp=my_fft.cal_fft_with_stage, fft_my=my_q_fft.cal_fft_with_stage, input_w=16, input_f=15)
    print(avg_fft_error_value)
    # plt.plot(avg_fft_error_value)
    # plt.show()
    min_error = np.PINF
    min_bits = np.zeros(int(np.log2(N)))
    # print(my_q_fft.wn_real)
    # print(my_q_fft.wn_imag)
    # print(my_q_fft.wn_real.hex)
    check_scala_np_wn("../wn_real.txt", "../wn_imag.txt",qlist, 256, 16)
    # for i in range(6):
    #     for j in range(6):
    #         for k in range(6):
    #             for m in range(6):
    #                 for l in range(6):
    #                     print(i,j,k,m,l)
    #                     max_bits = np.array([i,j,k,m,l,6,6,6])
    #                     #print(max_bits)
    #                     round_bits = 15 - max_bits
    #                     round_list = [[1,16,15]] # input round
    #                     round_list += [[1,16,int(x)] for x in round_bits]
    #                     qlist = round_list
    #                     #print(qlist)
    #                     my_q_fft = qFFT(N, qlist)
    #                     avg_fft_error_value = cal_diff_more_sample_avg(iter_num=100, fft_cmp=np.fft.fft, fft_my=my_q_fft.cal_fft_with_stage,
    #                                              input_w=16, input_f=15)
    #                     avg_fft_error_value = avg_fft_error_value.real + avg_fft_error_value.imag
    #                     if avg_fft_error_value < min_error:
    #                         min_error = avg_fft_error_value
    #                         min_bits = max_bits
    #
    # print(min_error,min_bits)