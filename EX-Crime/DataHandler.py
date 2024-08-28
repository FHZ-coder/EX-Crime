import pickle
import numpy as np
from Params import args
import matplotlib.pyplot as plt

class DataHandler:
    def __init__(self):
        if args.data == 'NYC':
            predir = 'Datasets/NYC_crime/'
        elif args.data == 'CHI':
            predir = 'Datasets/CHI_crime/'
        else:
            predir = None
        self.predir = predir
        with open(predir + 'trn.pkl', 'rb') as fs:
            trnT = pickle.load(fs)
        with open(predir + 'val.pkl', 'rb') as fs:
            valT = pickle.load(fs)
        with open(predir + 'tst.pkl', 'rb') as fs:
            tstT = pickle.load(fs)
        print("self.predir", self.predir)
        args.row, args.col, _, args.offNum = trnT.shape
        args.areaNum = args.row * args.col
        args.trnDays = trnT.shape[2]
        args.valDays = valT.shape[2]
        args.tstDays = tstT.shape[2]
        args.decay_step = args.trnDays//args.batch

        self.trnT = np.reshape(trnT, [args.areaNum, -1, args.offNum])
        self.valT = np.reshape(valT, [args.areaNum, -1, args.offNum])
        self.tstT = np.reshape(tstT, [args.areaNum, -1, args.offNum])
        self.mean = np.mean(trnT)
        self.std = np.std(trnT)
        self.mask1, self.mask2, self.mask3, self.mask4 = self.getSparsity()
        self.getTestAreas()
        self.construct_interval_Graph()
        #self.FFT_cycle()
        print('Row:', args.row, ', Col:', args.col)
        print('Sparsity:', np.sum(trnT!=0) / np.reshape(trnT, [-1]).shape[0])

    def zScore(self, data):
        return (data - self.mean) / self.std

    def zInverse(self, data):
        return data * self.std + self.mean

    def getSparsity(self):
        data = self.tstT
        day = data.shape[1]
        mask = 1 * (data > 0)
        p1 = np.zeros([data.shape[0], data.shape[2]])
        for cate in range(4):
            for region in range(mask.shape[0]):
                p1[region, cate] = np.sum(mask[region, :, cate], axis=0) / day
        mask1 = np.zeros_like(p1)
        mask2 = np.zeros_like(p1)
        mask3 = np.zeros_like(p1)
        mask4 = np.zeros_like(p1)
        for cate1 in range(4):
            for region1 in range(mask.shape[0]):
                if p1[region1, cate1] > 0 and p1[region1, cate1] <= 0.25:
                    mask1[region1, cate1] = 1
                elif p1[region1, cate1] > 0.25 and p1[region1, cate1] <= 0.5:
                    mask2[region1, cate1] = 1
                elif p1[region1, cate1] > 0.5 and p1[region1, cate1] <= 0.75:
                    mask3[region1, cate1] = 1
                elif p1[region1, cate1] > 0.75 and p1[region1, cate1] <= 1:
                    mask4[region1, cate1] = 1
        return mask1, mask2, mask3, mask4

    def getTestAreas(self):
        posTimes = np.sum(1 * (self.trnT!=0), axis=1)
        percent = posTimes / args.trnDays
        self.tstLocs = (percent > -1) * 1
    @classmethod
    def idEncode(cls, x, y):
        return x * args.col + y
    

    def FFT_cycle(self):  
        all_time_crime=np.concatenate((self.trnT, self.valT, self.tstT),axis=1)
        all_time_crime=np.sum(all_time_crime,axis=0)  #[days,crimetype] in it is  crime number
        N=all_time_crime.shape[0]
        t=np.linspace(1, N+1, N)
        cycle_list=[]
        for i in range(args.offNum):
            crime_interval_dict={}
            signal=all_time_crime[:,i]
            X_fft = np.fft.fft(signal)
            freq = np.fft.fftfreq(N, d=t[1]-t[0])
            pos_mask = np.where(freq > 0)
            freqs = freq[pos_mask]
            masked_signals = signal[pos_mask]

            interval_top_k = 5
            top_k_idxs = np.argpartition(masked_signals, -interval_top_k)[-interval_top_k:]
            top_k_amplitude = masked_signals[top_k_idxs]
            fft_periods = (1 / freqs[top_k_idxs]).astype(int)
            fft_periods=np.append(fft_periods,7)   
            fft_periods=np.append(fft_periods,14)
            fft_periods=np.append(fft_periods,21)
            fft_periods=np.append(fft_periods,28)
            for lag in fft_periods:
                acf_score = self.autocorrelation_lags(signal, lag)
                crime_interval_dict[lag]=acf_score
            top_two_keys = sorted(crime_interval_dict, key=crime_interval_dict.get, reverse=True)[:2]
            max_key = max(crime_interval_dict, key=crime_interval_dict.get)
            cycle_list.append(max_key)
        return cycle_list
            
    def autocorrelation(self,x,lags):
        n = len(x)
        x = np.array(x)
        result = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]\
            /(x[i:].std()*x[:n-i].std()*(n-i)) for i in range(1,lags+1)]
        return result
    def autocorrelation_lags(self,x,lags):
        n = len(x)
        x = np.array(x)
        result = np.correlate(x[lags:]-x[lags:].mean(),x[:n-lags]-x[:n-lags].mean())[0]\
            /(x[lags:].std()*x[:n-lags].std()*(n-lags)) 
        return result

    def construct_interval_Graph(self):
        crime_cycle=self.FFT_cycle()
        self.adj_matrix=np.zeros((args.areaNum,args.areaNum))
        mx = [-1, 0, 1, 0, -1, -1, 1, 1, 0]
        my = [0, -1, 0, 1, -1, 1, -1, 1, 0]
        def illegal(x, y):
            return x < 0 or y < 0 or x >= args.row or y >= args.col
        edges = list()
        for i in range(args.row): 
            for j in range(args.col):
                n1 = self.idEncode(i, j)  
                for k in range(len(mx)):
                    temx = i + mx[k]
                    temy = j + my[k]
                    if illegal(temx, temy):
                        continue
                    n2 = self.idEncode(temx, temy)
                    edges.append([n1, n2])
                    self.adj_matrix[n1][n2]=1  
        for k in range(args.offNum): 
            for interval in [i*crime_cycle[k] for i in range(args.iterative_cycle)]:
                for day in range(args.trnDays-interval):
                    for loc1 in range(args.areaNum):
                        for loc2 in range(loc1 + 1,args.areaNum): 
                            if self.trnT[loc1, day, k].any() and self.trnT[loc2, day+interval, k].any():
                                self.adj_matrix[loc1][loc2]=1
                                self.adj_matrix[loc2][loc1]=1
        return None
