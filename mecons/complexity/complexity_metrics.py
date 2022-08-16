"""
    Complexity class to compute different complexity metrics.
    This code was created based on Michael Schartner's code.
    Author: Michael Schartner, m.schartner@sussex.ac.uk
            Christian ferreyra, chrisferreyra13@gmail.com
    Date: 09.12.14 - 2022
"""

import numpy as np
from scipy import signal
from random import shuffle


class Complexity():
    def __init__(self,data=None):
        self.data=data
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        self._data=value

    @property
    def result(self):
        return self._result
    
    @result.setter
    def result(self,value):
        print("[INFO]: The result is ready!")
        self._result=value

    @staticmethod
    def detrending_normalization(data, first_mean=True):
        """Detrending and baseline subtraction on input data."""
        #TODO: think about the first_mean
        ro, co = np.shape(data)
        data_processed = np.zeros((ro, co))

        for i in range(ro):
            if first_mean:
                data_processed[i, :] = signal.detrend(
                    data[i, :]-np.mean(data[i, :]), axis=0)
            else:
                data_processed[i, :] = signal.detrend(data[i, :], axis=0)
                data_processed[i, :] = data_processed[i, :]-np.mean(data[i, :])

        return data_processed

    @staticmethod
    def lempel_ziv_welch_compression(binary_str):
        """Return the size of the dictionary of binary words after Lempel-Ziv-Welch compression."""
        d = {}
        w = ''
        i = 1
        for c in binary_str:
            wc = w + c
            if wc in d:
                w = wc
            else:
                d[wc] = wc
                w = c
            i += 1

        return len(d)

    @staticmethod
    def binarize_matrix(data):
        """Binarizes the input multidimensional time series based on Hilbert transform amplitude."""
        ro, co = np.shape(data)
        th = 0
        m = np.zeros((ro, co))
        binary_matrix = np.zeros((ro, co))
        for i in range(ro):
            m[i, :] = abs(signal.hilbert(data[i, :]))
            th = np.mean(m[i, :])
            for j in range(co):
                if m[i, :] >= th :
                    binary_matrix[i,j]=1

        return binary_matrix

    
    @staticmethod
    def binary_matrix_to_string(binary_matrix):
        """Creates one string being the binarized input matrix concatenated comlumn-by-column."""
        ro, co = np.shape(binary_matrix)
        binary_str = ''
        for j in range(co):
            for i in range(ro):
                if binary_matrix[i, j]==1:
                    binary_str += '1'
                else:
                    binary_str += '0'

        return binary_str
    
    @staticmethod
    def map_matrix_to_integer(binary_matrix):
        """Bijection, mapping each binary column of binary matrix psi onto an integer."""
        ro,co=np.shape(binary_matrix) 
        c=np.zeros(co)
        for t in range(co):
            for j in range(ro):
                c[t]=c[t]+binary_matrix[j,t]*(2**j)

        return c
    
    @staticmethod
    def compute_entropy(string):
        """Computes the Shannon entropy of a string."""
        string = list(string)
        prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]
        entropy = - sum([ p * np.log2(p) for p in prob ])

        return entropy
        
    @staticmethod
    def compute_synchrony(p1,p2,threshold=0.8):
        """Computes a binary synchrony time series between two phase time series."""
        differences=np.array(abs(p1-p2))
        sync_time_series=np.zeros(len(differences))
        for i in range(len(differences)):
            if differences[i]>np.pi:
                differences[i]=2*np.pi-differences[i]
            if differences[i]<threshold:
                sync_time_series[i]=1

        return sync_time_series
    
    @staticmethod
    def create_random_binary_matrix(n_rows,n_columns):
        """Creates a random binary matrix."""
        binary_matrix=np.random.rand(n_rows,n_columns)
        for i in range(n_rows):
            for j in range(n_columns):
                if binary_matrix[i,j]>0.5:
                    binary_matrix[i,j]=1
                else:
                    binary_matrix[i,j]=0

        return binary_matrix
    
    
    def compute_synchrony_matrix(self,data):
        """Computes binary synchrony matrix (by column) based on a multidimensional time series matrix."""
        phases_matrix=np.angle(signal.hilbert(data))
        ro,co=np.shape(phases_matrix)
        synch_matrix=np.zeros((ro, ro-1, co))  
        for i in range(ro):
            l=0
            for j in range(ro):
                if i!=j:
                    synch_matrix[i,l]=self.compute_synchrony(phases_matrix[i],phases_matrix[j])
                    l+=1
        
        return synch_matrix

    def compute_lempel_ziv_complexity(self, data=None):
        """Computes LZc and use shuffled result as normalization."""
        if data:
            self.data=data
        elif not self.data:
            print("Error: data is empty")
            return None

        data = self.detrending_normalization(self.data)
        binary_str = self.binary_matrix_to_string(data)
        m = list(binary_str)
        shuffle(m)
        w = ''
        for i in range(len(m)):
            w += m[i]

        self.result = self.lempel_ziv_welch_compression(binary_str)/float(self.lempel_ziv_welch_compression(w))
    
    
    def compute_amplitude_coalition_entropy(self,data):
        """Computes Amplitude Coalition Entropy (ACE), using shuffled result as normalization."""
        if data:
            self.data=data
        elif not self.data:
            print("Error: data is empty")
            return None

        data=self.detrending_normalization(self.data, first_mean=False)
        ro,co=np.shape(data)
        data=self.binarize_matrix(data)
        entropy=self.compute_entropy(data)
        # shuffle the data for normalization
        for i in range(ro):
            shuffle(data[i])
        
        shuffle_entropy=self.compute_entropy(data)
        self.result= entropy/float(shuffle_entropy)

    def compute_synchrony_coalition_entropy(self,data):
        """Computes Synchrony Coalition Entropy (SCE), using shuffled result as normalization."""
        if data:
            self.data=data
        elif not self.data:
            print("Error: data is empty")
            return None

        data = self.detrending_normalization(self.data)
        ro,co=np.shape(data)
        data=self.compute_synchrony_matrix(data)
        ce=np.zeros(ro)
        norm=self.compute_entropy(self.map_matrix_to_integer(self.create_random_binary_matrix(ro-1,co)))
        for i in range(ro):
            c=self.map_matrix_to_integer(data[i])
            ce[i]=self.compute_entropy(c)

        self.result = np.mean(ce)/norm
        #TODO: see if it should return ce/norm

    
    