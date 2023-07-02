import scipy.io
import numpy as np
import scipy.signal as spsg
from keras.utils import to_categorical

class Data():
    def __init__(self,training_percentage):
        self.data = None
        self.training_data = None
        self.training_results = None
        self.test_data = None
        self.test_results = None
        self.spectral_bands = {
            'alpha' : [8.,12.],
            'beta'  : [15.,30.],
            'gamma' : [40.,80.],
        }
        self.training_percentage = training_percentage
        self.test_percentage = 100-training_percentage
        self.fs = 2048
        self.num_columns = None
        self.num_trials_train = None 
        self.num_trials_test = None
        self.train_labels = None
        self.test_labels = None

        
    def import_data(self,file):
        """
            Loading data, rounding to 4 decimals
        """

        self.data = scipy.io.loadmat(file)
        self.data = self.data['ic_data3'] 
        self.data = self.data.round(decimals=4)
        
        data_solo = self.data[:,:,:,[0,1,6,7]]
        data_easy = self.data[:,:,:,[2,3,8,9]]
        data_hard = self.data[:,:,:,[4,5,10,11]]
        
        data_joined_solo = np.array(data_solo[:,:,:,0])
        data_joined_easy = np.array(data_easy[:,:,:,0])
        data_joined_hard = np.array(data_hard[:,:,:,0])

        for i in range(1,4):
            data_joined_solo = np.append(data_joined_solo, data_solo[:,:,:,i], axis = 2)
            data_joined_easy = np.append(data_joined_easy, data_easy[:,:,:,i], axis = 2)
            data_joined_hard = np.append(data_joined_hard, data_hard[:,:,:,i], axis = 2)
            
        self.data = np.stack([data_joined_solo, data_joined_easy, data_joined_hard], axis = 3)

        #downsampling
        
        self.data = self.data[:,::10,:,:]
        
        self.num_columns = self.data.shape[1]
        self.num_trials_train = int((self.data.shape[2]*self.training_percentage)/100)
        self.num_trials_test = self.data.shape[2]-self.num_trials_train
        
    def import_from_matrix(self,matrix):
        """
            Loading data from matrix
        """
        self.data = matrix 
        self.num_columns = self.data.shape[1]
        self.num_trials_train = int((self.data.shape[2]*self.training_percentage)/100)
        self.num_trials_test = self.data.shape[2]-self.num_trials_train
        
   

    def build_training_matrix(self):
        """            
            Builds the matrix for training the model.
            
        """
        
        training_amount = self.num_trials_train
        self.training_data = np.zeros([self.data.shape[0],self.data.shape[1],training_amount*self.data.shape[3]])
        self.training_data[:,:,:training_amount] = self.data[:,:,:training_amount,0]
        self.training_data[:,:,training_amount:training_amount*2] = self.data[:,:,:training_amount,1]
        self.training_data[:,:,training_amount*2:training_amount*3] = self.data[:,:,:training_amount,2]
        
    def build_test_matrix(self):
        """
            Builds the matrix for testing the model.
            
        """
        
        test_amount = self.num_trials_test
        
        self.test_data = np.zeros([self.data.shape[0],self.data.shape[1],test_amount*self.data.shape[3]])
        self.test_data[:,:,:test_amount] = self.data[:,:,:test_amount,0]
        self.test_data[:,:,test_amount:test_amount*2] = self.data[:,:,:test_amount,1]
        self.test_data[:,:,test_amount*2:test_amount*3] = self.data[:,:,:test_amount,2]
    
    def filter_data(self,data,range_filter):

        """
        Filters the data in the specified bandwidth range
        """

        low_freq, high_freq = self.spectral_bands[range_filter]
        low_freq, high_freq = low_freq/self.fs, high_freq/self.fs
        
        b,a = spsg.iirfilter(3, [low_freq,high_freq], btype='bandpass', ftype='butter')
        data = spsg.filtfilt(b, a, data, axis=1)
        
        return data

        
    def build_train_labels_log(self):
        """
        Building the train labels for logistic classifier
        """
        self.train_labels = np.zeros([self.num_trials_train*3,]) # 3 == Number of states
        self.train_labels[:self.num_trials_train] = 0
        self.train_labels[self.num_trials_train:self.num_trials_train*2] = 1
        self.train_labels[self.num_trials_train*2:self.num_trials_train*3] = 2
        self.train_labels = self.train_labels.reshape((self.num_trials_train)*3)

    def build_test_labels_log(self):
        """
        Building the test labels for logistic classifier
        """

        self.test_labels = np.zeros([self.num_trials_test*3,]) # 3 == Number of states
        self.test_labels[:self.num_trials_test] = 0
        self.test_labels[self.num_trials_test:self.num_trials_test*2] = 1
        self.test_labels[self.num_trials_test*2:self.num_trials_test*3] = 2
        self.test_labels = self.test_labels.reshape((self.num_trials_test*3))


    def build_train_labels_lin(self):
        """
        Building the train labels for logistic classifier
        """
        #print(self.num_trials_train*5)
        self.train_labels = np.zeros([self.num_trials_train*3,]) # 3 == Number of states
        self.train_labels[:self.num_trials_train] = 0
        self.train_labels[self.num_trials_train:self.num_trials_train*2] = 1
        self.train_labels[self.num_trials_train*2:self.num_trials_train*3] = 2
        self.train_labels = to_categorical(self.train_labels)
        

    def build_test_labels_lin(self):
        """
        Building the test labels for logistic classifier
        """

        self.test_labels = np.zeros([self.num_trials_test*3,]) # 3 == Number of states
        self.test_labels[:self.num_trials_test] = 0
        self.test_labels[self.num_trials_test:self.num_trials_test*2] = 1
        self.test_labels[self.num_trials_test*2:self.num_trials_test*3] = 2
        self.test_labels = to_categorical(self.test_labels)

