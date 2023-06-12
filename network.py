import numpy as np
import scipy.linalg
from sklearn.linear_model import LinearRegression,LogisticRegression
import sklearn.neighbors as sklnn
from numpy.linalg import inv, pinv

class Network():
    def __init__(self, T = None, K = None, N = None, L = None, W_in = None, W = None, W_back = None, W_out = None): 
        
        self.T = T #number of training time steps (integer)
        
        self.K = K #dimension of the input (integer) (may be None)
        self.N = N #dimension of the reservoir, i.e, number of nodes (integer)
        self.L = L #dimension of the output (integer)
        
        self.W_in = W_in #input connections (matrix of size self.N x self.K)
        self.W = W #adjacency matrix (matrix of size self.N x self.N)
  
        self.initial_state = None #initial state of the reservoir (state forgetting property)        
        self.trajectories = None #dynamics of the reservoir (matrix of size self.T x self.N) 
        self.regressor = None #regressor
        self.y_teach = None #desired output of the network (matrix of size self.L x self.T)
        self.y_teach_test = None #y_teach for doing the test (matrix of size self.L x (t_dismiss+t_autonom))                
        self.u = None #input (matrix of size self.K x self.T) 
        self.u_test = None #input durint training (matrix of size self.K x t_dismiss+t_autonom) 
        
        self.train_matrix = None #matrix where we will perform the mean of each state in training
        self.test_matrix = None #matrix where we will perform the mean of each state in testing
        
    def setup_network(self,d,k,inpu,reser,states,classifier):
        
        self.u = d.training_data
        self.train_matrix = np.zeros([k,d.num_trials_train*states,self.T])

        ########################
        # Input Layer
        ########################

        self.W_in = np.zeros([self.N,self.K]) #input matrix

        for i in np.arange(self.N):
            for j in np.arange(self.K):
                p = np.random.uniform()
                if 0 <= p <= inpu:
                    self.W_in[i,j] = np.random.uniform(-1., 1.)            
                else:
                    self.W_in[i,j] = 0

        ########################
        # Reservoir Layer
        ########################
        
        self.W = np.zeros([self.N,self.N]) #reservoir matrix
        for i in np.arange(self.N):
            for j in np.arange(self.N):
                p = np.random.uniform()
                if 0 <= p <= reser:
                    self.W[i,j] = np.random.uniform(-1., 1.)            
                else:
                    self.W[i,j] = 0

        self.W = self.W - self.W.T

		########################
        # Making sure the largest eigenvalue in module is < 1
        ########################
        alpha = 0.22/max(abs(scipy.linalg.eigvals(self.W)))
        self.W = alpha*self.W
        
        
    def compute_nodes_trajectories(self,num_columns, num_trials, test=False, t_autonom=None):
        #initial state of the reservoir
        if test == False:
            x_prev = self.initial_state
        if test == True:
            #x_prev = self.trajectories[-1,:,:]
            x_prev = np.ones((self.N,num_trials*3))
            
        leaking_rate = 0.01
        if test == False:
            self.trajectories = np.zeros([self.T,self.N,num_trials*3])
            for n in np.arange(self.T):
                x = x_prev*(1-leaking_rate) + leaking_rate*np.tanh(np.dot(self.W_in,self.u[:,n,:])+np.dot(self.W,x_prev)) #state update equation
                self.trajectories[n,:,:] = x
                x_prev = x
            self.train_matrix = self.trajectories
            return self

        elif test == True: 
            matrix = np.zeros((self.T,self.N,num_trials*3))
            for n in np.arange(num_trials):
                x = x_prev*(1-leaking_rate) + leaking_rate*np.tanh(np.dot(self.W_in,self.u_test[:,n,:])+np.dot(self.W,x_prev)) #state update equation
                matrix[n,:,:] = x
                x_prev = x
            self.test_matrix = matrix
            return self

            
    def train_network(self, num_states, classifier ,num_columns, num_trials, labels, num_nodes):
        """
			Method responsible for processing the training data trough the network and fitting the result to the desired classifier
		"""
        if classifier == 'log':
            #Define the initial state (which is not relevant due to state forgetting property)
            self.initial_state = np.ones((self.N,num_trials*3))
            #Data through network
            self.compute_nodes_trajectories(num_columns, num_trials)
            num_samples = np.shape(self.train_matrix)[2]

            self.train_matrix = self.train_matrix.reshape(num_samples,-1)  # reshape to (num_samples, 100*120)
            labels = labels.reshape((num_samples))  # reshape to (num_samples, 3*1)
            
            regressor = LogisticRegression(solver='newton-cg', random_state=0, max_iter = 10000)
            regressor.fit(self.train_matrix, labels)
            
            self.regressor = regressor

        
        elif classifier == 'lin':
            #Define the initial state (which is not relevant due to state forgetting property)
            self.initial_state = np.ones((self.N,num_trials*3))
            #Data trough network
            self.compute_nodes_trajectories(num_columns, num_trials)
            
            #self.train_matrix = self.train_matrix.reshape((num_nodes,num_trials*num_states),order='F')
            # Reshape matrices and labels for linear regression
            num_samples = np.shape(self.train_matrix)[2]

            self.train_matrix = self.train_matrix.reshape(num_samples,-1)  # reshape to (num_samples, 100*120)

            labels = labels.reshape(num_samples, -1)  # reshape to (num_samples, 3*1)

            # Append a column of ones to matrices for the intercept term
        
            self.train_matrix = np.hstack((np.ones((num_samples, 1)), self.train_matrix))
            
            penalty = 1
            # Calculate the least squares solution
            matrix = np.dot(self.train_matrix.T, self.train_matrix)
            matrix = matrix + np.identity(np.shape(matrix)[0])*penalty
            inv_matrix = scipy.linalg.inv(matrix)
            coefficients = np.dot(inv_matrix, np.dot(self.train_matrix.T, labels))
            # Extract the intercept and coefficients
            intercept = coefficients[0]
            weights = coefficients[1:]

            # Print the intercept and coefficients
            #print("Intercept:", intercept)
            #print("Weights:", weights)
            
            self.regressor = coefficients

        return self  
    
    
    def test_network(self, data, num_columns, num_trials, num_nodes, states, t_autonom):
        """
    		Method responsible for processing the testing data trough the network
        """ 
        self.u_test = data                        
                      
        self.compute_nodes_trajectories(num_columns, num_trials, test=True, t_autonom=t_autonom)
        
        return self
            