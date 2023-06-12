import sys
import numpy as np
import network as Network

class Reservoir():
    
    def __init__(self, d = None, filter_name = None, num_nodes = None, classifier = None,
                 input_probability = None, reservoir_probability = None):
        self.d = d 
        self.filter_name=filter_name
        self.classifier=classifier
        self.num_nodes=num_nodes
        self.input_probability=input_probability
        self.reservoir_probability=reservoir_probability
        self.Network=Network
        
        
    #Function that creates each reservoir separately
    def setup_reservoir(self, filter_name, classifier, num_nodes,
                 input_probability, reservoir_probability, d, Network):
        
        self.Network=Network
        self.d=d
        self.filter_name=filter_name
        self.classifier=classifier
        self.num_nodes=num_nodes
        self.input_probability=input_probability
        self.reservoir_probability=reservoir_probability    
        
        if classifier == 'log':
            self.d.build_train_labels_log()
            self.d.build_test_labels_log()
    
        elif classifier == 'lin':
            self.d.build_train_labels_lin()
            self.d.build_test_labels_lin()
            
        else:
        	print("This classifier is not supported for this test.")
        	sys.exit(1)

        self.d.build_training_matrix()
        self.d.build_test_matrix()
        self.Network.L = 3
        
        #Filtering the data
        if filter_name not in self.d.spectral_bands.keys():
        	print("The specified frequency band is not supported")
        	sys.exit(1)
        
        self.d.training_data = self.d.filter_data(self.d.training_data,filter_name)
        self.d.test_data = self.d.filter_data(self.d.test_data,filter_name)

        #Computing the absolute value of the data, to get rid of negative numbers
        self.d.training_data = np.abs(self.d.training_data)
        self.d.test_data = np.abs(self.d.test_data)
        
        ########################
        # Define the network parameters
        ########################
        
        self.Network.T = self.d.training_data.shape[1] #Number of training time steps
        self.Network.K = self.d.data.shape[0] #Input layer size
        
        self.Network.u = self.d.training_data
        self.Network.y_teach = self.d.training_results
        
        
    def training_network(self):
        self.Network.N = self.num_nodes #Reservoir layer size
        
        self.Network.setup_network(self.d,self.num_nodes,self.input_probability,self.reservoir_probability,self.d.data.shape[-1], self.classifier)
        
        self.Network.train_network(self.d.data.shape[-1],self.classifier,self.d.num_columns, self.d.num_trials_train, self.d.train_labels, self.Network.N) 
    

    def testing_network(self):
        self.Network.test_matrix = np.zeros([self.Network.N,self.d.num_trials_test*3,120])
        self.Network.test_network(self.d.test_data, self.d.num_columns,self.d.num_trials_test, self.Network.N, self.d.data.shape[-1], t_autonom=self.d.test_data.shape[1])
        
        if self.classifier == 'log':
            num_test_samples = np.shape(self.Network.test_matrix)[2]
            self.Network.test_matrix = self.Network.test_matrix.reshape(num_test_samples, -1)

            predictions = self.Network.regressor.predict(self.Network.test_matrix)
            mse = np.mean((predictions - self.d.test_labels) ** 2)
            #print("Mean Squared Error:", mse)
        
            count = 0
            for state in range(np.shape(predictions)[0]):
                if predictions[state] == self.d.test_labels[state]:
                    count += 1
                    
            print("accuracy log: ", count, "/", np.shape(predictions)[0], count/np.shape(predictions)[0] )
            
            
        elif self.classifier == 'lin':
            
            num_test_samples = np.shape(self.Network.test_matrix)[2]
            self.Network.test_matrix = self.Network.test_matrix.reshape(num_test_samples, -1)

            # Append a column of ones for intercept
            self.Network.test_matrix = np.hstack((np.ones((num_test_samples, 1)), self.Network.test_matrix))
            
            # Make predictions (261,3)
            predictions = np.dot(self.Network.test_matrix, self.Network.regressor)
            
            # Calculate mean squared error
            mse = np.mean((predictions - self.d.test_labels) ** 2)
            
            #print("Mean Squared Error:", mse)
            for row in range(np.shape(predictions)[0]):
                predictions[row] =[1 if np.max(predictions[row]) == i else 0 for i in predictions[row]]
            
            count = 0
            for state in range(np.shape(predictions)[0]):
                if (predictions[state] == self.d.test_labels[state]).all():
                    count += 1
                    
            print("accuracy lin: ", count, "/", np.shape(predictions)[0], count/np.shape(predictions)[0] )
