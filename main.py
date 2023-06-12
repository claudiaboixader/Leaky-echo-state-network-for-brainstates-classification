import reservoir as Reservoir
import network as Network
import data as Data


filter_name = 'gamma'
classifier = 'lin'
num_nodes = 75
input_probability = 0.5
reservoir_probability = 0.5

#Define the class data
file = 'dataClean-ICA3-25-T1.mat' 
d = Data.Data(80) #80% training 20% testing
d.import_data(file)


#Creating the reservoir layer
reservoir = Reservoir.Reservoir()
network_r = Network.Network()
d_r = Data.Data(80)
d_r.import_from_matrix(d.data)
reservoir.setup_reservoir(filter_name, classifier, num_nodes, 
                         input_probability, reservoir_probability, d_r, network_r)
        
#Training and testing the reservoir
reservoir.training_network()
#print('\nTESTING RESERVOIR')
reservoir.testing_network()

   