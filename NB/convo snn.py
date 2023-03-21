import numpy as np
import matplotlib.pyplot as plt
import pyNN.spiNNaker as sim

# Define network architecture
input_size = (28, 28)
conv_size = (5, 5)
num_filters = 16
pool_size = (2, 2)
hidden_size = 64
output_size = 10

# Initialize simulation
sim.setup(timestep=1.0)

# Create input layer
input_layer = sim.Population(np.prod(input_size), sim.SpikeSourceArray)

# Create convolutional layer
conv_layer = sim.Population(num_filters, sim.IF_curr_exp, 
                            cellparams={'tau_m': 20.0, 'v_rest': -65.0, 'v_reset': -65.0,
                                        'v_thresh': -52.0, 'tau_syn_E': 2.0},
                            label='conv_layer')

# Create pooling layer
pool_layer = sim.Population(num_filters, sim.IF_curr_exp, 
                            cellparams={'tau_m': 20.0, 'v_rest': -65.0, 'v_reset': -65.0,
                                        'v_thresh': -52.0, 'tau_syn_E': 2.0},
                            label='pool_layer')

# Create hidden layer
hidden_layer = sim.Population(hidden_size, sim.IF_curr_exp, 
                              cellparams={'tau_m': 20.0, 'v_rest': -65.0, 'v_reset': -65.0,
                                          'v_thresh': -52.0, 'tau_syn_E': 2.0},
                              label='hidden_layer')

# Create output layer
output_layer = sim.Population(output_size, sim.IF_curr_exp, 
                              cellparams={'tau_m': 20.0, 'v_rest': -65.0, 'v_reset': -65.0,
                                          'v_thresh': -52.0, 'tau_syn_E': 2.0},
                              label='output_layer')

# Define synaptic weights
conv_weights = np.random.randn(num_filters, np.prod(conv_size)) * 0.01
hidden_weights = np.random.randn(hidden_size, num_filters * np.prod(pool_size)) * 0.01
output_weights = np.random.randn(output_size, hidden_size) * 0.01

# Create synapses
input_conv_syn = sim.Projection(input_layer, conv_layer, 
                                sim.FromListConnector(conv_weights.tolist()), 
                                receptor_type='excitatory', label='input_conv_syn')
conv_pool_syn = sim.Projection(conv_layer, pool_layer, 
                               sim.OneToOneConnector(), receptor_type='excitatory', label='conv_pool_syn')
pool_hidden_syn = sim.Projection(pool_layer, hidden_layer, 
                                 sim.FromListConnector(hidden_weights.tolist()), 
                                 receptor_type='excitatory', label='pool_hidden_syn')
hidden_output_syn = sim.Projection(hidden_layer, output_layer, 
                                   sim.FromListConnector(output_weights.tolist()), 
                                   receptor_type='excitatory', label='hidden_output_syn')

# Define input spike train
input_spikes = np.zeros(input_size)
input_spikes[14, 14] = 1.0
input_spikes = input_spikes.flatten()

# Run simulation
simtime = 100.0
input_layer.set_spikes([(i, [t]) for i, t in enumerate(np.arange(0, simtime, 1.0)) if input_spikes[i] > 0])
sim.run(simtime)

# Retrieve output
