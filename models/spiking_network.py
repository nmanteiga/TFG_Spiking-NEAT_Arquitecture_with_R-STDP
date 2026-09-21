from .lif_neuron import LIFNeuron

'''
overall: initializing the memory, merging the sensory inputs with the delayed hidden spikes, accumulating the electrical 
         currents, stepping the biology forward by one frame, and passing the controller actions out to Mario
'''

class SpikingNetwork:
    def __init__(self, genome, config):
        ''' build the network '''

        # used on activate to save the info across spikes
        self.last_spikes = {}

        # identifies the sensory inputs and motor outputs which refer to what the AI sees and touches
        self.input_keys = config.genome_config.input_keys
        self.output_keys = config.genome_config.output_keys

        # instantiates a biological LIFNeuron for every node gene
        self.neurons = {}
        for node_id in genome.nodes:
            self.neurons[node_id] = LIFNeuron()

        # maps every active synapse with its pre-synaptic node, post-synaptic node, and weight.
        self.connections = []
        for (connection_id,connection_gene) in genome.connections.items():
            if connection_gene.enabled:
                in_node, out_node = connection_id
                weight = connection_gene.weight
                self.connections.append((in_node, out_node, weight))

        # to measure the energy efficiency
        self.total_spikes = 0
        self.synaptic_operations = 0



    def activate(self, inputs):
        ''' run one frame of the game at a time '''

        # pair two lists together into a dictionary
        # like this current_spikes[node_id] tells ian input node is active or not
        current_spikes = {node_id: val for node_id, val in zip(self.input_keys, inputs)}
        # adds the info to the previous spikes list
        current_spikes.update(self.last_spikes)

        # preparing the accumulator
        incoming_currents = {}
        for node_id in self.neurons:
            incoming_currents[node_id] = 0.0

        '''
        preparing flow of signal accross connections:
            input neuron (pixels on the screen) -> hidden neuron -> output neuron (buttons)
            |--------------------- this part ----------------------|
        '''
        for (in_node, out_node, weight) in self.connections:
            if (current_spikes.get(in_node, 0.0) > 0.0):
                incoming_currents[out_node] += weight
                self.synaptic_operations += 1               # 1 synop executed

        '''
        for the flow between the hidden neuron and the output neuron, a principle called axon conduction delay is used
        axon conduction delay refers to the time the electrical signals travel accross the entire brain:
            -> If Neuron A spikes in frame t, its signal arrives at Neuron B in frame t+1
                this is why current_spikes.update(self.last_spikes) exists
        right now:
            input neuron (pixels on the screen) -> hidden neuron -> output neuron (buttons)
                                                |---------------- this part ----------------|
        '''
        outputs = []
        new_spikes = {}

        for node_id, neuron in self.neurons.items():
            spike = neuron.step(incoming_currents[node_id])
            new_spikes[node_id] = spike
            if spike > 0.0:
                self.total_spikes += 1

        # extract outputs in the EXACT order defined by self.output_keys
        outputs = [new_spikes.get(key, 0.0) for key in self.output_keys]

        self.last_spikes = new_spikes
        return outputs