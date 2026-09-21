class LIFNeuron:
    def __init__(self, threshold=1.0, leak_factor=0.9):
        self.membrane_potential = 0.0
        self.resting_potential = 0.0
        self.threshold = threshold
        self.leak_factor = leak_factor
        
    def step(self, incoming_current):
        '''
        incomming_current:  float is the sum of all the spikes that arrived from other neurons during this frame, 
                            multiplied by their respective synaptic weights
        '''
        # integrate
        self.membrane_potential += incoming_current

        if (self.membrane_potential <= self.resting_potential):
            self.membrane_potential = self.resting_potential

        # leak
        self.membrane_potential *= self.leak_factor

        # fire
        if (self.membrane_potential >= self.threshold):
            self.membrane_potential = self.resting_potential
            return 1.0
        else:
            return 0.0