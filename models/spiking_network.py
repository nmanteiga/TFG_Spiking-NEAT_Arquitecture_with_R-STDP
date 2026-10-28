from .lif_neuron import LIFNeuron

'''
overall: initializing the memory, merging the sensory inputs with the delayed hidden spikes, accumulating the electrical
         currents, stepping the biology forward by one frame, and passing the controller actions out to Mario

with plastic=True the network also learns while it plays (R-STDP):
    traces       -> short memory of each node's recent spikes
    eligibility  -> per synapse: "was this synapse involved in a pre-before-post firing?"
    reward()     -> dopamine (reward better/worse than expected) turns eligibility into weight changes
'''

TINY = 1e-4     # traces / eligibilities below this are set to exactly 0 and skipped (event-driven: no work, no energy)


class SpikingNetwork:
    def __init__(self, genome, config, plastic=False, learning_rate=0.01, trace_decay=0.8, eligibility_decay=0.95,
                 a_plus=1.0, a_minus=1.0, reward_scale=15.0, baseline_rate=0.05):
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
        # lists (not tuples) so R-STDP can change the weight in place: connection[2] = new_weight
        self.connections = []
        for (connection_id,connection_gene) in genome.connections.items():
            if connection_gene.enabled:
                in_node, out_node = connection_id
                weight = connection_gene.weight
                self.connections.append([in_node, out_node, weight])

        # to measure the energy efficiency (read by energy_model.snn_energy)
        self.total_spikes = 0
        self.synaptic_operations = 0
        self.neuron_updates = 0         # one membrane update (leak) per neuron per frame
        self.input_spikes = 0           # active input pixels, to report how sparse the input really is
        self.frames = 0
        self.plasticity_updates = 0     # R-STDP synapse work: eligibility changes/decays + weight updates
        self.trace_updates = 0          # R-STDP node work: trace decays (cheaper, counted apart)
        self.n_neurons = len(self.neurons)
        self.n_connections = len(self.connections)
        self.n_inputs = len(self.input_keys)
        self.potential_outputs = [0.0] * len(self.output_keys)   # one membrane potential per output (button)

        # ---------------------------------- R-STDP ----------------------------------
        # a learning rate of 0 (evolution can choose it) means no weight can ever change: then traces and eligibility
        # would be wasted work (and wasted energy), so the network behaves exactly as a non-plastic one
        self.plastic = plastic and learning_rate > 0.0
        if self.plastic:
            self.learning_rate = learning_rate          # how big each weight change is (step 4: evolved per genome)
            self.trace_decay = trace_decay              # traces fade x0.8 per frame  -> remember ~5 frames
            self.eligibility_decay = eligibility_decay  # eligibility fades x0.95     -> remember ~20 frames
            self.a_plus = a_plus                        # strength of "pre before post" (potentiation)
            self.a_minus = a_minus                      # strength of "post before pre" (depression)
            self.reward_scale = reward_scale            # game reward per frame is about -15..15 -> about -1..1
            self.baseline_rate = baseline_rate          # how fast the "expected reward" follows the real one
            self.weight_max = config.genome_config.weight_max_value

            # pre trace: every node that SENDS spikes through a synapse (inputs/pixels AND neurons)
            # post trace: every node that RECEIVES spikes through a synapse (only neurons: inputs receive nothing)
            # nodes without synapses get no trace: nobody would read it, so keeping it would be wasted work (energy)
            self.pre_trace = {in_node: 0.0 for (in_node, _, _) in self.connections}
            self.post_trace = {out_node: 0.0 for (_, out_node, _) in self.connections}

            # one eligibility per synapse, same order as self.connections
            self.eligibility = [0.0] * len(self.connections)
            # sign of every weight when the network is built: R-STDP may never flip it
            self.signs = [1.0 if weight >= 0.0 else -1.0 for (_, _, weight) in self.connections]

            self.output_set = set(self.output_keys)
            self.reward_average = 0.0                   # "expected reward" (running average)
            self.last_dopamine = 0.0                    # kept only to inspect / plot


    def activate(self, inputs):
        ''' run one frame of the game at a time '''

        # pair two lists together into a dictionary
        # like this current_spikes[node_id] tells ian input node is active or not
        current_spikes = {node_id: val for node_id, val in zip(self.input_keys, inputs)}
        # adds the info to the previous spikes list
        current_spikes.update(self.last_spikes)

        self.frames += 1
        self.input_spikes += sum(1 for v in inputs if v > 0.0)
        self.neuron_updates += len(self.neurons)

        if self.plastic:
            # R-STDP step 1: every trace fades a little
            self._decay(self.pre_trace, self.trace_decay)
            self._decay(self.post_trace, self.trace_decay)
            # R-STDP step 2: pre traces of the spikes DELIVERED this frame (before the connection loop, so the
            # eligibility below sees the pre spike that just arrived)
            for node_id, spike in current_spikes.items():
                if spike > 0.0 and node_id in self.pre_trace:
                    self.pre_trace[node_id] = 1.0   # set to 1, not +1: always-on pixels (ground) stay at 1, not ~5

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

        if self.plastic:
            # R-STDP step 3: eligibility of every synapse
            self._update_eligibility(current_spikes, new_spikes)
            # R-STDP step 4: post traces of the spikes FIRED this frame. done AFTER the eligibility on purpose:
            # if a pre spike arrives and the post neuron fires in the same frame, that pre spike caused it, so the
            # depression ("post before pre") must only see post spikes from EARLIER frames
            for node_id, spike in new_spikes.items():
                if spike > 0.0 and node_id in self.post_trace:
                    self.post_trace[node_id] = 1.0

        # extract outputs in the EXACT order defined by self.output_keys
        outputs = [new_spikes.get(key, 0.0) for key in self.output_keys]
        # extract potentials in the EXACT order defined by self.output_keys
        # (the potentials are not in new_spikes: each LIFNeuron object keeps its own in after_leak)
        self.potential_outputs = [self.neurons[key].after_leak for key in self.output_keys]

        self.last_spikes = new_spikes
        return outputs


    def reward(self, game_reward, chosen_action):
        '''
        called by the training script after every env.step(), with the reward of that frame and the index of the
        button that was pressed. turns eligibility into weight changes:
            dopamine   = reward - expected reward          (better or worse than expected?)
            new weight = weight + learning_rate * dopamine * eligibility
        '''
        if not self.plastic:
            return

        # dopamine: compare with the expected reward BEFORE updating it (otherwise the average already includes it)
        r = game_reward / self.reward_scale
        dopamine = r - self.reward_average
        self.reward_average += self.baseline_rate * (r - self.reward_average)
        self.last_dopamine = dopamine
        if dopamine == 0.0:
            return

        # only the output of the button that was actually pressed takes credit (or blame)
        chosen_output = self.output_keys[chosen_action]

        for i, connection in enumerate(self.connections):
            e = self.eligibility[i]
            if e == 0.0:
                continue                                    # not involved -> nothing to do
            out_node = connection[1]
            if out_node in self.output_set and out_node != chosen_output:
                continue                                    # an output that was not chosen: leave it alone

            weight = connection[2] + self.learning_rate * dopamine * e
            # keep the sign and the limits of the genome: positive stays in [0, max], negative in [-max, 0]
            if self.signs[i] > 0:
                weight = min(max(weight, 0.0), self.weight_max)
            else:
                weight = max(min(weight, 0.0), -self.weight_max)
            connection[2] = weight
            self.plasticity_updates += 1


    def _update_eligibility(self, current_spikes, new_spikes):
        ''' per synapse: fade, + if post fired (pre trace), - if pre arrived (post trace) '''
        for i, (in_node, out_node, _) in enumerate(self.connections):
            e = self.eligibility[i]

            # fade (skipped when it is already 0: event-driven)
            if e != 0.0:
                e *= self.eligibility_decay
                self.plasticity_updates += 1
                if abs(e) < TINY:
                    e = 0.0

            # depression: the pre spike arrives AFTER the post neuron fired (post trace still from earlier frames)
            if current_spikes.get(in_node, 0.0) > 0.0 and self.post_trace[out_node] > 0.0:
                e -= self.a_minus * self.post_trace[out_node]
                self.plasticity_updates += 1

            # potentiation: the post neuron fires and the pre node was active just before (or now)
            if new_spikes[out_node] > 0.0 and self.pre_trace.get(in_node, 0.0) > 0.0:
                e += self.a_plus * self.pre_trace[in_node]
                self.plasticity_updates += 1

            self.eligibility[i] = e


    def _decay(self, traces, decay):
        ''' fade every non-zero trace; tiny ones become exactly 0 so they cost nothing afterwards '''
        for node_id, value in traces.items():
            if value != 0.0:
                value *= decay
                traces[node_id] = value if value >= TINY else 0.0
                self.trace_updates += 1
