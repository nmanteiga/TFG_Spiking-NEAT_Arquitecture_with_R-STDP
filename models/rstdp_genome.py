from random import random

from neat.attributes import FloatAttribute
from neat.genes import DefaultConnectionGene, DefaultNodeGene
from neat.genome import DefaultGenome, DefaultGenomeConfig

'''
genome for Spiking NEAT + R-STDP: the normal NEAT genome (nodes + connections) plus ONE extra gene, the learning rate
of R-STDP. evolution treats it like a weight: random at the start, inherited by the children, mutated a little
every generation. if learning while playing does not help, evolution can push it to 0 (learning switched off)

the config file needs a [RSTDPGenome] section (instead of [DefaultGenome]) with the learning_rate_* values
'''

# the same machinery neat-python uses for the connection weights: reads learning_rate_init_mean, _init_stdev,
# _init_type, _mutate_rate, _mutate_power, _replace_rate, _min_value and _max_value from the config
LEARNING_RATE = FloatAttribute("learning_rate")


class RSTDPGenomeConfig(DefaultGenomeConfig):
    ''' DefaultGenomeConfig + the learning_rate_* parameters '''

    def __init__(self, params, section_name):
        super().__init__(params, section_name)
        for p in LEARNING_RATE.get_config_params():
            setattr(self, p.name, p.interpret(params, section_name))
            self._params.append(p)          # so they are also written when neat saves the config (checkpoints)


class RSTDPGenome(DefaultGenome):

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DefaultNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return RSTDPGenomeConfig(param_dict, cls.__name__)

    def __init__(self, key):
        super().__init__(key)
        self.learning_rate = None

    def configure_new(self, config):
        ''' first generation: random learning rate (from the config's init mean / stdev, clamped to min / max) '''
        super().configure_new(config)
        self.learning_rate = LEARNING_RATE.init_value(config)

    def configure_crossover(self, genome1, genome2, config):
        ''' child of two parents: takes the learning rate of one of them at random (like any matching gene) '''
        super().configure_crossover(genome1, genome2, config)
        self.learning_rate = genome1.learning_rate if random() < 0.5 else genome2.learning_rate

    def mutate(self, config):
        ''' every generation: sometimes nudged a little, rarely replaced by a new random value '''
        super().mutate(config)
        self.learning_rate = LEARNING_RATE.mutate_value(self.learning_rate, config)

    def __str__(self):
        return f"Learning rate: {self.learning_rate:.5f}\n" + super().__str__()
