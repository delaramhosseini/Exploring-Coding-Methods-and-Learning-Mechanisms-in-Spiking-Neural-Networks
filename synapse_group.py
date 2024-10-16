from pymonntorch import Behavior
import torch

# _____________________________________________________Full connectivity_____________________________________________________________________


class FullConnectivityFirstOption(Behavior):
    def initialize(self, synapse):
        synapse.J0 = self.parameter("J0", None)
        self.alpha = self.parameter("alpha", 100) / 100

        self.N = synapse.src.size

        synapse.C = self.N
        synapse.W = synapse.matrix(synapse.J0 / self.N)
        synapse.I = synapse.dst.vector()

    def forward(self, synapse):
        # print(synapse.W)
        pre_spike = synapse.src.spike
        synapse.I += torch.sum(synapse.W[pre_spike], axis=0) - synapse.I * self.alpha


class FullConnectivitySecondOption(Behavior):
    def initialize(self, synapse):
        self.standardـdeviation = self.parameter("standardـdeviation", 0) / 100
        synapse.J0 = self.parameter("J0", 1)
        self.alpha = self.parameter("alpha", 100) / 100
        self.ignore_last_neuron = self.parameter("ignore_last_neuron", False)

        self.N = synapse.src.size

        synapse.C = self.N

        mean = synapse.J0 / self.N
        variation = abs(self.standardـdeviation * mean)
        synapse.W = synapse.matrix(mode=f"normal({mean}, {variation})")
        if self.ignore_last_neuron:
            synapse.W[:, -1] = 0
        synapse.I = synapse.dst.vector()

    def forward(self, synapse):
        pre_spike = synapse.src.spike
        synapse.I += torch.sum(synapse.W[pre_spike], axis=0) - synapse.I * self.alpha


# _______________________________________Random connectivity: fixed coupling probability________________________________________________


class Scaling(Behavior):
    def initialize(self, synapse):
        self.p = self.parameter("p", None)
        synapse.J0 = self.parameter("J0", None)
        self.alpha = self.parameter("alpha", 100) / 100
        self.standardـdeviation = self.parameter("standardـdeviation", 0) / 100

        self.N = synapse.src.size
        synapse.C = self.p * self.N

        mean = synapse.J0 / synapse.C
        variation = abs(self.standardـdeviation * mean)
        synapse.W = synapse.matrix(mode=f"normal({mean}, {variation})")
        self.test = torch.rand_like(synapse.W)
        synapse.W[self.test > (self.p)] = 0
        synapse.I = synapse.dst.vector()

    def forward(self, synapse):
        pre_spike = synapse.src.spike
        synapse.I += torch.sum(synapse.W[pre_spike], axis=0) - synapse.I * self.alpha


# __________________________________Random connectivity: fixed number of presynaptic partners________________________________________________


class FixedAAndFinite(Behavior):
    def initialize(self, synapse):
        synapse.J0 = self.parameter("J0", None)
        synapse.C = self.parameter("C", None)
        self.alpha = self.parameter("alpha", 100) / 100
        self.standardـdeviation = self.parameter("standardـdeviation", 0) / 100

        self.total_size_of_matrix = synapse.src.size * synapse.dst.size
        mean = synapse.J0 / synapse.C
        variation = abs(self.standardـdeviation * mean)
        synapse.W = synapse.matrix(mode=f"normal({mean}, {variation})")
        random_indices = torch.zeros_like(synapse.W)
        for i in range(synapse.dst.size):
            random_indices[:, i] = torch.randperm(synapse.src.size)
        synapse.W[random_indices > synapse.C] = 0
        synapse.I = synapse.dst.vector()

    def forward(self, synapse):
        pre_spike = synapse.src.spike
        synapse.I += torch.sum(synapse.W[pre_spike], axis=0) - synapse.I * self.alpha
