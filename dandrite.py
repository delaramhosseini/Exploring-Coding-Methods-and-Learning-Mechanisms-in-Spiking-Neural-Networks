from pymonntorch import Behavior
import torch


class Dandrite(Behavior):
    def initialize(self, ng):
        ng.I = torch.tensor(ng.I_inp)

    def forward(self, ng):
        ng.I = torch.tensor(ng.I_inp)
        for synapse in ng.afferent_synapses["All"]:
            ng.I += synapse.I
