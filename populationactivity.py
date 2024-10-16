from pymonntorch import Behavior
import torch


class PopulationActivity(Behavior):
    def initialize(self, ng):
        ng.T = torch.sum(ng.spike) / (ng.size)

    def forward(self, ng):
        ng.T = torch.sum(ng.spike) / (ng.size)
