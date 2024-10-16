from pymonntorch import Behavior


class CountSpikes(Behavior):
    def initialize(self, ng):
        ng.num_of_each_nueron_spikes = ng.vector()

    def forward(self, ng):
        ng.num_of_each_nueron_spikes += ng.spike.byte()
