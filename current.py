from pymonntorch import Behavior
import torch
import random
import numpy as np

torch.manual_seed(45)


class ConstanceCurrent(Behavior):
    def initialize(self, ng):
        self.current = self.parameter("current", None)
        self.noise_range = self.parameter("noise_range", 0)

        ng.I_inp = ng.vector(self.current)

    def forward(self, ng):
        rand = (ng.vector("uniform") - 0.5) * self.noise_range
        ng.I_inp = ng.vector(self.current) + rand


class UniformCurrent(Behavior):
    def initialize(self, ng):
        self.current = self.parameter("current", 10) * 2
        self.tau = self.parameter("tau_I", 1)
        self.noise_range = self.parameter("noise_range", 0)

        ng.I_inp = ng.vector("uniform") * self.current

    def forward(self, ng):
        rand = (ng.vector("uniform") - 0.5) * self.noise_range
        ng.I_inp += (ng.vector("uniform") - (ng.I_inp / self.current)) * self.tau + rand


class UniformCurrentInOneLine(Behavior):

    def initialize(self, ng):
        self.max_current = self.parameter("current", 6) * 2
        self.step = self.parameter("step", 0.5)
        self.noise_range = self.parameter("noise_range", 0)
        self.initial_current = self.parameter("initial_current", None)
        ng.I_inp = ng.vector(random.random()) * self.max_current
        if self.initial_current != None:
            ng.I_inp = (ng.vector(self.initial_current) / 100) * self.max_current / 2

    def forward(self, ng):
        I = (random.random() - (ng.I_inp / self.max_current)) * self.step
        ng.I_inp += I + (random.random() - 0.5) * self.noise_range


class StepCurrent(Behavior):
    def initialize(self, ng):
        self.t0 = self.parameter("t0", 0)
        self.t1 = self.parameter("t1", 100)
        self.initial_current = self.parameter("current", 0)
        self.current0 = self.parameter("current0", 10)
        self.current1 = self.parameter("current1", 0)
        self.noise_range = self.parameter("noise_range", 0)

        ng.I_inp = ng.vector(self.initial_current)

    def forward(self, ng):
        rand = (ng.vector("uniform") - 0.5) * self.noise_range
        if self.t0 < ng.network.iteration * ng.network.dt:
            ng.I_inp = ng.vector(self.current0) + rand

        if self.t1 < ng.network.iteration * ng.network.dt:
            ng.I_inp = ng.vector(self.current1) + rand
        ng.I_inp += rand
