from pymonntorch import Behavior
import torch
import copy
import numpy as np
import scipy
import scipy.stats


class TimeToFirstSpike(Behavior):
    def initialize(self, ng):
        self.input_time = int(
            self.parameter("time", required=True) / ng.network.dt
        )  # time to iteration
        self.rest_time = int(self.parameter("rest_time", 10) / ng.network.dt)
        self.data = torch.tensor(self.parameter("data", required=True))
        self.number_of_data = self.data.shape[0]
        self.size = self.data.shape[1]
        self.epsilon = self.parameter("epsilon", 0.001)
        max_data = self.data.max()
        min_data = self.data.min()
        self.range = self.parameter("range", None) or (min_data, max_data)

        # print("range: ", self.range)
        ng.network.inp_duration = self.input_time + self.rest_time

        self.first_layer_input = []
        ng.network.current_inp_idx = 0

        for d in self.data:
            self.first_layer_input.append(self.encode_ttfs(ng, d))

        ng.spike = ng.vector() != 0

    def encode_ttfs(self, ng, d):
        data = copy.deepcopy(d)
        encode_matrix = torch.zeros((ng.network.inp_duration, self.size)) != 0

        # mapping data to [0, 1]
        data = (data - self.range[0]) / (self.range[1] - self.range[0])

        for t in range(self.input_time):
            thereshold = 1 - ((t + 1) / self.input_time)
            encode_matrix[t][:] = thereshold <= data
            data[thereshold <= data] = -1
        return encode_matrix

    def forward(self, ng):
        ng.spike = torch.logical_or(
            self.first_layer_input[ng.network.current_inp_idx][
                (ng.network.iteration - 1) % ng.network.inp_duration
            ],
            ng.spike,
        )
        if (ng.network.iteration - 1) % ng.network.inp_duration == 0:
            ng.network.current_inp_idx = (
                1 + ng.network.current_inp_idx
            ) % self.number_of_data


class PoissonDistribution(Behavior):
    def initialize(self, ng):
        self.method = self.parameter("method", "poisson")
        self.input_time = int(
            self.parameter("time", required=True) / ng.network.dt
        )  # time to iteration
        self.rest_time = int(self.parameter("rest_time", 10) / ng.network.dt)
        self.data = torch.tensor(self.parameter("data", required=True))
        self.number_of_data = self.data.shape[0]
        self.size = self.data.shape[1]
        self.epsilon = self.parameter("epsilon", 0.001)
        max_data = self.data.max()
        min_data = self.data.min()
        self.range = self.parameter("range", None) or (min_data, max_data)

        ng.network.inp_duration = self.input_time + self.rest_time

        self.first_layer_input = []
        ng.network.current_inp_idx = 0

        for d in self.data:
            self.first_layer_input.append(self.encode_poisson(ng, d))

        ng.spike = ng.vector() != 0

    def encode_poisson(self, ng, d):
        data = copy.deepcopy(d)
        data = (data - self.range[0]) / (self.range[1] - self.range[0])
        # data = (data * (1 - self.epsilon)) + self.epsilon
        num_neurons = len(data)

        encode_matrix = np.zeros((self.size, ng.network.inp_duration)) != 0

        for i in range(num_neurons):
            spike_times = np.random.poisson(data[i], self.input_time)
            for j, t in enumerate(spike_times):
                if t > 0:
                    encode_matrix[i, j : t + j] = 1
        return torch.tensor(encode_matrix.T)

    def forward(self, ng):
        ng.spike = torch.logical_or(
            self.first_layer_input[ng.network.current_inp_idx][
                (ng.network.iteration - 1) % ng.network.inp_duration
            ],
            ng.spike,
        )
        if (ng.network.iteration - 1) % ng.network.inp_duration == 0:
            ng.network.current_inp_idx = (
                1 + ng.network.current_inp_idx
            ) % self.number_of_data


class NumericalCoding(Behavior):
    def initialize(self, ng):
        self.data = self.parameter("data", required=True)
        self.range = self.parameter("range", [0, 10])
        self.epsilon = self.parameter("epsilon", default=0.001)
        self.std_dev = self.parameter("std_dev", 1)
        self.time = int(
            (self.parameter("time", default=10) / ng.network.dt)
        )  # time to iteration number
        ng.network.input_period = int(
            self.parameter("input_period", default=1000000) / ng.network.dt
        )
        ng.spike = ng.vector(0) != 0
        ng.encoded_matrix = self.gen_encode_matrix(ng)
        print(ng.encoded_matrix)

    def forward(self, ng):
        ng.spike = torch.logical_or(
            (
                ng.encoded_matrix[(ng.network.iteration - 1) % ng.network.input_period]
                if ng.network.iteration % ng.network.input_period <= self.time
                else ng.vector(0) != 0
            ),
            ng.spike,
        )

    def gen_encode_matrix(self, ng):
        range_length = np.abs(self.range[0] - self.range[1])
        scaled_value = 10 * self.data / range_length  # -> [0,10]
        N = ng.size
        encoded_matrix = torch.zeros((self.time, N), dtype=torch.bool)
        prob_values = torch.tensor(
            [
                self.norm_prob(scaled_value, mean=i + 1, std_dev=self.std_dev)
                for i in range(N)
            ]
        )
        scaled_prob = prob_values / prob_values.max()

        for t in range(self.time):
            threshold = (self.time - t) / (self.time + 1)
            encoded_matrix[t, :] = scaled_prob >= threshold
            scaled_prob[scaled_prob >= threshold] = 0
        return encoded_matrix

    def norm_prob(self, value, mean=0, std_dev=1):

        # Calculate the probability density at x
        pdf_at_x = scipy.stats.norm.pdf(value, loc=mean, scale=std_dev)

        return pdf_at_x
