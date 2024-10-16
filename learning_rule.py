from pymonntorch import Behavior
import torch


class STDP(Behavior):
    def initialize(self, synapse):
        self.positive_tau = self.parameter("positive_tau", 1)
        self.negative_tau = self.parameter("negative_tau", 1)
        self.A_negative = self.parameter("A_negative", 0) / synapse.C
        self.A_positive = self.parameter("A_positive", 0) / synapse.C
        self.normalize = self.parameter("normalize", True)
        self.high_limit = (
            self.parameter("high_limit", 0) or (synapse.J0 / synapse.C) * 10
        )
        self.low_limit = (
            self.parameter("low_limit", 0) or (-synapse.J0 / synapse.C) * 10
        )

        synapse.x_pre = synapse.src.vector()
        synapse.y_post = synapse.dst.vector()

    def forward(self, synapse):
        d_x_pre = (
            -(synapse.x_pre / self.positive_tau) + synapse.src.spike.byte()
        ) * synapse.network.dt

        d_y_post = (
            -(synapse.y_post / self.negative_tau) + synapse.dst.spike.byte()
        ) * synapse.network.dt

        d_W = (
            (
                self.A_negative
                * torch.mm(
                    synapse.src.spike.byte().to(torch.float64).reshape(-1, 1),
                    synapse.y_post.to(torch.float64).reshape(1, -1),
                )
            )
            * ((-synapse.W + self.low_limit) * abs(synapse.W))
            + (
                self.A_positive
                * torch.mm(
                    synapse.x_pre.to(torch.float64).reshape(-1, 1),
                    synapse.dst.spike.byte().to(torch.float64).reshape(1, -1),
                )
            )
            * (self.high_limit - synapse.W)
            * abs(synapse.W)
        ) * synapse.network.dt

        if self.normalize:
            d_W -= d_W.sum(axis=0) / synapse.W.shape[0]
        synapse.W += d_W

        synapse.x_pre += d_x_pre
        synapse.y_post += d_y_post

        if synapse.network.iteration % synapse.network.inp_duration == 0:
            synapse.x_pre = synapse.src.vector()
            synapse.y_post = synapse.dst.vector()
            synapse.src.u = synapse.src.vector(synapse.src.u_rest)
            synapse.dst.u = synapse.dst.vector(synapse.dst.u_rest)


class RSTDP(Behavior):
    def initialize(self, synapse):
        self.positive_tau = self.parameter("positive_tau", 1)
        self.negative_tau = self.parameter("negative_tau", 1)
        self.A_negative = self.parameter("A_negative", 0) / synapse.C
        self.A_positive = self.parameter("A_positive", 0) / synapse.C
        self.normalize = self.parameter("normalize", True)
        self.high_limit = self.parameter("high_limit", 0) or (
            (synapse.J0 / synapse.C) * 10
        )
        self.low_limit = self.parameter("low_limit", 0) or (
            (-synapse.J0 / synapse.C) * 10
        )
        self.reward = self.parameter("reward", 10)
        self.punishment = self.parameter("punishment", 3)

        synapse.dopamin = synapse.dst.vector()
        synapse.x_pre = synapse.src.vector()
        synapse.y_post = synapse.dst.vector()
        synapse.C_RSTDP = synapse.matrix()

    def forward(self, synapse):
        d_x_pre = (
            -(synapse.x_pre / self.positive_tau) + synapse.src.spike.byte()
        ) * synapse.network.dt

        d_y_post = (
            -(synapse.y_post / self.negative_tau) + synapse.dst.spike.byte()
        ) * synapse.network.dt

        d_W = (
            -(
                self.A_negative
                * torch.mm(
                    synapse.src.spike.byte().to(torch.float64).reshape(-1, 1),
                    synapse.y_post.to(torch.float64).reshape(1, -1),
                )
            )
            + (
                self.A_positive
                * torch.mm(
                    synapse.x_pre.to(torch.float64).reshape(-1, 1),
                    synapse.dst.spike.byte().to(torch.float64).reshape(1, -1),
                )
            )
        ) * synapse.network.dt

        synapse.x_pre += d_x_pre
        synapse.y_post += d_y_post
        synapse.C_RSTDP += d_W

        if synapse.network.iteration % synapse.network.inp_duration == 0:
            max_num_of_spikes = synapse.dst.num_of_each_nueron_spikes.max()
            result = synapse.dst.num_of_each_nueron_spikes == max_num_of_spikes
            synapse.dopamin = self.punishment * result.byte().to(torch.float64)
            synapse.dopamin[synapse.network.current_inp_idx] *= (self.reward or 1) / (
                self.punishment or 1
            )
            synapse.C_RSTDP /= abs(synapse.C_RSTDP).max()
            synapse.dopamin = torch.diag(synapse.dopamin)
            d_W = torch.mm(
                synapse.C_RSTDP.to(torch.float64), synapse.dopamin.to(torch.float64)
            )
            d_W *= abs(
                (self.high_limit - synapse.W) * (self.low_limit - synapse.W) * synapse.W
            )
            if self.normalize:
                d_W -= d_W.sum(axis=0) / synapse.W.shape[0]

            synapse.W += d_W

            synapse.C_RSTDP = synapse.matrix()
            synapse.dst.num_of_each_nueron_spikes = synapse.dst.vector()
            synapse.x_pre = synapse.src.vector()
            synapse.y_post = synapse.dst.vector()
            synapse.src.u = synapse.src.vector(synapse.src.u_rest)
            synapse.dst.u = synapse.dst.vector(synapse.dst.u_rest)
