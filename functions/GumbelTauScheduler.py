class GumbelTauScheduler:
    def __init__(self, initial_tau=1.0, min_tau=0.1, decay_rate=0.95, decay_step=1):
        self.tau = initial_tau
        self.min_tau = min_tau
        self.decay_rate = decay_rate # 0.95 = 5% decrease per step
        self.decay_step = decay_step
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch % self.decay_step == 0:
            self.tau = max(self.min_tau, self.tau * self.decay_rate)

    def get_tau(self):
        return self.tau