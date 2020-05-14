from .. import GPModel


class SafetyMeasure(GPModel):
    def __init__(self, env, gp):
        super(SafetyMeasure, self).__init__(self, env, gp)

    def update(self, state, action, new_state, reward, failed):
        if not failed:
            update_value = self.measure(new_state)
        else:
            update_value = np.array([0.])

        stateaction = self.env.stateaction_space[state, action]
        self.gp.append_data(stateaction, update_value)

    def _query(self, x, return_covar=False):
        prediction = self.gp.predict(x)
        mean = prediction.mean.numpy()
        if return_covar:
            return mean, prediction.covar.numpy()
        else:
            return mean

    def measure(self, state):
        measure_slice = self[state, :]
        return np.atleast_1d(measure_slice.mean(axis=-1))

    def level_set(self, state, lambda_threshold, gamma_threshold):
        query = self.env.stateaction_space[state, :]

        measure_slice, covar_slice = self.query(query, return_covar=True)
        level_set = norm.cdf(
            (measure_slice - lambda_threshold) / np.sqrt(covar_slice)
        )

        in_level_set = level_set > gamma_threshold
        level_set[not in_level_set] = 0
        level_set[in_level_set] = 1
        return level_set
