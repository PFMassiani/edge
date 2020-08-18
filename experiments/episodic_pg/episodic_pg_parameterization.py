from edge.envs import Slip
from edge.reward import AffineReward, ConstantReward
from edge.agent import Agent
from edge.model.policy_models import GaussianPolicy


class LowGoalSlip(Slip):
    # * This goal incentivizes the agent to run fast
    def __init__(self, dynamics_parameters=None):
        super(LowGoalSlip, self).__init__(
            dynamics_parameters=dynamics_parameters,
            random_start=True
        )

        reward = AffineReward(self.stateaction_space, [(1, 0), (0, 0)])
        self.reward = reward


class PenalizedSlip(LowGoalSlip):
    def __init__(self, penalty_level=100, dynamics_parameters=None):
        super(PenalizedSlip, self).__init__(dynamics_parameters)

        def penalty_condition(state, action, new_state, reward):
            return self.is_failure_state(new_state)

        penalty = ConstantReward(self.reward.stateaction_space, -penalty_level,
                                 reward_condition=penalty_condition)

        self.reward += penalty


class PGOptimizer(Agent):
    def __init__(self, env, discount_rate, step_size, features_function, n_features, initial_weight, initial_var):
        policy = GaussianPolicy(
            env.stateaction_space,
            discount_rate,
            step_size,
            features_function,
            n_features,
            initial_weight,
            initial_var
        )
        super(PGOptimizer, self).__init__(env, policy)
        self.policy = policy

    def get_next_action(self):
        return self.policy.get_action(self.state)

    def update_models(self, episode):
        self.policy.update(episode)

    def fit_models(self):
        pass

    def step(self):
        """
        Chooses an action according to the policy, takes a step in the Environment, and updates the models. The action
        taken is available in self.last_action.
        :return: new_state, reward, failed
        """
        old_state = self.state
        action = self.get_next_action()
        new_state, reward, failed = self.env.step(action)
        self.state = new_state
        self.last_action = action
        return new_state, reward, failed
