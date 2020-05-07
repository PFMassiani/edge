import unittest

from edge.reward import ConstantReward
from edge.space import Box, StateActionSpace


class RewardTest(unittest.TestCase):
    def test_constant(self):
        space = StateActionSpace(*Box(0, 1, (10, 10)).sets)

        reward = ConstantReward(10)

        total = 0
        for t in range(10):
            s, a = space.get_tuple(space.sample())
            total += reward.get_reward(
                s,
                a,
                space.state_space.sample(),
                False
            )
        self.assertEqual(total, 100)

    def test_rewarded(self):
        space = StateActionSpace(*Box(0, 1, (10, 10)).sets)
        # `rewarded` should be a Subspace, but this is not implemented yet
        rewarded = StateActionSpace(*Box([0, 0], [0.5, 0.5], (10, 10)).sets)
        unrewarded = StateActionSpace(*Box([0.5, 0.5], [1, 1], (10, 10)).sets)

        reward = ConstantReward(10, rewarded_set=rewarded)

        total = 0
        for t in range(10):
            s, a = space.get_tuple(space.sample())
            sampled_space = rewarded.state_space if t % 2 == 0 \
                else unrewarded.state_space

            total += reward.get_reward(
                s,
                a,
                sampled_space.sample(),
                False
            )
        self.assertEqual(total, 50)

    def test_unrewarded(self):
        space = StateActionSpace(*Box(0, 1, (10, 10)).sets)
        # `rewarded` should be a Subspace, but this is not implemented yet
        rewarded = StateActionSpace(*Box([0, 0], [0.5, 0.5], (10, 10)).sets)
        unrewarded = StateActionSpace(*Box([0.5, 0.5], [1, 1], (10, 10)).sets)

        reward = ConstantReward(10, rewarded_set=unrewarded)

        total = 0
        for t in range(10):
            s, a = space.get_tuple(space.sample())
            sampled_space = rewarded.state_space if t % 2 == 0 \
                else unrewarded.state_space

            total += reward.get_reward(
                s,
                a,
                sampled_space.sample(),
                False
            )
        self.assertEqual(total, 50)

    def test_addition(self):
        space = StateActionSpace(*Box(0, 1, (10, 10)).sets)
        # `rewarded` should be a Subspace, but this is not implemented yet
        s = StateActionSpace(*Box([0, 0], [0.5, 0.5], (10, 10)).sets)

        r1 = ConstantReward(1, rewarded_set=s)
        r2 = ConstantReward(1, unrewarded_set=s)

        reward = r1 + r2

        total = 0
        for t in range(100):
            s, a = space.get_tuple(space.sample())
            total += reward.get_reward(
                s,
                a,
                space.state_space.sample(),
                False
            )
        self.assertEqual(total, 100)

    def test_substraction(self):
        space = StateActionSpace(*Box(0, 1, (10, 10)).sets)
        # `rewarded` should be a Subspace, but this is not implemented yet
        s = StateActionSpace(*Box([0, 0], [0.5, 0.5], (10, 10)).sets)

        r1 = ConstantReward(1, rewarded_set=s)
        r2 = ConstantReward(1, rewarded_set=s)

        reward = r1 - r2

        total = 0
        for t in range(100):
            s, a = space.get_tuple(space.sample())
            total += reward.get_reward(
                s,
                a,
                space.state_space.sample(),
                False
            )
        self.assertEqual(total, 0)
