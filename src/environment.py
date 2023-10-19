from typing import Optional

import gym
import habitat
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.utils.gym_adapter import HabGymWrapper


class NavRLEnvX(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._config = config
        # self._core_env_config = config.TASK_CONFIG
        if len(self._config.task.reward_measure) > 0:
            self._reward_measure_names = [self._config.task.reward_measure]
            self._reward_scales = [1.0]
        else:
            self._reward_measure_names = self._config.task.reward_measure
            self._reward_scales = self._config.task.reward_scales

        self._success_measure_name = self._config.task.success_measure
        habitat.logger.info('NavRLEnvX: '
                            f'Reward Measures={self._reward_measure_names}, '
                            f'Reward Scales={self._reward_scales}, '
                            f'Success Measure={self._success_measure_name}')
        self._previous_measure = None
        self._previous_action = None
        super().__init__(self._config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._get_reward_measure()
        return observations

    def _get_reward_measure(self):
        current_measure = 0.0
        for reward_measure_name, reward_scale in zip(
                self._reward_measure_names, self._reward_scales
        ):
            if "." in reward_measure_name:
                reward_measure_name = reward_measure_name.split('.')
                measure = self._env.get_metrics()[
                    reward_measure_name[0]
                ][reward_measure_name[1]]
            else:
                measure = self._env.get_metrics()[reward_measure_name]
            current_measure += measure * reward_scale
        return current_measure

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._config.task.slack_reward - 1.0,
            self._config.task.success_reward + 1.0,
        )

    def get_reward(self, observations):
        reward = self._config.task.slack_reward

        current_measure = self._get_reward_measure()

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._episode_success() * self._config.task.success_reward

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


@habitat.registry.register_env(name="GymHabitatEnvX")
class GymHabitatEnvX(gym.Wrapper):
    """
    A registered environment that wraps a RLTaskEnv with the HabGymWrapper
    to use the default gym API.
    """

    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        base_env = NavRLEnvX(config=config, dataset=dataset)
        env = HabGymWrapper(base_env)
        super().__init__(env)
