"""Ibaba Life-saver drone environment package."""

from gymnasium.envs.registration import register

from .custom_env import ACTIONS, OBS_DIM, IbabaDroneEnv, heuristic_action

register(id="IbabaDrone-v0", entry_point="environment.custom_env:IbabaDroneEnv")

__all__ = ["ACTIONS", "OBS_DIM", "IbabaDroneEnv", "heuristic_action"]
