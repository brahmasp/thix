import functools
import gymnasium as gym
import gymnasium_robotics
import embodied
from embodied.envs.from_gymnasium import FromGymnasium
import numpy as np
from PIL import Image
import os

gym.register_envs(gymnasium_robotics)

class FrankaKitchen(embodied.Env):
    def __init__(self, task, repeat=4, max_episode_steps=280, terminate=True, size=(64, 64)):
        if 'MUJOCO_GL' not in os.environ:
            os.environ['MUJOCO_GL'] = 'egl'
        task_list = task.split('_')
        self._env = FromGymnasium(gym.make('FrankaKitchen-v1', tasks_to_complete=task_list,
                                           terminate_on_tasks_completed=terminate,
                                           render_mode='rgb_array', max_episode_steps=max_episode_steps))
        self._env = embodied.wrappers.ActionRepeat(self._env, repeat)
        self._size = size

    @functools.cached_property
    def obs_space(self):
        obs_space = self._env.obs_space.copy()
        obs_space['image'] = embodied.Space(np.uint8, (*self._size, 3))
        return obs_space

    @functools.cached_property
    def act_space(self):
        return  self._env.act_space

    def step(self, action):
        obs = self._env.step(action)
        image = self._env.render()
        image = Image.fromarray(image)
        image = image.resize(self._size, Image.BILINEAR)
        obs['image'] = np.array(image)
        return obs