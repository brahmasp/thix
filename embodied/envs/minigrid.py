import gymnasium
from minigrid.wrappers import (FullyObsWrapper, ObservationWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper,
                               ImgObsWrapper)
import embodied
from embodied.envs.from_gymnasium import FromGymnasium
from PIL import Image
import numpy as np

class ResizeObservation(ObservationWrapper):
    def __init__(self, env, size=(64, 64)):
        super().__init__(env)
        self.env = env
        self.size = size

    def observation(self, obs):
        image = obs['image'] if isinstance(obs, dict) else obs
        img = Image.fromarray(image)
        img = img.resize(self.size, Image.Resampling.BILINEAR)
        if isinstance(obs, dict):
            obs['image'] = np.array(img)
            return obs
        else:
            return np.array(img)


class Minigrid(FromGymnasium):
    def __init__(self, task: str, rgb: bool = True, fully_observable: bool = False, hide_mission: bool = False, size=(64, 64)):
        env = gymnasium.make(f"MiniGrid-{task}-v0", render_mode="rgb_array")
        if fully_observable:
            if rgb:
                env = RGBImgObsWrapper(env, tile_size=8)
            else:
                env = FullyObsWrapper(env)
        else:
            if rgb:
                env = RGBImgPartialObsWrapper(env, tile_size=8)
        if hide_mission:
            env = ImgObsWrapper(env)
        # Apply resize wrapper last
        env = ResizeObservation(env, size=size)
        super().__init__(env=env)

    @property
    def obs_space(self):
        obs_space = super().obs_space
        obs_space.pop('image')
        obs_space['image'] = embodied.Space(np.uint8, (64, 64, 3))
        return obs_space