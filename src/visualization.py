from typing import Dict, List, Optional, Tuple
import numpy as np

from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import tile_images, draw_collision


def observations_to_image(observation: Dict, info: Dict) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    render_obs_images: List[np.ndarray] = []
    for sensor_name in observation:
        if len(observation[sensor_name].shape) > 1:
            obs_k = observation[sensor_name]
            if not isinstance(obs_k, np.ndarray):
                obs_k = obs_k.cpu().numpy()
            if obs_k.dtype != np.uint8:
                obs_k = obs_k * 255.0
                obs_k = obs_k.astype(np.uint8)
            if obs_k.shape[2] == 1:
                obs_k = np.concatenate([obs_k for _ in range(3)], axis=2)
            render_obs_images.append(obs_k)

    assert (
        len(render_obs_images) > 0
    ), "Expected at least one visual sensor enabled."

    shapes_are_equal = len(set(x.shape for x in render_obs_images)) == 1
    if not shapes_are_equal:
        render_frame = tile_images(render_obs_images)
    else:
        render_frame = np.concatenate(render_obs_images, axis=1)

    # draw collision
    if "collisions.is_collision" in info and info["collisions.is_collision"]:
        render_frame = draw_collision(render_frame)

    if "top_down_map.map" in info:
        info_top_down_map = {
            'map': info['top_down_map.map'],
            'fog_of_war_mask': info['top_down_map.fog_of_war_mask'],
            'agent_map_coord': info['top_down_map.agent_map_coord'],
            'agent_angle': info['top_down_map.agent_angle'],
        }
        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info_top_down_map, render_frame.shape[0]
        )
        render_frame = np.concatenate((render_frame, top_down_map), axis=1)
    return render_frame