import os
import numpy as np
from gym import spaces
from typing import Any, List, Optional, Sequence, Tuple
from random import choice
# from skimage.transform import resize
import cv2

from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    RGBSensor,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat import logger
from habitat.core.dataset import Episode
from habitat.config import Config, read_write
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoalSensor
import quaternion
import lmdb
from PIL import Image
import random

import habitat
import habitat_sim
from habitat_sim import bindings as hsim
from habitat.utils.geometry_utils import (
    quaternion_from_coeff
)
from habitat_sim.utils.common import (
    quat_from_angle_axis,
    quat_from_coeffs,
    quat_to_coeffs,
    quat_to_angle_axis
)
from habitat.tasks.nav.instance_image_nav_task import (
    InstanceImageParameters
)
from habitat_sim.agent.agent import SixDOFPose
from habitat_sim.agent.agent import AgentState as hs_AgentState


@registry.register_sensor
class GibsonImageGoalFeatureSensor(Sensor):
    r"""Sensor for viewpoint to past ImageGoal observations which are used in ImageGoal Navigation.

    RGBSensor needs to be one of the Simulator sensors.
    This sensor return the rgb image taken from the goal position to reach with
    random rotation.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """
    cls_uuid: str = "gibsonimagegoalfeature_sensor"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                f"ImageGoalNav requires one RGB sensor, {len(rgb_sensor_uuids)} detected"
            )

        (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        self._current_episode_id: Optional[str] = None
        self._current_image_goal = None
        super().__init__(config=config)
        
        self.lmdb_env = lmdb.open(config.feat_path ,map_size=int(1e12),readonly=True, lock=False)
        self.eval = "eval" in config.feat_path
        

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_goal_views(self):
        return self._current_goal_views
    
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1024,),
            dtype=np.float32,
        )

    def get_observation(
        self,
        *args: Any,
        observations,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        # episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        # if episode_uniq_id == self._current_episode_id:
        #     return self._current_image_goal
        
        scene_name = episode.scene_id.split("/")[-1].split(".")[0]
        episode_id = episode.episode_id
 
        if self.eval:
            random_view = 0
        else:
            random_view = choice([0, 1, 2, 3])
        key = f'{scene_name}_{episode_id}_0_{random_view}'

        try:
            with self.lmdb_env.begin() as lmdb_txn:
                goal_feats = (np.frombuffer(lmdb_txn.get(key.encode()), dtype=np.float16)).astype(np.float32)
                source_rotation = (np.frombuffer(lmdb_txn.get((key+"_sr").encode()))).astype(np.float32)
        except:
            print(key)
            print(lmdb_txn.get(key.encode()))
        
        setattr(self,"_current_goal_views",[source_rotation.copy().tolist()])
        # setattr(self,"_current_goal_views",[source_rotation])
        
        # self._current_image_goal = goal_feats.copy()
        # self._current_episode_id = episode_uniq_id

        return goal_feats


@registry.register_sensor(name="ImageGoalSensorV2")
class ImageGoalSensorV2(Sensor):
    r"""Sensor for ImageGoal observations which are used in
    ImageGoal Navigation.

    RGBSensor needs to be one of the Simulator sensors.
    This sensor return the rgb image taken from the goal position to reach with
    random rotation.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """
    cls_uuid: str = "imagegoal_sensor_v2"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                "ImageGoalSensorV2 requires one RGB sensor, "
                f"{len(rgb_sensor_uuids)} detected"
            )

        # (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        self._current_scene_id = None
        self._current_episode_id = None
        self._current_image_goal = None
        self._sampling_type = getattr(config, 'sampling_type', 'uniform')
        channels = getattr(config, 'channels', ['rgb'])
        self.bias = getattr(config, 'bias', [0,0,0])
        self.augmentation = getattr(config, 'augmentation', None)
        self.return_view = getattr(config, 'return_view', False)
        logger.info(f"goal sensor bias: {self.bias}")

        if self.augmentation != None and self.augmentation.activate:
            self.aug_sensor_uuid = "aug_sensor"
            self._add_sensor(self.aug_sensor_uuid)

        if isinstance(channels, list):
            self._channels = channels
        elif isinstance(channels, str):
            # string with / to separate modalities
            self._channels = channels.split('/')
        else:
            raise ValueError(f'Unknown data type for channels!')

        self._channel2uuid = {}
        self._channel2range = {}
        self._shape = None
        self._current_goal_views = []
        self._setup_channels()
        self._set_space()
        super().__init__(config=config)

    def _get_sensor_uuid(self, sensor_type):
        sensors = self._sim.sensor_suite.sensors
        sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, sensor_type)
        ]
        if len(sensor_uuids) != 1:
            raise ValueError(
                f"ImageGoalSensorV2 requires one {sensor_type} sensor, "
                f"{len(sensor_uuids)} detected"
            )

        return sensor_uuids[0]

    def _setup_channels(self):
        self._channel2uuid = {}
        self._channel2range = {}
        last_idx = 0
        if 'rgb' in self._channels:
            self._channel2uuid['rgb'] = self._get_sensor_uuid(RGBSensor)
            self._channel2range['rgb'] = (last_idx, last_idx + 3)
            last_idx += 3

        if len(self._channel2uuid.keys()) == 0:
            raise ValueError('ImageGoalSensorV2 requires at least one channel')

    def _set_space(self):
        self._shape = None
        for k in self._channel2uuid.keys():
            uuid = self._channel2uuid[k]
            ospace = self._sim.sensor_suite.observation_spaces.spaces[uuid]
            if self._shape is None:
                self._shape = [ospace.shape[0], ospace.shape[1], 0]
            else:
                if ((self._shape[0] != ospace.shape[0]) or
                    (self._shape[1] != ospace.shape[1])):
                    raise ValueError('ImageGoalSensorV2 requires all '
                                     'base sensors to have the same with '
                                     'and hight, {uuid} has shape {ospace.shape}')

            if len(ospace.shape) == 3:
                self._shape[2] += ospace.shape[2]
            else:
                self._shape[2] += 1

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.iinfo(np.uint8).min,
            high=np.iinfo(np.uint8).max,
            shape=self._shape,
            dtype=np.uint8)

    def _get_image_goal_at(self, position, rotation):
        position = position + np.array(self.bias)
        obs = self._sim.get_observations_at(
            position=position, rotation=rotation)
        goal = []
        if 'rgb' in self._channel2uuid.keys():
            goal.append(obs[self._channel2uuid['rgb']].astype(
                self.observation_space.dtype))

        return np.concatenate(goal, axis=2)

    def _add_sensor(
        self, sensor_uuid: str
    ) -> None:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = sensor_uuid
        spec.sensor_type = habitat_sim.SensorType.COLOR
        spec.resolution = (512, 512)
        spec.hfov = 120
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self._sim.add_sensor(spec)

    def _get_augmented_image_goal(
        self, img_params: InstanceImageParameters
    ):
        if self.aug_sensor_uuid not in self._sim._sensors:
            self._add_sensor(self.aug_sensor_uuid)

        agent = self._sim.get_agent(0)
        agent_state = agent.get_state()
        agent.set_state(
            hs_AgentState(
                position=agent_state.position,
                rotation=agent_state.rotation,
                sensor_states={
                    **agent_state.sensor_states,
                    self.aug_sensor_uuid: SixDOFPose(
                        position=np.array(img_params.position),
                        rotation=quaternion_from_coeff(img_params.rotation),
                    ),
                },
            ),
            infer_sensor_states=False,
        )

        self._sim._sensors[self.aug_sensor_uuid].draw_observation()
        img = self._sim._sensors[self.aug_sensor_uuid].get_observation()[:, :, :3]

        return img

    def _hfov_center_crop(self, img, shfov, thfov):
        H,W,C = img.shape
        ratio = thfov / shfov
        s = max(int(H*(1-ratio)/2), 1)
        return cv2.resize(img[s:-s,s:-s,...], (W,H))

    def _get_episode_image_goal(self, episode: Episode):
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)
        self._current_goal_views = []
        view = []

        rng = None
        if self._sampling_type == 'random':
            rng = np.random.RandomState()
            angle = rng.uniform(0, 2 * np.pi)
            view = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        elif self._sampling_type == 'uniform':
            # to be sure that the rotation is the same for
            # the same episode_id
            # since the task is using pointnav Dataset.
            seed = abs(hash(episode.episode_id)) % (2 ** 32)
            rng = np.random.RandomState(seed)
            angle = rng.uniform(0, 2 * np.pi)
            view = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        
        source_rotation = np.array(view, dtype=np.float32).tolist()

        if self.augmentation != None and self.augmentation.activate:
            hfov = rng.randint(*self.augmentation.hfov)
            height = rng.uniform(*self.augmentation.height)
            pitch = rng.uniform(*self.augmentation.pitch)

            cam_goal_position = goal_position + np.array([0,1.25+height,0])
            cam_source_rotation = quat_to_coeffs(
                quat_from_coeffs(source_rotation) * \
                    quat_from_angle_axis(np.deg2rad(pitch), np.array([1,0,0])))

            img_params = InstanceImageParameters(
                position=cam_goal_position,
                rotation=cam_source_rotation,
                hfov=120,   # unused
                image_dimensions=(512,512) # unused
            )

            goal_observation = self._get_augmented_image_goal(
                img_params
            )
            goal_observation = self._hfov_center_crop(goal_observation, 120, hfov)

            # goal_observation = self._get_image_goal_at(
            #     (goal_position + np.array([0,height,0])).tolist(), 
            #     cam_source_rotation
            # )
        else:
            goal_observation = self._get_image_goal_at(
                goal_position.tolist(), source_rotation)
        
        self._current_goal_views.append(source_rotation)

        return goal_observation

    def get_goal_views(self):
        return self._current_goal_views

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ):
        if episode.scene_id != self._current_scene_id:
            self._current_scene_id = episode.scene_id

        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        self._current_image_goal = self._get_episode_image_goal(
            episode
        )
        self._current_episode_id = episode_uniq_id

        if self.return_view:
            return {
                "goal_image": self._current_image_goal, 
                "goal_view": self._current_goal_views[0]
            }
        else:
            return self._current_image_goal


@registry.register_sensor(name="PanoramicRGBSensor")
class PanoramicRGBSensor(Sensor):
    cls_uuid: str = "panoramic_rgb_sensor"
    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                "PanoramicRGBSensor requires one RGB sensor, "
                f"{len(rgb_sensor_uuids)} detected"
            )
        self._hfov = self._sim.sensor_suite.sensors[rgb_sensor_uuids[0]].config.hfov
        self._n_turn = 360 // self._hfov
        self._turn_angle = 360 // self._n_turn

        channels = getattr(config, 'channels', ['rgb'])
        if isinstance(channels, list):
            self._channels = channels
        elif isinstance(channels, str):
            # string with / to separate modalities
            self._channels = channels.split('/')
        else:
            raise ValueError(f'Unknown data type for channels!')
        
        self._channel2uuid = {}
        self._channel2range = {}
        self._space = None
        self._setup_channels()
        self._set_space()
        super().__init__(config=config)

    def _setup_channels(self):
        self._channel2uuid = {}
        self._channel2range = {}
        last_idx = 0
        if 'rgb' in self._channels:
            self._channel2uuid['rgb'] = self._get_sensor_uuid(RGBSensor)
            self._channel2range['rgb'] = (last_idx, last_idx + 3)
            last_idx += 3

        if len(self._channel2uuid.keys()) == 0:
            raise ValueError('PanoramicRGBSensor requires at least one channel')

    def _get_sensor_uuid(self, sensor_type):
        sensors = self._sim.sensor_suite.sensors
        sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, sensor_type)
        ]
        if len(sensor_uuids) != 1:
            raise ValueError(
                f"PanoramicRGBSensor requires one {sensor_type} sensor, "
                f"{len(sensor_uuids)} detected"
            )

        return sensor_uuids[0]

    def _set_space(self):
        self._shape = None
        for k in self._channel2uuid.keys():
            uuid = self._channel2uuid[k]
            ospace = self._sim.sensor_suite.observation_spaces.spaces[uuid]
            if self._shape is None:
                self._shape = [ospace.shape[0], ospace.shape[1], 0, 0]
            else:
                if ((self._shape[0] != ospace.shape[0]) or
                    (self._shape[1] != ospace.shape[1])):
                    raise ValueError('PanoramicRGBSensor requires all '
                                     'base sensors to have the same with '
                                     'and hight, {uuid} has shape {ospace.shape}')

            if len(ospace.shape) == 3:
                self._shape[2] += ospace.shape[2]
            else:
                self._shape[2] += 1
            
        self._shape[-1] = self._n_turn
    
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid
    
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=self._shape,
            dtype=np.float32)

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_panoramic_image_at(self, position, rotation):
        images = []
        r = rotation
        for i in range(self._n_turn):
            q = quaternion.from_euler_angles([0,np.deg2rad(-self._turn_angle),0])
            if i >= 1:
                r *= q
            obs = self._sim.get_observations_at(
                position=position, rotation=r)
            if 'rgb' in self._channel2uuid.keys():
                images.append(obs[self._channel2uuid['rgb']].astype(
                    self.observation_space.dtype))

        return np.stack(images, axis=3)

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        panoramic_image = self._get_panoramic_image_at(
            agent_state.position.tolist(), agent_state.rotation
        )

        return panoramic_image


@registry.register_sensor(name="QueriedImageSensor")
class QueriedImageSensor(Sensor):
    cls_uuid: str = "queried_image_sensor"
    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim

        # (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        self._current_scene_id = None
        self._current_episode_id = None
        self._current_image_goal = None
        self._sampling_type = getattr(config, 'sampling_type', 'uniform')
        channels = getattr(config, 'channels', ['rgb'])
        if isinstance(channels, list):
            self._channels = channels
        elif isinstance(channels, str):
            # string with / to separate modalities
            self._channels = channels.split('/')
        else:
            raise ValueError(f'Unknown data type for channels!')

        self._channel2uuid = {}
        self._channel2range = {}
        self._shape = None
        self._current_goal_views = []
        self._setup_channels()
        self._set_space()
        self._get_all_objects_images(config)
        super().__init__(config=config)
        
    def _get_all_objects_images(self, config):
        object_categories = ["bed", "chair", "couch", "potted-plant", "toilet", "tv"]
        self.object_images = {}
        for cate in object_categories:
            self.object_images[cate] = [
                cv2.resize(np.array(Image.open(os.path.join(config.image_root, "%s%02d.jpg" % (cate, i)))), (256, 256))
                for i in range(config.image_num)
            ]
    
    def _get_sensor_uuid(self, sensor_type):
        sensors = self._sim.sensor_suite.sensors
        sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, sensor_type)
        ]
        if len(sensor_uuids) != 1:
            raise ValueError(
                f"ImageGoalSensorV2 requires one {sensor_type} sensor, "
                f"{len(sensor_uuids)} detected"
            )

        return sensor_uuids[0]

    def _setup_channels(self):
        self._channel2uuid = {}
        self._channel2range = {}
        last_idx = 0
        if 'rgb' in self._channels:
            self._channel2uuid['rgb'] = self._get_sensor_uuid(RGBSensor)
            self._channel2range['rgb'] = (last_idx, last_idx + 3)
            last_idx += 3

        if len(self._channel2uuid.keys()) == 0:
            raise ValueError('ImageGoalSensorV2 requires at least one channel')

    def _set_space(self):
        self._shape = None
        for k in self._channel2uuid.keys():
            uuid = self._channel2uuid[k]
            ospace = self._sim.sensor_suite.observation_spaces.spaces[uuid]
            if self._shape is None:
                self._shape = [ospace.shape[0], ospace.shape[1], 0]
            else:
                if ((self._shape[0] != ospace.shape[0]) or
                    (self._shape[1] != ospace.shape[1])):
                    raise ValueError('ImageGoalSensorV2 requires all '
                                     'base sensors to have the same with '
                                     'and hight, {uuid} has shape {ospace.shape}')

            if len(ospace.shape) == 3:
                self._shape[2] += ospace.shape[2]
            else:
                self._shape[2] += 1

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=self._shape,
            dtype=np.float32)
    
    def _get_episode_image_goal(self, episode: Episode):
        object_goal = episode.object_category.replace(" ", "-")
        
        return random.choice(self.object_images[object_goal])
        
    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ):
        if episode.scene_id != self._current_scene_id:
            self._current_scene_id = episode.scene_id

        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        self._current_image_goal = self._get_episode_image_goal(
            episode
        )
        self._current_episode_id = episode_uniq_id

        return self._current_image_goal


@registry.register_sensor(name="EpisodeIDSensor")
class EpisodeIDSensor(Sensor):
    r'''Sensor for quering episode id
    
    '''
    cls_uuid: str = "episode_id_sensor"
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=np.iinfo(np.int64).max,
            shape=(1,),
            dtype=np.int64,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        return np.array([int(episode.episode_id)])


@registry.register_sensor(name="KeypointMatchingSensor")
class KeypointMatchingSensor(Sensor):
    r'''Sensor to calculate image keypoints
    
    '''
    cls_uuid: str = "goalimage_keypoint_sensor"
    def __init__(self, sim, config, **kwargs: Any):
        self.image_goal_sensor = registry.get_sensor(
            config.goal_sensor_config.type
        )(sim=sim, config=config.goal_sensor_config,)
        self.max_matched_pts = config.max_matched_pts
        self.lowes_threshold = config.lowes_threshold
        self.output_shape = self.max_matched_pts * 4
        self.descriptor = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        
        self._current_scene_id = None
        self._current_episode_id = None
        self._current_image_goal = None
        
        super().__init__(config=config)
        
    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH
    
    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=np.finfo(np.float32).max,
            shape=(self.output_shape,),
            dtype=np.float32,
        )
        
    def _get_origin_image_pt(self, matched, kp1, kp2):
        return kp1[matched.queryIdx].pt + kp2[matched.trainIdx].pt
        
    def _match_and_filter(self, kp1, kp2, des1, des2):
        if isinstance(des1, type(None)) or isinstance(des2, type(None)):
            return np.array([])
        elif len(kp1) < 2 or len(kp2) < 2:
            return np.array([])
    
        matches = self.matcher.knnMatch(
            des1.astype(np.uint8), 
            des2.astype(np.uint8), 
            k=2
        )
        
        good_match = []
        matched_points = []
        for m,n in matches:
            if m.distance < self.lowes_threshold * n.distance:
                good_match.append(m)
                matched_points.append(self._get_origin_image_pt(m, kp1, kp2))
        
        return np.asarray(matched_points)

    # This is called whenver reset is called or an action is taken
    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        if episode.scene_id != self._current_scene_id:
            self._current_scene_id = episode.scene_id

        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            kp_goal, des_goal = self._current_image_goal
        else:
            image_goal = self.image_goal_sensor.get_observation(
                observations=observations,
                episode=episode
            ).astype(np.uint8)
            kp_goal, des_goal = self.descriptor.detectAndCompute(image_goal, None)
            self._current_image_goal = (kp_goal, des_goal)
        
        rgb = observations["rgb"]
        kp_obs, des_obs = self.descriptor.detectAndCompute(rgb, None)
        
        matched_points: np.ndarray = self._match_and_filter(
            kp_goal, kp_obs, des_goal, des_obs
        )[:self.max_matched_pts].flatten()
        
        matched_point_features = -np.ones((self.output_shape,), dtype=np.float32)
        matched_point_features[:len(matched_points)] = matched_points
        
        return matched_point_features