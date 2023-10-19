from email.policy import default
from nis import match
from typing import (
    Dict,
    Iterable,
    Optional,
    Union,
    List,
    Tuple
)
import copy
import torch
import numpy as np
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F
from transformers import CLIPModel,CLIPProcessor

from habitat import logger
from habitat.config import Config
from habitat.tasks.nav.nav import (
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
)
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.models.simple_cnn import SimpleCNN
from habitat_baselines.rl.ppo.policy import NetPolicy, Net, Policy
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.utils.common import get_num_actions
from habitat_baselines.common.baseline_registry import baseline_registry

from habitat.tasks.nav.nav import (
    EpisodicGPSSensor,
    EpisodicCompassSensor,
)
from habitat.tasks.nav.instance_image_nav_task import (
    InstanceImageGoalSensor,
    InstanceImageGoalHFOVSensor
)
from src.sensor import (
    GibsonImageGoalFeatureSensor,
    ImageGoalSensorV2,
    QueriedImageSensor,
    KeypointMatchingSensor
)
from habitat_baselines.rl.ddppo.policy import resnet
from src import resnet as fast_resnet
from src.transforms import (
    ResizeCenterCropper,
    ResizeRandomCropper,
)
from src.models import EarlyFuseCNNEncoder
# from superglue.models.matching import Matching
import clip
import torchvision.transforms.functional as F
import cv2
import time


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


@baseline_registry.register_policy
class NavNetPolicy(NetPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size=512,
        num_recurrent_layers=2,
        rnn_type="GRU",
        resnet_baseplanes=32,
        backbone="resnet18",
        goal_backbone="clip",
        normalize_visual_inputs=False,
        obs_transform=ResizeCenterCropper(size=(256, 256)),  # noqa : B008
        force_blind_policy=False,
        visual_encoder_embedding_size=512,
        visual_obs_inputs=['*'],
        visual_encoder_init=None,
        **kwargs
    ):
        super().__init__(
            NavNet(
                observation_space=observation_space,
                action_space=action_space,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                goal_backbone=goal_backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                obs_transform=obs_transform,
                force_blind_policy=force_blind_policy,
                visual_encoder_embedding_size=visual_encoder_embedding_size,
                visual_obs_inputs=visual_obs_inputs,
                visual_encoder_init=visual_encoder_init,
                **kwargs
            ),
            action_space,
        )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):

        features, rnn_hidden_states, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    @classmethod
    def from_config(
        cls,
        config: Config,
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
    # def from_config(cls, config, envs):
        ppo_config = config.habitat_baselines.rl.ppo
        if ppo_config.random_crop:
            obs_transform = ResizeRandomCropper(
                size=(ppo_config.input_size, ppo_config.input_size))
        else:
            obs_transform = ResizeCenterCropper(
                size=(ppo_config.input_size, ppo_config.input_size))

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=ppo_config.hidden_size,
            rnn_type=ppo_config.rnn_type,
            num_recurrent_layers=ppo_config.num_recurrent_layers,
            backbone=ppo_config.backbone,
            goal_backbone=ppo_config.goal_backbone,
            resnet_baseplanes=32,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            obs_transform=obs_transform,
            force_blind_policy=False,
            visual_encoder_embedding_size=ppo_config.visual_encoder_embedding_size,
            visual_obs_inputs=ppo_config.visual_obs_inputs,
            visual_encoder_init=None,
            rgb_color_jitter=ppo_config.rgb_color_jitter,
            tie_inputs_and_goal_param=ppo_config.tie_inputs_and_goal_param,
            goal_embedding_size=ppo_config.goal_embedding_size,
            task_type_embed=ppo_config.task_type_embed,
            task_type_embed_size=ppo_config.task_type_embed_size,
            enable_feature_matching=ppo_config.enable_feature_matching,
            cam_visual=config.habitat_baselines.eval.cam_visual,
            film_reduction=ppo_config.film_reduction,
            film_layers=ppo_config.film_layers,
        )


class NavNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        num_recurrent_layers,
        rnn_type,
        backbone,
        goal_backbone,
        resnet_baseplanes,
        normalize_visual_inputs,
        obs_transform=ResizeCenterCropper(size=(256, 256)),  # noqa: B008
        force_blind_policy=False,
        visual_encoder_embedding_size=512,
        visual_obs_inputs=['*'],
        visual_encoder_init=None,
        rgb_color_jitter=0.,
        tie_inputs_and_goal_param=False,
        goal_embedding_size=128,
        enable_feature_matching=False,
        cam_visual=False,
        film_reduction="none",
        film_layers=[0,1,2,3],
        **kwargs
    ):
        super().__init__()

        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action
        ObsEncoder = EarlyFuseCNNEncoder

        logger.info('Type of observation encoder: {}'.format(ObsEncoder))
        tied_param = {}

        # construct goal encoder
        
        logger.info(f'Create goal encoder : {goal_backbone}')
        if GibsonImageGoalFeatureSensor.cls_uuid in observation_space.spaces:
            self._goal_sensor_uuid = GibsonImageGoalFeatureSensor.cls_uuid
        elif ImageGoalSensorV2.cls_uuid in observation_space.spaces:
            self._goal_sensor_uuid = ImageGoalSensorV2.cls_uuid
        elif QueriedImageSensor.cls_uuid in observation_space.spaces:
            self._goal_sensor_uuid = QueriedImageSensor.cls_uuid
        elif InstanceImageGoalSensor.cls_uuid in observation_space.spaces:
            self._goal_sensor_uuid = InstanceImageGoalSensor.cls_uuid
        else :
            raise ValueError(f"Wrong goal sensor")

        if enable_feature_matching:
            self.matching_module = FeatureMatchingModule(
                visual_obs_inputs=visual_obs_inputs,
                goal_sensor=self._goal_sensor_uuid
            )
            self.matched_feature_fc = nn.Sequential(
                Flatten(),
                nn.Linear(
                    self.matching_module.output_shape,
                    64
                ),
                nn.ReLU(True),
            )
            rnn_input_size += 64
        else:
            self.matching_module = None
            
        goal_observation_space = spaces.Dict(
            {"rgb": observation_space.spaces[self._goal_sensor_uuid]} if "feature" not in self._goal_sensor_uuid else {"feature": observation_space.spaces[self._goal_sensor_uuid]}
        )

        if goal_backbone != "none":
            self.goal_visual_encoder = ObsEncoder(
                goal_observation_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                backbone=backbone if goal_backbone == "tied" else goal_backbone,
                normalize_visual_inputs=normalize_visual_inputs,
                obs_transform=obs_transform,
                visual_encoder_embedding_size=visual_encoder_embedding_size,
                visual_obs_inputs=["rgb"] if "feature" not in self._goal_sensor_uuid else ['feature'],
                visual_encoder_init=visual_encoder_init,
                rgb_color_jitter=rgb_color_jitter,
            )

            if goal_backbone == "tied":
                logger.warning("Tied the goal encoder's parameters with visual encoder!")
                tied_param = {self.goal_visual_encoder.encoder, self.goal_visual_encoder.v_output_shape}
            else:
                tied_param = {}

            self.goal_visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(
                    np.prod(self.goal_visual_encoder.output_shape),
                    goal_embedding_size
                ),
                nn.ReLU(True),
            )

            rnn_input_size += goal_embedding_size
        else:
            self.goal_visual_encoder = None

        self._hidden_size = hidden_size

        logger.info('Create visual inputs encoder')
        visual_observation_space = copy.deepcopy(observation_space)
        if goal_backbone != "none":
            visual_observation_space.spaces.pop(self._goal_sensor_uuid)
        self.visual_encoder = ObsEncoder(
            visual_observation_space if not force_blind_policy else spaces.Dict({}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            obs_transform=obs_transform,
            visual_encoder_embedding_size=visual_encoder_embedding_size,
            visual_obs_inputs=visual_obs_inputs,
            visual_encoder_init=visual_encoder_init,
            rgb_color_jitter=rgb_color_jitter,
            tied_params=tied_param if tie_inputs_and_goal_param else None,
            cam_visual=cam_visual,
            film_reduction=film_reduction,
            film_layers=film_layers,
        )

        if not self.visual_encoder.is_blind:
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape), hidden_size
                ),
                nn.ReLU(True),
            )

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def perception_embedding_size(self):
        self._hidden_size

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _format_pose(self, pose):
        """
        Args:
            pose: (N, 4) Tensor containing x, y, heading, time
        """

        x, y, theta, time = torch.unbind(pose, dim=1)
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        e_time = torch.exp(-time)
        formatted_pose = torch.stack([x, y, cos_theta, sin_theta, e_time], 1)

        return formatted_pose

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, rnn_build_seq_info=None):
        x = []
        aux_loss_state = {}
        # forward visual encoder
        if not self.is_blind:
            if "visual_features" in observations:
                visual_feats = observations["visual_features"]
            else:
                visual_feats, cam_visual = self.visual_encoder(observations)
                if not isinstance(cam_visual, type(None)):
                    observations['cam_visual'] = cam_visual

            visual_feats = self.visual_fc(visual_feats)

            x.append(visual_feats)
        
        # forward goal encoder
        if self.goal_visual_encoder != None:
            goal_image = observations[self._goal_sensor_uuid]
            goal_output, _ = self.goal_visual_encoder({"rgb": goal_image})
            x.append(self.goal_visual_fc(goal_output))

        # forward matching module
        if self.matching_module != None:
            matched_features = self.matching_module(observations)
            x.append(self.matched_feature_fc(matched_features))

        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().squeeze(dim=-1)
        )
        x.append(prev_actions)

        x = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks, rnn_build_seq_info)
        aux_loss_state["rnn_output"] = out

        return out, rnn_hidden_states, aux_loss_state


class FeatureMatchingModule(nn.Module):
    def __init__(
            self, 
            visual_obs_inputs, 
            goal_sensor, 
            conf_threshold=0.8,
            max_matched_pts=128,
        ) -> None:
        super().__init__()
        
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': -1
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        self.matching = Matching(config).eval()
        
        self.goal_sensor = goal_sensor
        self.visual_obs_inputs = visual_obs_inputs
        self.conf_threshold = conf_threshold
        # self.lowes_threshold = lowes_threshold
        self.max_matched_pts = max_matched_pts
        self.output_shape = max_matched_pts * 4
        self.current_episode_ids = {}
        self.goal_image_features = {}
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    def __call__(self, observations):
        # we only compare rgb image with one goal image
        assert len(self.visual_obs_inputs) == 2
        # require episode id to determine when to recaluculate feature for goal image
        # assert "episode_id_sensor" in observations
        
        # episode_ids = observations["episode_id_sensor"].cpu().numpy()
        
        # matched_point_features = []
        
        with torch.no_grad():
            self.matching.eval()
            
            ts = time.time()
            
            target_tensor = F.rgb_to_grayscale(observations[self.goal_sensor].permute(0,3,1,2)).float() / 255
            source_tensor = F.rgb_to_grayscale(observations['rgb'].permute(0,3,1,2)).float() / 255

            pred = {'image0': target_tensor, 'image1': source_tensor}

            pred = self.matching(pred)
            
            output = []
            for i in range(target_tensor.shape[0]):
                kpts0 = pred['keypoints0'][i]
                kpts1 = pred['keypoints1'][i]
                matches = pred['matches0'][i]
                confidence = pred['matching_scores0'][i] #.cpu().numpy()
                
                valid = torch.bitwise_and(matches > -1, confidence > self.conf_threshold)
                sorted = torch.argsort(confidence[valid], descending=True)[:self.max_matched_pts]
                mkpts0 = kpts0[valid][sorted]
                mkpts1 = kpts1[matches[valid][sorted].long()]
                
                matched_points = torch.cat((mkpts0, mkpts1), dim=1).flatten() / target_tensor.shape[-1]
                matched_point_features = -torch.ones((self.output_shape,), dtype=torch.float32, device=self.dummy_param.device)
                matched_point_features[:matched_points.shape[0]] = matched_points
                output.append(matched_point_features) 
            
            tu = time.time() - ts
                
        return torch.stack(output, dim=0)
