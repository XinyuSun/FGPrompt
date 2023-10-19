import torch
from torch import nn as nn
from torchvision.transforms import ColorJitter

from habitat import logger

from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)

from habitat_sim.utils.common import d3_40_colors_rgb

from habitat_baselines.rl.models.simple_cnn import SimpleCNN
from src import resnet as fast_resnet
from src import midfusion as midfusion_resnet
from src.transforms import ResizeCenterCropper
import clip
from torchvision import transforms
import PIL
from einops import rearrange
import numpy as np
import cv2

from pytorch_grad_cam import EigenCAM

VISUAL_SENSORS_UUID = [
    'rgb', 'depth', 'semantic', 'feature', 'imagegoal_sensor_v2', 'panoramic_rgb_sensor', 'queried_image_sensor', 'instance_imagegoal'
]

MOD_SEP = '/'


def make_resnet_encoder(make_backbone, n_input_channels, baseplanes, ngroups,
                        spatial_size=128, normalize_inputs=False,
                        after_compression_flat_size=2048, film_reduction=None, film_layers=None):
    modules = []
    if normalize_inputs:
        modules.append(RunningMeanAndVar(n_input_channels))

    if film_reduction != None:
        backbone = make_backbone(n_input_channels, baseplanes, ngroups, film_reduction, film_layers)
    else:
        backbone = make_backbone(n_input_channels, baseplanes, ngroups)
    modules.append(backbone)
    final_spatial = int(spatial_size * backbone.final_spatial_compress)
    num_compression_channels = int(
        round(after_compression_flat_size / (final_spatial ** 2))
    )
    compression = nn.Sequential(
        nn.Conv2d(
            backbone.final_channels,
            num_compression_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.GroupNorm(1, num_compression_channels),
        nn.ReLU(True),
    )
    modules.append(compression)
    output_shape = (
        num_compression_channels,
        final_spatial,
        final_spatial,
    )
    return nn.Sequential(*modules), output_shape


def make_simplecnn_encoder(n_input_channels, spatial_size=128,
                           normalize_inputs=False, output_size=2048):
    input_shape = (spatial_size, spatial_size, n_input_channels)
    output_shape = (output_size, 1, 1)
    modules = []
    if normalize_inputs:
        modules.append(RunningMeanAndVar(n_input_channels))

    backbone = SimpleCNN(input_shape, output_size, output_shape)
    modules.append(backbone)

    return nn.Sequential(*modules), output_shape


class EarlyFuseCNNEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        baseplanes=32,
        ngroups=32,
        spatial_size=128,
        backbone=None,
        normalize_visual_inputs=False,
        obs_transform=ResizeCenterCropper(size=(256, 256)),  # noqa: B008
        visual_encoder_embedding_size=512,
        visual_obs_inputs=['*'],
        visual_encoder_init=None,
        rgb_color_jitter=0.,
        tied_params=None,
        cam_visual=False,
        film_reduction='none',
        film_layers=[0,1,2,3],
    ):
        super().__init__()
        
        self.obs_transform = obs_transform
        self.cam_visual = cam_visual
        if self.obs_transform is not None and not backbone == "clip_feat":
            logger.info(f'use obs_transform: {type(self.obs_transform)}')
            observation_space = self.obs_transform.transform_observation_space(
                observation_space, trans_keys=VISUAL_SENSORS_UUID
            )
        self._n_input = {k: 0 for k in VISUAL_SENSORS_UUID}
        logger.info('observation_space kyes: {}'
                    .format(observation_space.spaces.keys()))
        logger.info('visual_obs_inputs: {}'.format(visual_obs_inputs))
        visual_uuid = VISUAL_SENSORS_UUID
        if ((visual_obs_inputs is not None) and (len(visual_obs_inputs) > 0) and (visual_obs_inputs[0] != '*')):
            if isinstance(visual_obs_inputs, list):
                visual_uuid = visual_obs_inputs
            else:
                visual_uuid = visual_obs_inputs.split(MOD_SEP)

        self._id2rgb = {k: False for k in VISUAL_SENSORS_UUID}
        self._colormap = torch.from_numpy(d3_40_colors_rgb)
        self._rgb_aug = None
        if "rgb" in visual_uuid or "imagegoal_sensor_v2" in visual_uuid and not backbone == "clip_feat":
            if rgb_color_jitter > 0:
                logger.info(f'use RGB color jitter= {rgb_color_jitter}')
                self._rgb_aug = ColorJitter(
                    brightness=rgb_color_jitter, contrast=rgb_color_jitter,
                    saturation=rgb_color_jitter, hue=rgb_color_jitter)

        num_input_channels = 0
        for v_uuid in visual_uuid:
            if v_uuid in observation_space.spaces:
                logger.info("{} observation_space: {}"
                            .format(v_uuid, observation_space.spaces[v_uuid]))
                
                if v_uuid == "semantic":
                    self._n_input[v_uuid] = 1
                elif v_uuid == "feature":
                    self._n_input[v_uuid] = observation_space.spaces[v_uuid].shape[0]

                if v_uuid == "rgb":
                    spatial_size = observation_space.spaces[v_uuid].shape[0]
                
                if v_uuid in [
                    "rgb", "imagegoal_sensor_v2", \
                        "queried_image_sensor", \
                        "instance_imagegoal"]:
                    self._n_input[v_uuid] = \
                            observation_space.spaces[v_uuid].shape[2]

                self._n_ouput_concat = 1
                if v_uuid == "panoramic_rgb_sensor":
                    self._n_ouput_concat = observation_space.spaces[v_uuid].shape[-1]

                num_input_channels += self._n_input[v_uuid]

        
        self._backbone = backbone
        if backbone == "clip":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model_encoder, _ = clip.load("RN50", device=device)
            self.clip_preprocessor = transforms.Compose([
                transforms.Resize(224, interpolation=PIL.Image.BICUBIC),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
            # self.clip_preprocessor = transforms.Compose([transforms.ToPILImage(),  # tensor[CHW]  
            #                                             clip_preprocessor        
            #                                 ])
            self.clip_model_encoder.eval()
            self.output_shape = 1024
        elif backbone == "clip_feat":
            self.output_shape = 1024
        else:    
            if not self.is_blind:
                if tied_params is not None:
                    logger.info('encoder is tied to goal params')
                    self.encoder = tied_params[0]
                    v_output_shape = tied_params[1]
                else:
                    if 'simple_cnn' in backbone:
                        self.encoder, v_output_shape = \
                            make_simplecnn_encoder(
                                num_input_channels,
                                spatial_size,
                                normalize_visual_inputs,
                                visual_encoder_embedding_size
                            )
                    else:
                        _film_reduction = None
                        _film_layers = None
                        if 'fast_resnet' in backbone:
                            restnet_type = backbone[len('fast_'):]
                            backbone_enc = getattr(fast_resnet, restnet_type)
                        elif 'midfusion_resnet' in backbone:
                            restnet_type = backbone[len('midfusion_'):]
                            backbone_enc = getattr(midfusion_resnet, restnet_type)
                            _film_reduction = film_reduction
                            _film_layers = film_layers
                        elif 'resnet' in backbone:
                            backbone_enc = getattr(resnet, backbone)
                        else:
                            raise ValueError('unknown type of backbone {}'
                                            .format(backbone))

                        self.encoder, v_output_shape = \
                            make_resnet_encoder(
                                backbone_enc,
                                num_input_channels,
                                baseplanes,
                                ngroups,
                                spatial_size,
                                normalize_visual_inputs,
                                visual_encoder_embedding_size,
                                _film_reduction,
                                _film_layers,
                            )

                num_compression_channels = v_output_shape[0]
                final_spatial = v_output_shape[1]
                logger.info('early fuse encoder(type: {}, in: {}, out: {})'.format(
                    backbone,
                    (num_input_channels, spatial_size, spatial_size),
                    v_output_shape)
                )

                self.output_shape = (
                    num_compression_channels * self._n_ouput_concat,
                    final_spatial,
                    final_spatial,
                )
                self.v_output_shape = v_output_shape

    @property
    def is_blind(self):
        n_inputs = 0
        for v in self._n_input.values():
            n_inputs += v
        return n_inputs == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def _rgb_augment(self, rgb_obs, seeds=None):
        rgb_3 = None
        if rgb_obs.size(1) == 3:
            rgb_3 = rgb_obs
        elif rgb_obs.size(1) == 4:
            rgb_3 = self._rgb_aug(rgb_obs[:, :3, :, :])
        else:
            raise NotImplementedError(
                f"{rgb_obs.size(1)} channles are detected, "
                f"only 3 or 4 is currently supported for rgb_aug")

        if seeds is None:
            rgb_3 = self._rgb_aug(rgb_3)
        else:
            assert rgb_3.shape[0] == seeds.shape[0]
            outputs = []
            for i in range(seeds.shape[0]):
                torch.manual_seed(seeds[i])
                outputs.append(self._rgb_aug(rgb_3[i]).unsqueeze(0))

            rgb_3 = torch.cat(outputs, dim=0)

        rgb_obs[:, :3, :, :] = rgb_3
        return rgb_obs

    def forward(self, observations):
        if self.is_blind:
            return None
        
        elif getattr(self, "_backbone") == "clip":
            image_goal = observations['rgb'].permute(0, 3, 1, 2) / 255.0
            # input = [self.clip_preprocessor(ig).cuda() for ig in image_goal]
            input = self.clip_preprocessor(image_goal)
            # input = torch.stack(input,dim=0)
            with torch.no_grad():
                target_encoding = self.clip_model_encoder.encode_image(input)
            x = target_encoding.type(torch.float32)
        
        elif getattr(self, "_backbone") == "clip_feat":
            x = observations['rgb'] 
            
        else:    
            seeds = None
            cnn_input = []
            for v_uuid in self._n_input.keys():
                if self._n_input[v_uuid] > 0:
                    v_observations = observations[v_uuid].float()

                    if v_uuid == "panoramic_rgb_sensor":
                        n_panoramic = v_observations.shape[-1]
                        v_observations = rearrange(v_observations, "b h w c n -> b h w (n c)")

                    if v_uuid in ["rgb", "imagegoal_sensor_v2", "panoramic_rgb_sensor", "instance_imagegoal"]:
                        v_observations = v_observations / 255.0  # normalize RGB

                    # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
                    v_observations = v_observations.permute(0, 3, 1, 2)

                    if self.obs_transform:
                        v_observations = self.obs_transform(v_observations, seeds)

                    if v_uuid in ["rgb", "imagegoal_sensor_v2", "panoramic_rgb_sensor", "instance_imagegoal"] and self._rgb_aug is not None:
                        if v_uuid == "panoramic_rgb_sensor":
                            v_observations = rearrange(v_observations, "b (n c) h w -> b c h (n w)", n=n_panoramic)
                            v_observations = self._rgb_augment(v_observations, seeds)
                            v_observations = rearrange(v_observations, "b c h (n w) -> b c h w n", n=n_panoramic)
                        else:
                            v_observations = self._rgb_augment(v_observations, seeds)

                    cnn_input.append(v_observations)

            if self._n_input["panoramic_rgb_sensor"] > 0:
                v_obs_imagegoal = cnn_input[0]
                v_obs_panoramic = cnn_input[1]
                if len(v_obs_panoramic.shape) == 4:
                    v_obs_panoramic = rearrange(v_obs_panoramic, "b (n c) h w -> b c h w n", n=n_panoramic)
                n_panoramic = v_obs_panoramic.shape[-1]
                x = []
                for i in range(n_panoramic):
                    cnn_input = torch.cat([v_obs_imagegoal, v_obs_panoramic[:,:,:,:,i]], dim=1)
                    x.append(self.encoder(cnn_input))
                x = torch.cat(x, dim=1)
                return x

            cnn_input = torch.cat(cnn_input, dim=1)
            
            def visualize_cam(input, layer, image):
                visual = []
                cam = EigenCAM(model=self.encoder, target_layers=[layer], use_cuda=True)
                for i,v in enumerate(cam(input)):
                    v = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    v = cv2.resize(v, image[i].shape[:2])
                    v = cv2.addWeighted(v, 0.5, image[i].cpu().numpy(), 0.5, 0)
                    visual.append(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
                    
                return np.stack(visual, axis=0)
            
            if self.cam_visual:
                l = self.encoder[1].layers[0]
                cam_visual_rgb = visualize_cam(cnn_input, l, observations['rgb'])

                # l = self.encoder[1].layers[0]
                # cam_visual_goal = visualize_cam(cnn_input, l, observations['imagegoal_sensor_v2'])
                # l = self.encoder[1].stem_o.layers[0]
                # cam_visual_o = visualize_cam(cnn_input, l, observations['rgb'])
                # l = self.encoder[1].stem_g.layers[0]
                # cam_visual_g = visualize_cam(cnn_input, l, observations['imagegoal_sensor_v2'])
                # cam_visual_o = []
                # for l in self.encoder[1].stem_o.layers:
                #     cam = EigenCAM(model=self.encoder, target_layers=[l], use_cuda=True)
                #     cam_visual_o.append(cam(cnn_input)[:,:,:,None])
                # cam_visual_o = np.concatenate(cam_visual_o, axis=1)
                
                # cam_visual_g = []
                # for l in self.encoder[1].stem_g.layers:
                #     cam = EigenCAM(model=self.encoder, target_layers=[l], use_cuda=True)
                #     cam_visual_g.append(cam(cnn_input)[:,:,:,None])
                # cam_visual_g = np.concatenate(cam_visual_g, axis=1)
                
                # cam_visual = np.concatenate((cam_visual_goal, cam_visual_rgb), axis=2)
                cam_visual = cam_visual_rgb
                
            else:
                cam_visual = None
                            
            x = self.encoder(cnn_input)
        return x, cam_visual
