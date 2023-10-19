
import copy
import numbers

import numpy as np
import torch
import torch.nn as nn
import torchvision
from gym.spaces import Box

from habitat import logger
from habitat_baselines.utils.common import CustomFixedCategorical


class ResizeRandomCropper(nn.Module):
    def __init__(
        self, size, scale=(0.8, 1.0), channels_last: bool = False
    ):
        r"""An nn module the resizes and randomly crops your input.
        Args:
            size: A sequence (w, h) or int of the size you wish to
                    resize/center_crop. If int, assumes square crop
            channels_list: indicates if channels is the last dimension
        """
        super().__init__()
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        assert len(size) == 2, "forced input size must be len of 2 (w, h)"
        self._size = size
        self._scale = scale
        self.channels_last = channels_last
        self._transform = torchvision.transforms.RandomResizedCrop(
            self._size, scale=self._scale
        )

    def transform_observation_space(
        self, observation_space, trans_keys=["rgb", "depth", "semantic"]
    ):
        size = self._size
        observation_space = copy.deepcopy(observation_space)
        if size:
            for key in observation_space.spaces:
                if (
                    key in trans_keys
                    and observation_space.spaces[key].shape != size
                ):
                    logger.info(
                        "Overwriting CNN input size of %s: %s" % (key, size)
                    )
                    observation_space.spaces[key] = overwrite_gym_box_shape(
                        observation_space.spaces[key], size
                    )
        self.observation_space = observation_space
        return observation_space

    def _batch_transform(self, input, seeds=None):
        if seeds is None:
            return self._transform(input)

        assert input.shape[0] == seeds.shape[0]
        # print(f'seeds: {seeds}')
        outputs = []
        for i in range(seeds.shape[0]):
            torch.manual_seed(seeds[i])
            outputs.append(self._transform(input[i]).unsqueeze(0))

        return torch.cat(outputs, dim=0)

    def forward(self, input: torch.Tensor, seeds: torch.Tensor = None) -> torch.Tensor:
        if self._size is None:
            return input

        assert len(input.shape) == 4
        if self.channels_last:
            input_t = input.permute(0, 3, 1, 2)
            input_t = self._batch_transform(input_t, seeds)  # [ BxCxHxW ]
            input_t = input_t.permute(0, 2, 3, 1).contiguous()
        else:
            input_t = self._batch_transform(input, seeds)

        return input_t


class ResizeCenterCropper(nn.Module):
    def __init__(self, size, channels_last: bool = False):
        r"""An nn module the resizes and center crops your input.
        Args:
            size: A sequence (w, h) or int of the size you wish to
            resize/center_crop. If int, assumes square crop
            channels_list: indicates if channels is the last dimension
        """
        super().__init__()
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        assert len(size) == 2, "forced input size must be len of 2 (w, h)"
        self._size = size
        self.channels_last = channels_last

    def transform_observation_space(
        self, observation_space, trans_keys=["rgb", "depth", "semantic"]
    ):
        size = self._size
        observation_space = copy.deepcopy(observation_space)
        if size:
            for key in observation_space.spaces:
                if (
                    key in trans_keys
                    and observation_space.spaces[key].shape != size
                ):
                    logger.info(
                        "Overwriting CNN input size of %s: %s" % (key, size)
                    )
                    observation_space.spaces[key] = overwrite_gym_box_shape(
                        observation_space.spaces[key], size
                    )
        self.observation_space = observation_space
        return observation_space

    def forward(self, input: torch.Tensor, seeds: torch.Tensor = None) -> torch.Tensor:
        if self._size is None:
            return input

        return center_crop(
            image_resize_shortest_edge(
                input, max(self._size), channels_last=self.channels_last
            ),
            self._size,
            channels_last=self.channels_last,
        )


def _to_tensor(v) -> torch.Tensor:
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


def image_resize_shortest_edge(
    img, size: int, channels_last: bool = False
) -> torch.Tensor:
    """Resizes an img so that the shortest side is length of size while
        preserving aspect ratio.

    Args:
        img: the array object that needs to be resized (HWC) or (NHWC)
        size: the size that you want the shortest edge to be resize to
        channels: a boolean that channel is the last dimension
    Returns:
        The resized array as a torch tensor.
    """
    img = _to_tensor(img)
    no_batch_dim = len(img.shape) == 3
    if len(img.shape) < 3 or len(img.shape) > 5:
        raise NotImplementedError()
    if no_batch_dim:
        img = img.unsqueeze(0)  # Adds a batch dimension
    if channels_last:
        h, w = img.shape[-3:-1]
        if len(img.shape) == 4:
            # NHWC -> NCHW
            img = img.permute(0, 3, 1, 2)
        else:
            # NDHWC -> NDCHW
            img = img.permute(0, 1, 4, 2, 3)
    else:
        # ..HW
        h, w = img.shape[-2:]

    # Percentage resize
    scale = size / min(h, w)
    h = int(h * scale)
    w = int(w * scale)
    img = torch.nn.functional.interpolate(
        img.float(), size=(h, w), mode="area"
    ).to(dtype=img.dtype)
    if channels_last:
        if len(img.shape) == 4:
            # NCHW -> NHWC
            img = img.permute(0, 2, 3, 1)
        else:
            # NDCHW -> NDHWC
            img = img.permute(0, 1, 3, 4, 2)
    if no_batch_dim:
        img = img.squeeze(dim=0)  # Removes the batch dimension
    return img


def center_crop(img, size, channels_last: bool = False):
    """Performs a center crop on an image.

    Args:
        img: the array object that needs to be resized (either batched
            or unbatched)
        size: A sequence (w, h) or a python(int) that you want cropped
        channels_last: If the channels are the last dimension.
    Returns:
        the resized array
    """
    if channels_last:
        # NHWC
        h, w = img.shape[-3:-1]
    else:
        # NCHW
        h, w = img.shape[-2:]

    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    assert len(size) == 2, "size should be (h,w) you wish to resize to"
    cropx, cropy = size

    startx = w // 2 - (cropx // 2)
    starty = h // 2 - (cropy // 2)
    if channels_last:
        return img[..., starty: starty + cropy, startx: startx + cropx, :]
    else:
        return img[..., starty: starty + cropy, startx: startx + cropx]


def overwrite_gym_box_shape(box: Box, shape) -> Box:
    if box.shape == shape:
        return box
    shape = list(shape) + list(box.shape[len(shape):])
    low = box.low if np.isscalar(box.low) else np.min(box.low)
    high = box.high if np.isscalar(box.high) else np.max(box.high)
    return Box(low=low, high=high, shape=shape, dtype=box.dtype)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


class DiagGaussianNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.fc_mean = nn.Linear(num_inputs, num_outputs)
        self.logstd = AddBias(torch.zeros(num_outputs))

        nn.init.orthogonal_(self.fc_mean.weight, gain=0.01)
        nn.init.constant_(self.fc_mean.bias, 0)

    def forward(self, x):
        action_mean = self.fc_mean(x)

        zeros = torch.zeros_like(action_mean)
        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class MaskedCategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, action_masks=None):
        x = self.linear(x)
        # Mask out actions
        if action_masks is not None:
            x.masked_fill_(action_masks, -float("Inf"))
        return CustomFixedCategorical(logits=x)
