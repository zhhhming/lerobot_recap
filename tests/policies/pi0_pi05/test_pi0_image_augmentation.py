#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi0.configuration_pi0 import PI0Config, PI0ImageAugmentationConfig
from lerobot.policies.pi0.modeling_pi0 import PI0Policy


class _ImagePreprocessorOnly:
    _augment_image = PI0Policy._augment_image
    _preprocess_images = PI0Policy._preprocess_images

    def __init__(self, config: PI0Config, training: bool):
        self.config = config
        self.training = training
        self._device_anchor = torch.nn.Parameter(torch.empty(()))

    def parameters(self):
        yield self._device_anchor


def _make_config(augmentation: PI0ImageAugmentationConfig) -> PI0Config:
    config = PI0Config(device="cpu", image_augmentation=augmentation)
    config.input_features = {
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.images.left_wrist_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    return config


def test_pi0_image_augmentation_is_skipped_in_eval_mode():
    config = _make_config(PI0ImageAugmentationConfig(enable=True, brightness=1.0))
    preprocessor = _ImagePreprocessorOnly(config, training=False)
    image = torch.full((2, 3, 224, 224), 0.5)

    images, masks = preprocessor._preprocess_images({"observation.images.base_0_rgb": image})

    assert torch.allclose(images[0], image * 2.0 - 1.0)
    assert masks[0].all()
    assert torch.all(images[1] == -1.0)
    assert not masks[1].any()


def test_pi0_image_augmentation_runs_only_for_training():
    torch.manual_seed(0)
    config = _make_config(
        PI0ImageAugmentationConfig(
            enable=True,
            probability=1.0,
            crop_scale=1.0,
            rotation_degrees=0.0,
            brightness=1.0,
            contrast=0.0,
            saturation=0.0,
        )
    )
    preprocessor = _ImagePreprocessorOnly(config, training=True)
    image = torch.full((2, 3, 224, 224), 0.5)

    images, _ = preprocessor._preprocess_images({"observation.images.base_0_rgb": image})

    assert not torch.allclose(images[0], image * 2.0 - 1.0)
