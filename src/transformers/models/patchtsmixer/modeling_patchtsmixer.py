# coding=utf-8
# Copyright 2023 IBM and HuggingFace Inc. team. All Rights Reserved.
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
""" PyTorch PatchTSMixer model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.modules.activation import MultiheadAttention

from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import set_seed
from transformers.utils import ModelOutput

from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_patchtsmixer import PatchTSMixerConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PatchTSMixerConfig"


PATCHTSMIXER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "ibm/patchtsmixer-etth1-pretrain",
    # See all PatchTSMixer models at https://huggingface.co/models?filter=patchtsmixer
]


PATCHTSMIXER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PatchTSMixerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        mask_input (`bool`, *optional*, defaults to `False`):
            If True, Masking will be enabled. False otherwise.
"""

PATCHTSMIXER_INPUTS_DOCSTRING = r"""
    Parameters:
        past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
            Context values of the time series. For a pretraining task, this denotes the input time series to predict
            the masked portion. For a forecasting task, this denotes the history/past time series values. Similarly,
            for classification or regression tasks, it denotes the appropriate context values of the time series.

            For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series, it is greater
            than 1.

        target_values (`torch.FloatTensor` of shape `(batch_size, target_len, num_input_channels)` for forecasting,
            `(batch_size, num_targets)` for regression, or `(batch_size,)` for classification, *optional*): Target values
            of the time series, that serve as labels for the model. The `target_values` is what the Transformer needs
            during training to learn to output, given the `past_values`. Note that, this is NOT required for a
            pretraining task.

            For a forecasting task, the shape is be `(batch_size, target_len, num_input_channels)`. Even if we want to forecast
            only specific channels by setting the indices in `forecast_channel_indices` parameter, pass the target data
            with all channels, as channel Filtering for both prediction and target will be manually applied before the
            loss computation.

            For a classification task, it has a shape of `(batch_size,)`.

            For a regression task, it has a shape of `(batch_size, num_targets)`.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.

        return_loss (`bool`,  *optional*):
            Whether to return the loss in the `forward` call.
"""


class PatchTSMixerGatedAttention(nn.Module):
    """
    Module that applies gated attention to input data.

    Args:
        in_size (`int`): The input size.
        out_size (`int`): The output size.
    """

    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.attn_layer = nn.Linear(in_size, out_size)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        attn_weight = self.attn_softmax(self.attn_layer(inputs))
        inputs = inputs * attn_weight
        return inputs


class PatchTSMixerTranspose(nn.Module):
    """
    Transpose the input tensor according to specified dimensions.

    Args:
        *dims (`int`): Variable-length list of dimensions to permute the input tensor. The input tensor is
            transposed based on the order of dimensions provided.
        contiguous (`bool`, *optional*, defaults to False): If True, the transposed tensor is made contiguous.

    Returns:
        `torch.Tensor`: The transposed tensor.

    """

    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor`): Input to be transposed.
        Returns:
            `torch.Tensor`: transposed tensor.
        """
        if self.contiguous:
            return inputs.transpose(*self.dims).contiguous()
        else:
            return inputs.transpose(*self.dims)


class PatchTSMixerNormLayer(nn.Module):
    def __init__(
        self,
        norm_mlp="LayerNorm",
        mode="common_channel",
        num_features=16,
    ):
        super().__init__()
        self.norm_mlp = norm_mlp
        self.mode = mode
        self.num_features = num_features
        if "batch" in norm_mlp.lower():
            self.norm = nn.Sequential(
                PatchTSMixerTranspose(1, 2),
                nn.BatchNorm1d(num_features),
                PatchTSMixerTranspose(1, 2),
            )
        else:
            self.norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, num_features))`):
                Input to the normalization layer.
        Returns:
            `torch.Tensor` of shape `((batch_size, num_channels, num_patches, num_features))`
        """
        if "batch" in self.norm_mlp.lower():
            if self.mode in ["common_channel", "mix_channel"]:
                # reshape the data
                inputs_reshaped = torch.reshape(
                    inputs,
                    (
                        inputs.shape[0] * inputs.shape[1],
                        inputs.shape[2],
                        inputs.shape[3],
                    ),
                )  # inputs_reshaped: [batch_size*num_channels, num_patches, num_features]
            else:
                inputs_reshaped = inputs
            inputs_reshaped = self.norm(
                inputs_reshaped
            )  # inputs_reshaped: [batch_size*num_channels, num_patches, num_features]
            # put back data to the original shape
            if self.mode in ["common_channel", "mix_channel"]:
                inputs = torch.reshape(
                    inputs_reshaped,
                    (
                        inputs.shape[0],
                        inputs.shape[1],
                        inputs.shape[2],
                        inputs.shape[3],
                    ),
                )
            else:
                inputs = inputs_reshaped
        else:
            inputs = self.norm(inputs)

        return inputs


class PatchTSMixerMLP(nn.Module):
    def __init__(
        self, in_features, out_features, expansion_factor, dropout, last_dropout=True
    ):
        super().__init__()
        num_hidden = in_features * expansion_factor
        self.fc1 = nn.Linear(in_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, out_features)
        self.last_dropout = last_dropout
        if last_dropout:
            self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, num_features))`):
                Input to the MLP layer.
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        inputs = self.dropout1(nn.functional.gelu(self.fc1(inputs)))
        inputs = self.fc2(inputs)
        if self.last_dropout:
            inputs = self.dropout2(inputs)
        return inputs


class PatchTSMixerChannelFeatureMixerBlock(nn.Module):
    """This module mixes the features in the channel dimension.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        num_features = (config.num_features,)
        num_input_channels = config.num_input_channels
        expansion_factor = config.expansion_factor
        dropout = config.dropout
        mode = config.mode
        gated_attn = config.gated_attn
        norm_mlp = config.norm_mlp

        self.mode = mode

        self.norm = PatchTSMixerNormLayer(
            norm_mlp=norm_mlp, mode=mode, num_features=num_features
        )

        self.mlp = PatchTSMixerMLP(
            num_input_channels, num_input_channels, expansion_factor, dropout
        )

        self.gated_attn = gated_attn
        if gated_attn:
            self.gab = PatchTSMixerGatedAttention(
                in_size=num_input_channels, out_size=num_input_channels
            )

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, num_features))`):
                input to the MLP layer
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        residual = inputs
        inputs = self.norm(inputs)

        inputs = inputs.permute(0, 3, 2, 1)

        if self.gated_attn:
            inputs = self.gab(inputs)

        inputs = self.mlp(inputs)

        inputs = inputs.permute(0, 3, 2, 1)

        out = inputs + residual
        return out


class PatchMixerBlock(nn.Module):
    """This module mixes the patch dimension.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        num_patches = config.num_patches
        num_features = config.num_features
        expansion_factor = config.expansion_factor
        dropout = config.dropout
        mode = config.mode
        gated_attn = config.gated_attn
        self_attn = config.self_attn
        self_attn_heads = config.self_attn_heads
        norm_mlp = config.norm_mlp
        self.norm_mlp = config.norm_mlp
        self.mode = config.mode

        self.norm = PatchTSMixerNormLayer(
            norm_mlp=norm_mlp, mode=mode, num_features=num_features
        )

        self.self_attn = self_attn

        self.mlp = PatchTSMixerMLP(num_patches, num_patches, expansion_factor, dropout)

        self.gated_attn = gated_attn
        if gated_attn:
            self.gab = PatchTSMixerGatedAttention(
                in_size=num_patches, out_size=num_patches
            )

        if self_attn:
            self.self_attn_layer = MultiheadAttention(
                embed_dim=num_features,
                num_heads=self_attn_heads,
                dropout=dropout,
                add_bias_kv=True,
                add_zero_attn=False,
                batch_first=True,
            )
            self.norm_attn = PatchTSMixerNormLayer(
                norm_mlp=norm_mlp, mode=mode, num_features=num_features
            )

    def forward(self, data):
        """
        Args:
            data (`torch.Tensor`): Input tensor.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        residual = data

        data = self.norm(data)

        if self.self_attn:
            data_reshaped = data
            if self.mode in ["common_channel", "mix_channel"]:
                data_reshaped = torch.reshape(
                    data, (data.shape[0] * data.shape[1], data.shape[2], data.shape[3])
                )
                #  (batch_size, num_patches, num_features) if flatten
                #  (batch_size, n_vars, num_patches, num_features) if common_channel

            x_attn, _ = self.self_attn_layer(
                data_reshaped, data_reshaped, data_reshaped, need_weights=False
            )

            if self.mode in ["common_channel", "mix_channel"]:
                x_attn = torch.reshape(
                    x_attn, (data.shape[0], data.shape[1], data.shape[2], data.shape[3])
                )
                #  (batch_size, num_patches, num_features) if flatten
                #  (batch_size, n_vars, num_patches, num_features) if common_channel

        # Transpose so that num_patches is the last dimension
        if self.mode == "flatten":
            data = data.transpose(1, 2)
        elif self.mode in ["common_channel", "mix_channel"]:
            data = data.transpose(2, 3)
        else:
            raise ValueError(
                f"self.mode has to be one of ['flatten', 'common_channel', 'mix_channel'], but is {self.mode}."
            )

        data = self.mlp(data)

        if self.gated_attn:
            data = self.gab(data)

        # Transpose back
        if self.mode == "flatten":
            data = data.transpose(1, 2)
        elif self.mode in ["common_channel", "mix_channel"]:
            data = data.transpose(2, 3)
        else:
            raise ValueError(
                f"self.mode has to be one of ['flatten', 'common_channel', 'mix_channel'], but is {self.mode}."
            )

        if self.self_attn:
            data = self.norm_attn(data + x_attn)

        out = data + residual
        return out


class FeatureMixerBlock(nn.Module):
    """This module mixes the hidden feature dimension.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        num_features = config.num_features
        expansion_factor = config.expansion_factor
        dropout = config.dropout
        gated_attn = config.gated_attn
        mode = config.mode
        norm_mlp = config.norm_mlp

        self.norm = PatchTSMixerNormLayer(
            norm_mlp=norm_mlp, mode=mode, num_features=num_features
        )

        self.mlp = PatchTSMixerMLP(
            num_features, num_features, expansion_factor, dropout
        )

        self.gated_attn = gated_attn

        if self.gated_attn:
            self.gab = PatchTSMixerGatedAttention(
                in_size=num_features, out_size=num_features
            )

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, num_features)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        residual = hidden
        hidden = self.norm(hidden)
        hidden = self.mlp(hidden)

        if self.gated_attn:
            hidden = self.gab(hidden)

        out = hidden + residual
        return out


class PatchTSMixerLayer(nn.Module):
    """
    The `PatchTSMixer` layer that does all three kinds of mixing.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        mode = config.mode

        self.patch_mixer = PatchMixerBlock(
            config=config,
        )
        self.feature_mixer = FeatureMixerBlock(
            config=config,
        )
        # define a cross series mixer
        self.mode = mode
        if mode == "mix_channel":
            self.channel_feature_mixer = PatchTSMixerChannelFeatureMixerBlock(
                config=config,
            )

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, num_features)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        if self.mode == "mix_channel":
            hidden = self.channel_feature_mixer(hidden)

        hidden = self.patch_mixer(hidden)
        hidden = self.feature_mixer(
            hidden
        )  # hidden: (batch_size, num_patches, num_features)
        return hidden


class PatchTSMixerBlock(nn.Module):
    """The main coputing framework of the `PatchTSMixer` model.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        num_layers = config.num_layers

        self.mixers = nn.ModuleList(
            [
                PatchTSMixerLayer(
                    config=config,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, data, output_hidden_states: bool = False):
        """
        Args:
            data (`torch.Tensor`): The input tensor.
            output_hidden_states (`bool`, *optional*, defaults to False.):
                Whether to output the hidden states as well.

        Returns:
            `torch.Tensor`: The embedding. `list`: List of all hidden states if `output_hidden_states` is set to
            `True`.
        """
        all_hidden_states = []

        embedding = data

        for mod in self.mixers:
            embedding = mod(embedding)
            if output_hidden_states is True:
                all_hidden_states.append(embedding)

        if output_hidden_states is True:
            return embedding, all_hidden_states
        else:
            return embedding, None


class PatchTSMixer(nn.Module):
    """
    The entire network. It does the patching operation and
    then applies the necessary `PatchTSMixerBlock`s.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    # @get_class_params
    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        num_patches = config.num_patches
        patch_len = config.patch_len
        num_input_channels = config.num_input_channels
        num_features = config.num_features
        num_layers = config.num_layers
        mode = config.mode

        self.mode = mode
        self.use_positional_encoding = config.use_positional_encoding

        if mode == "flatten":
            self.patcher = nn.Linear(num_input_channels * patch_len, num_features)

        elif mode in ["common_channel", "mix_channel"]:
            self.patcher = nn.Linear(patch_len*3, num_features*3)

        self.num_patches = num_patches
        self.patch_len = patch_len
        self.num_input_channels = num_input_channels
        self.num_features = num_features
        self.num_layers = num_layers

        self.mlp_mixer_encoder = PatchTSMixerBlock(
            config=config,
        )

        if self.use_positional_encoding:
            self.position_enc = positional_encoding(
                config.positional_encoding,
                config.learn_positional_encoding,
                num_patches,
                num_features,
            )

    def forward(self, input_ts, output_hidden_states: bool = False):
        """
        Args:
            input_ts (`torch.Tensor` of shape `(batch_size, num_vars, num_patch, patch_len)`):
                The patch input.
            output_hidden_states (`bool`, *optional*, defaults to False.):
                Whether to output the hidden states as well.

        Returns:
            `torch.Tensor`: The embedding. `list`: List of all hidden states if `output_hidden_states` is set to
            `True`.
        """
        batch_size = input_ts.shape[0]

        if self.mode == "flatten":
            input_ts = input_ts.permute(
                0, 2, 1, 3
            )  # input_ts: [bs  x num_patch x n_vars  x patch_len]
            input_ts = torch.reshape(
                input_ts,
                (
                    batch_size,
                    self.num_patches,
                    self.num_input_channels * self.patch_len,
                ),
            )  # input_ts: [bs x num_patch x patch_len * n_vars]

        input_ts = input_ts.reshape(input_ts.shape[0],input_ts.shape[1],input_ts.shape[2]//3,input_ts.shape[3]*3)

        patches = self.patcher(
            input_ts
        )  # flatten: [bs x num_patch x num_features]   common_channel/mix_channel: [bs x n_vars x num_patch x num_features]

        patches = patches.reshape(patches.shape[0],patches.shape[1],patches.shape[2]*3,patches.shape[3]//3)
        
        
        if self.use_positional_encoding:
            patches = patches + self.position_enc

        embedding, all_hidden_states = self.mlp_mixer_encoder(
            patches, output_hidden_states=output_hidden_states
        )

        return embedding, all_hidden_states


class PatchTSMixerForecastHead(nn.Module):
    """Forecasting Head.

    Args:
        config (`PatchTSMixerConfig`, *required*): Configuration.
    """

    def __init__(
        self,
        config: PatchTSMixerConfig,
        distribution_output=None,
    ):
        super().__init__()

        num_patches = config.num_patches
        num_input_channels = config.num_input_channels
        patch_len = config.patch_len
        num_features = config.num_features
        forecast_len = config.forecast_len
        head_dropout = config.head_dropout
        mode = config.mode
        forecast_channel_indices = config.forecast_channel_indices

        self.forecast_len = forecast_len
        self.nvars = num_input_channels
        self.num_features = num_features
        self.num_input_channels = num_input_channels
        self.patch_len = patch_len
        self.num_patches = num_patches
        self.forecast_channel_indices = forecast_channel_indices

        if self.forecast_channel_indices is not None:
            self.forecast_channel_indices.sort()
        self.mode = mode
        self.distribution_output = distribution_output
        if self.mode in ["common_channel", "mix_channel"]:
            if distribution_output is None:
                self.base_forecast_block = nn.Sequential(
                    nn.Dropout(head_dropout),
                    nn.Linear((num_patches * num_features), forecast_len),
                )
            else:
                self.base_forecast_block = nn.Sequential(
                    nn.Dropout(head_dropout),
                    distribution_output.get_parameter_projection(
                        num_patches * num_features
                    ),
                )

            self.flatten = nn.Flatten(start_dim=-2)

        else:
            if distribution_output is None:
                self.base_forecast_block = nn.Sequential(
                    nn.Dropout(head_dropout),
                    nn.Linear(
                        (num_patches * num_features), forecast_len * num_input_channels
                    ),
                )
            else:
                self.base_forecast_block = nn.Sequential(
                    nn.Dropout(head_dropout),
                    distribution_output.get_parameter_projection(
                        num_patches * num_features
                    ),
                )

            self.flatten = nn.Flatten(start_dim=1)

    def forward(self, hidden_features):
        """

        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size, num_patch, num_features)` in `flatten` mode
                or `(batch_size, n_vars, num_patch, num_features)` in `common_channel`/`mix_channel` mode.):
                Input hidden features.

        Returns:
            `torch.Tensor` of shape `(batch_size, forecast_len, nvars)`.

        """
        if self.mode in ["common_channel", "mix_channel"]:
            hidden_features = self.flatten(
                hidden_features
            )  # [batch_size x n_vars x num_patch * num_features]

            forecast = self.base_forecast_block(
                hidden_features
            )  # [batch_size x n_vars x forecast_len]
            if isinstance(forecast, tuple):
                forecast = tuple(z.transpose(-1, -2) for z in forecast)
            else:
                forecast = forecast.transpose(
                    -1, -2
                )  # [batch_size x forecast_len x n_vars]

        else:
            hidden_features = self.flatten(
                hidden_features
            )  # hidden_features: [batch_size x num_patches*num_features]
            forecast = self.base_forecast_block(
                hidden_features
            )  # [batch_size x forecast_len * nvars]

            if isinstance(forecast, tuple):
                forecast = tuple(
                    z.reshape(-1, self.forecast_len, self.nvars) for z in forecast
                )
            else:
                forecast = forecast.reshape(
                    -1, self.forecast_len, self.nvars
                )  # [batch_size x forecast_len x n_vars]

        if self.forecast_channel_indices is not None:
            if isinstance(forecast, tuple):
                forecast = tuple(
                    z[..., self.forecast_channel_indices] for z in forecast
                )
            else:
                forecast = forecast[
                    ..., self.forecast_channel_indices
                ]  # [batch_size x forecast_len x n_vars]

        return forecast


class PatchTSMixerLinearHead(nn.Module):
    """Linear head for Classification and Regression.

    Args:
        config (`PatchTSMixerConfig`, *required*):

    """

    def __init__(
        self,
        config: PatchTSMixerConfig,
        distribution_output=None,
    ):
        super().__init__()

        num_patches = config.num_patches
        num_input_channels = config.num_input_channels
        num_features = config.num_features
        head_dropout = config.head_dropout
        output_dim = config.num_targets
        output_range = config.output_range
        head_agg = config.head_agg
        mode = config.mode

        self.nvars = num_input_channels
        self.num_features = num_features
        self.num_input_channels = num_input_channels
        self.head_dropout = head_dropout
        self.output_dim = output_dim
        self.mode = mode
        self.head_agg = head_agg
        self.output_range = output_range
        self.num_patches = num_patches
        self.distribution_output = distribution_output
        if self.head_agg is None:
            mul_factor = self.num_patches
        else:
            mul_factor = 1

        if mode != "flatten":
            if distribution_output is None:
                self.projection = nn.Linear(
                    num_features * num_input_channels * mul_factor, output_dim
                )
            else:
                self.projection = distribution_output.get_parameter_projection(
                    num_features * num_input_channels * mul_factor
                )

            if self.head_agg is None:
                self.flatten = nn.Flatten(start_dim=-3)
            else:
                self.flatten = nn.Flatten(start_dim=-2)
        else:
            if distribution_output is None:
                self.projection = nn.Linear(num_features * mul_factor, output_dim)
            else:
                self.projection = distribution_output.get_parameter_projection(
                    num_features * mul_factor
                )

            if self.head_agg is None:
                self.flatten = nn.Flatten(start_dim=-2)
            else:
                self.flatten = None

        self.dropout = nn.Dropout(head_dropout)

    def forward(self, hidden_features):
        """
        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size x num_patch x num_features)` in `flatten` mode
                or `(batch_size x n_vars x num_patch x num_features)` in `common_channel`/`mix_channel` mode.): Input
                hidden features.

        Returns:
            `torch.Tensor` of shape `(batch_size x output_dim)`.
        """
        hidden_features = hidden_features.transpose(
            -1, -2
        )  # batch_size x num_features x num_patch or batch_size x n_vars x num_features x num_patch
        if self.head_agg == "use_last":
            hidden_features = hidden_features[
                ..., -1
            ]  # # batch_size x num_features (flatten) or # batch_size x n_vars x num_features (common_channel)
        elif self.head_agg == "max_pool":
            hidden_features = hidden_features.max(
                dim=-1
            ).values  # batch_size x n_vars x num_features or batch_size x num_features
        elif self.head_agg == "avg_pool":
            hidden_features = hidden_features.mean(
                dim=-1
            )  # batch_size x n_vars x num_features or batch_size x num_features

        if self.flatten:
            hidden_features = self.flatten(hidden_features)
        hidden_features = self.dropout(hidden_features)
        hidden_features = self.projection(hidden_features)  # batch_size x output_dim

        if (self.distribution_output is None) and (self.output_range is not None):
            hidden_features = (
                torch.sigmoid(hidden_features)
                * (self.output_range[1] - self.output_range[0])
                + self.output_range[0]
            )
        return hidden_features


class PatchTSMixerPreTrainedModel(PreTrainedModel):
    # Weight initialization
    config_class = PatchTSMixerConfig
    base_model_prefix = "model"
    main_input_name = "past_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (PatchTSMixerEncoder)):
            module.gradient_checkpointing = value


class PatchTSMixerPretrainHead(nn.Module):
    """Pretraining head.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        num_patches = config.num_patches
        num_features = config.num_features
        num_input_channels = config.num_input_channels
        patch_len = config.patch_len
        head_dropout = config.head_dropout
        mode = config.mode
        self.mode = mode
        self.patch_len = patch_len
        self.num_input_channels = num_input_channels
        self.num_patches = num_patches

        if self.mode in ["common_channel", "mix_channel"]:
            self.base_pt_block = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(num_features, patch_len),
            )
        else:
            self.base_pt_block = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(num_features, patch_len * num_input_channels),
            )

    def forward(self, hidden_features):
        """
        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size x num_patch x num_features)` in `flatten` mode
                or `(batch_size x n_vars x num_patch x num_features)` in `common_channel`/`mix_channel` mode.): Input
                hidden features.

        Returns:
            `torch.Tensor` of shape `(batch_size x n_vars x num_patch x patch_len)`.
        """

        if self.mode == "flatten":
            hidden_features = self.base_pt_block(
                hidden_features
            )  # hidden_features: [batch_size x num_patch x n_vars*patch_len]
            hidden_features = torch.reshape(
                hidden_features,
                (
                    hidden_features.shape[0],
                    hidden_features.shape[1],
                    self.patch_len,
                    self.num_input_channels,
                ),
            )  # [batch_size x num_patch x patch_len x n_vars]
            hidden_features = hidden_features.permute(
                0, 3, 1, 2
            )  # [batch_size x nvars x num_patch  x patch_len]
            return hidden_features
        elif self.mode in ["common_channel", "mix_channel"]:
            forecast = self.base_pt_block(
                hidden_features
            )  # [batch_size x n_vars x num_patch x patch_len]
            return forecast
        else:
            raise ValueError(
                f"self.mode has to be one of ['flatten', 'common_channel', 'mix_channel'], but is {self.mode}."
            )


# TODO: add copied from after PatchTST master merge
def positional_encoding(position_embedding_type, learned, q_len, d_model):
    # Positional encoding
    if position_embedding_type is None:
        # position_embedding_type = None and learned = False can be used to measure impact of positional encoding
        position_enc = torch.empty((q_len, d_model))
        nn.init.uniform_(position_enc, -0.02, 0.02)
        learned = False
    elif position_embedding_type == "zeros":
        position_enc = torch.empty((q_len, d_model))
        nn.init.uniform_(position_enc, -0.02, 0.02)
    elif position_embedding_type == "normal":
        position_enc = torch.zeros((q_len, 1))
        nn.init.normal_(position_enc, mean=0.0, std=0.1)
    elif position_embedding_type == "uniform":
        position_enc = torch.zeros((q_len, 1))
        nn.init.uniform_(position_enc, a=0.0, b=0.1)
    elif position_embedding_type == "sincos":
        position_enc = torch.zeros(q_len, d_model)
        position = torch.arange(0, q_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        position_enc[:, 0::2] = torch.sin(position * div_term)
        position_enc[:, 1::2] = torch.cos(position * div_term)
        position_enc = position_enc - position_enc.mean()
        position_enc = position_enc / (position_enc.std() * 10)
    else:
        raise ValueError(
            f"{position_embedding_type} is not a valid positional encoder. Available types are 'normal', 'zeros', 'zero', uniform', 'sincos', None."
        )
    return nn.Parameter(position_enc, requires_grad=learned)


# TODO: add copied from after PatchTST master merge
def compute_num_patches(sequence_length, patch_length, stride):
    return (max(sequence_length, patch_length) - patch_length) // stride + 1


# TODO: add copied from after PatchTST master merge
def random_masking(
    inputs: torch.Tensor,
    mask_ratio: float,
    unmasked_channel_indices: list = None,
    channel_consistent_masking: bool = False,
    mask_value: int = 0,
    seed_number: Optional[int] = None,
):
    """random_masking: Mask the input considering the control variables.

    Args:
        inputs (`torch.Tensor` of shape `(batch_size, num_channels, sequence_length, num_features)`):
            The input tensor to mask.
        mask_ratio (`float`):
            Mask ratio.
        unmasked_channel_indices (list, *optional*):
            indices of unmasked channels. These channels will not be masked. Defaults to None.
        channel_consistent_masking (bool, *optional* defaults to False):
            When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary
            across channels. Defaults to False.
        mask_value (int, *optional* defaults to 0):
            Value to use for masking.
        seed_number (int, *optional*):
            Value to set for the random seed.

    Returns:
        `tuple(torch.Tensor)`: inputs_mask, masked input, same shape as input Tensor and mask tensor of shape [bs x c x
        n]
    """
    if seed_number:
        set_seed(seed_number)

    batch_size, num_channels, sequence_length, num_features = inputs.shape
    device = inputs.device

    len_keep = int(sequence_length * (1 - mask_ratio))

    if channel_consistent_masking:
        noise = torch.rand(
            batch_size, 1, sequence_length, device=device
        )  # noise in [0, 1], bs x 1 x  L
        noise = noise.repeat(1, num_channels, 1)  # bs x num_channels x time
    else:
        noise = torch.rand(
            batch_size, num_channels, sequence_length, device=device
        )  # noise in [0, 1], bs x num_channels x L

    mask = torch.ones(
        batch_size, num_channels, sequence_length, device=device
    )  # mask: [bs x num_channels x num_patch]
    mask[:, :, :len_keep] = 0

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(
        ids_shuffle, dim=-1
    )  # ids_restore: [bs x num_channels x L]

    mask = torch.gather(mask, dim=-1, index=ids_restore)
    mask = mask.unsqueeze(-1).repeat(
        1, 1, 1, num_features
    )  # mask: [bs x num_channels x num_patches x patch_length]
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0

    inputs_mask = inputs.masked_fill(mask.bool(), mask_value)
    return inputs_mask, mask[..., 0]


# TODO: add copied from after PatchTST master merge
def forecast_masking(
    inputs: torch.Tensor,
    patch_lengths: list,
    mix_ratio: list = None,
    unmasked_channel_indices: list = None,
    mask_value: int = 0,
    seed_number: Optional[int] = None,
):
    """forecast_masking Mask last K patches where K is from the patch_lengths list.
    For every batch, distribute the patch lengths based on mix_ratio Ignore masks for column indices mentioned in
    cv_channel_indices

    Args:
        inputs (`torch.Tensor` of shape `(batch_size, num_channels, num_patch, patch_len)` or
                    `(batch_size, tsg1, tag2, num_channels, num_patch, patch_len)`):
            Input to mask
        patch_lengths (`list`): List of patch lengths to mask in the end of the data.
        mix_ratio (`list`, *optional*): List of weights to use for each patch length. For Ex.
            if patch_lengths is [5,4] and mix_ratio is [1,1], then equal weights to both patch lengths. Defaults to
            None.
        unmasked_channel_indices (`list`, *optional*):
            Control Variable channel indices. These channels will not be masked. Defaults to None.
        mask_value (`int`, *optional*): Value to use for masking. Defaults to 0.

    Returns:
        `tuple(torch.Tensor)`: inputs_mask of the same shape as inputs Tensor and Mask tensor of shape `(batch_size,
        num_channels , num_patch)` or `(batch_size, tsg1, tsg2, num_channels, num_patch)`
    """
    if seed_number:
        set_seed(seed_number)

    if mix_ratio is None:
        mix_ratio = [1 for t in patch_lengths]

    batch_size, num_channels, sequence_length, num_features = inputs.shape
    mask = torch.zeros(batch_size, num_channels, sequence_length, device=inputs.device)

    t_list = []
    total_length = 0
    total_ratio = sum(mix_ratio)

    for i, j in zip(patch_lengths, mix_ratio):
        if i <= 0 or i >= sequence_length:
            raise Exception(
                "masked_patch_len should be greater than 0 and less than total patches."
            )
        temp_len = int(batch_size * j / total_ratio)
        t_list.append([i, j, temp_len])
        total_length += temp_len

    t_list = sorted(t_list, key=lambda x: x[2])

    if total_length < batch_size:
        t_list[0][2] = t_list[0][2] + (batch_size - total_length)
    elif total_length > batch_size:
        t_list[-1][2] = t_list[-1][2] + (total_length - batch_size)

    b1 = 0
    for p, r, l in t_list:
        b2 = b1 + l
        mask[b1:b2, :, -p:] = 1
        b1 = b2

    perm = torch.randperm(mask.shape[0])
    mask = mask[perm]

    mask = mask.unsqueeze(-1).repeat(
        1, 1, 1, num_features
    )  # mask: [batch_size x num_channels x num_patch x patch_len]
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0

    inputs_mask = inputs.masked_fill(mask.bool(), mask_value)
    return inputs_mask, mask[..., 0]


# TODO: add copied from after PatchTST master merge
class PatchTSMixerPatchify(nn.Module):
    """
    Parameters:
    A class to patchify the time series sequence into different patches
        sequence_length (`int`, required): input sequence length. patch_length (`int`, required): patch length. stride
        (`int`, required): stride between patches.
    Returns:
        z: output tensor data [batch_size x num_input_channels x num_patches x patch_length]
    """

    def __init__(
        self,
        sequence_length: int,
        patch_length: int,
        stride: int,
        padding: bool = False,  # TODO: use this to set whether we want to pad zeros to the sequence
    ):
        super().__init__()

        assert (
            sequence_length > patch_length
        ), f"Sequence length ({sequence_length}) has to be greater than the patch length ({patch_length})"

        self.sequence_length = sequence_length
        self.patch_length = patch_length
        self.stride = stride

        # get the number of patches
        self.num_patches = compute_num_patches(sequence_length, patch_length, stride)
        new_sequence_length = patch_length + stride * (self.num_patches - 1)
        self.s_begin = sequence_length - new_sequence_length

    def forward(self, past_values: torch.Tensor):
        """
        Parameters:
            past_values (torch.Tensor, required): Input of shape [batch_size x sequence_length x num_input_channels]
        Returns:
            x: output tensor data [batch_size x num_input_channels x num_patches x patch_length]
        """
        sequence_length = past_values.shape[-2]
        assert (
            sequence_length == self.sequence_length
        ), f"Input sequence length ({sequence_length}) doesn't match model configuration ({self.sequence_length})."

        x = past_values[
            :, self.s_begin :, :
        ]  # x: [batch_size x new_sequence_length x num_channels]
        x = x.unfold(
            dimension=-2, size=self.patch_length, step=self.stride
        )  # x: [batch_size x num_patches x num_input_channels x patch_length]
        x = x.transpose(
            -2, -3
        ).contiguous()  # x: [batch_size x num_input_channels x num_patches x patch_length]
        return x


# TODO: add copied from after PatchTST master merge
class PatchTSMixerMasking(nn.Module):
    """
    Class to random or forcast masking.

    Parameters:
        mask_type (str, *optional*): Masking type. Allowed values are random, forecast. Defaults to random.
        mask_ratio (`float`, *optional*): Mask ratio.
        mask_patches (`list`, *optional*): List of patch lengths to mask in the end of the data.
        mask_patch_ratios (`list`, *optional*): List of weights to use for each patch length. For Ex. if
            patch_lengths is [5,4] and mix_ratio is [1,1], then equal weights to both patch lengths. Defaults to None.
        unmasked_channel_indices (`list`, *optional*):
            Control Variable channel indices. These channels will not be masked. Defaults to None.
        channel_consistent_masking (bool, *optional*):
            When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary
            across channels. Defaults to True.
        mask_value (`int`, *optional*): Value to use for masking. Defaults to 0.
        seed_number (`int`, *optional*): Random seed, when None seed is not set. Defaults to None.
    """

    def __init__(
        self,
        mask_type: str = "random",
        mask_ratio=0.5,
        mask_patches: list = [2, 3],
        mask_patch_ratios: list = [1, 1],
        channel_consistent_masking: bool = False,
        unmasked_channel_indices: list = None,
        mask_value=0,
        seed_number: Optional[int] = None,
    ):
        self.mask_ratio = mask_ratio
        self.channel_consistent_masking = channel_consistent_masking
        self.mask_type = mask_type
        self.mask_patches = mask_patches
        self.mask_patch_ratios = mask_patch_ratios
        self.unmasked_channel_indices = unmasked_channel_indices
        self.mask_value = mask_value
        if self.unmasked_channel_indices is not None:
            self.unmasked_channel_indices.sort()
        self.seed_number = seed_number

        super().__init__()

    def forward(self, patch_input: torch.Tensor):
        """
        Parameters:
            patch_input (`torch.Tensor` of shape `(batch_size, num_input_channels, num_patches, patch_length)`):
                patched input
        Returns:
            x_mask (`torch.Tensor` of shape `(batch_size, num_input_channels, num_patches, patch_length)`) :
                Masked patched input
            mask (`torch.Tensor` of shape `(batch_size, num_input_channels, num_patches)`):
                Bool tensor indicating True on masked points

        """

        if self.mask_type == "random":
            x_mask, mask = random_masking(
                inputs=patch_input,
                mask_ratio=self.mask_ratio,
                unmasked_channel_indices=self.unmasked_channel_indices,
                channel_consistent_masking=self.channel_consistent_masking,
                mask_value=self.mask_value,
                seed_number=self.seed_number,
            )
        elif self.mask_type == "forecast":
            x_mask, mask = forecast_masking(
                inputs=patch_input,
                patch_lengths=self.mask_patches,
                mix_ratio=self.mask_patch_ratios,
                unmasked_channel_indices=self.unmasked_channel_indices,
                mask_value=self.mask_value,
                seed_number=self.seed_number,
            )
        else:
            raise ValueError("Invalid mask type")

        mask = mask.bool()  # mask: [batch_size x num_input_channels x num_patch]

        return x_mask, mask


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesStdScaler with TimeSeries->PatchTSMixer
class PatchTSMixerStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along some given dimension `dim`, and then normalizes it
    by subtracting from the mean and dividing by the standard deviation.

    Args:
        dim (`int`):
            Dimension along which to calculate the mean and standard deviation.
        keepdim (`bool`, *optional*, defaults to `False`):
            Controls whether to retain dimension `dim` (of length 1) in the scale tensor, or suppress it.
        minimum_scale (`float`, *optional*, defaults to 1e-5):
            Default scale that is used for elements that are constantly zero along dimension `dim`.
    """

    def __init__(self, dim: int, keepdim: bool = False, minimum_scale: float = 1e-5):
        super().__init__()
        if not dim > 0:
            raise ValueError(
                "Cannot compute scale along dim = 0 (batch dimension), please provide dim > 0"
            )
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale

    @torch.no_grad()
    def forward(
        self, data: torch.Tensor, weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        denominator = weights.sum(self.dim, keepdim=self.keepdim)
        denominator = denominator.clamp_min(1.0)
        loc = (data * weights).sum(self.dim, keepdim=self.keepdim) / denominator

        variance = (((data - loc) * weights) ** 2).sum(
            self.dim, keepdim=self.keepdim
        ) / denominator
        scale = torch.sqrt(variance + self.minimum_scale)
        return (data - loc) / scale, loc, scale


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesMeanScaler with TimeSeries->PatchTSMixer
class PatchTSMixerMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along dimension `dim`, and scales the data
    accordingly.

    Args:
        dim (`int`):
            Dimension along which to compute the scale.
        keepdim (`bool`, *optional*, defaults to `False`):
            Controls whether to retain dimension `dim` (of length 1) in the scale tensor, or suppress it.
        default_scale (`float`, *optional*, defaults to `None`):
            Default scale that is used for elements that are constantly zero. If `None`, we use the scale of the batch.
        minimum_scale (`float`, *optional*, defaults to 1e-10):
            Default minimum possible scale that is used for any item.
    """

    def __init__(
        self,
        dim: int = -1,
        keepdim: bool = True,
        default_scale: Optional[float] = None,
        minimum_scale: float = 1e-10,
    ):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale
        self.default_scale = default_scale

    @torch.no_grad()
    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # shape: (N, [C], T=1)
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        scale = ts_sum / torch.clamp(num_observed, min=1)

        # If `default_scale` is provided, we use it, otherwise we use the scale
        # of the batch.
        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)

        # apply default scale where there are no observations
        scale = torch.where(num_observed > 0, scale, default_scale)

        # ensure the scale is at least `self.minimum_scale`
        scale = torch.clamp(scale, min=self.minimum_scale)
        scaled_data = data / scale

        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        return scaled_data, torch.zeros_like(scale), scale


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesNOPScaler with TimeSeries->PatchTSMixer
class PatchTSMixerNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along dimension `dim`, and therefore applies no scaling to the input data.

    Args:
        dim (`int`):
            Dimension along which to compute the scale.
        keepdim (`bool`, *optional*, defaults to `False`):
            Controls whether to retain dimension `dim` (of length 1) in the scale tensor, or suppress it.
    """

    def __init__(self, dim: int, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = torch.ones_like(data, requires_grad=False).mean(
            dim=self.dim, keepdim=self.keepdim
        )
        loc = torch.zeros_like(data, requires_grad=False).mean(
            dim=self.dim, keepdim=self.keepdim
        )
        return data, loc, scale


class InjectScalerStatistics4D(nn.Module):
    def __init__(self, num_features: int, num_patches: int, expansion: int = 2):
        super().__init__()
        self.inverse_transform = nn.Sequential(
            nn.Linear(num_features + 2, expansion * num_features),
            nn.Linear(expansion * num_features, num_features),
        )

        self.map_scale = nn.Sequential(
            nn.Linear(2, 2 * expansion), nn.Linear(2 * expansion, 2)
        )
        self.num_patches = num_patches

    def forward(self, inputs: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `(batch_size, num_input_channels, num_patch, num_features)`)
            loc (`torch.Tensor` of shape `(batch_size, 1, num_input_channels)`)
            scale (`torch.Tensor` of shape `(batch_size, 1, num_input_channels)`)
        Returns:
            `torch.Tensor` of shape `(batch_size, num_input_channels, num_patch, num_features)`
        """

        mean = loc.transpose(-1, -2)  # [batch_size x n_channels x 1 ]
        mean = mean.unsqueeze(-2)  # [batch_size x n_channels x 1 x 1]
        mean = mean.repeat(
            1, 1, self.num_patches, 1
        )  # [batch_size x n_channels x num_patch x 1]

        stdev = scale.transpose(-1, -2)  # [batch_size x n_channels x 1 ]
        stdev = stdev.unsqueeze(-2)  # [batch_size x n_channels x 1 x 1]
        stdev = stdev.repeat(
            1, 1, self.num_patches, 1
        )  # [batch_size x n_channels x num_patch x 1]

        concat_stats = torch.cat(
            [mean, stdev], dim=-1
        )  # [batch_size x n_channels x num_patch x 2]

        concat_stats = self.map_scale(
            concat_stats
        )  # [batch_size x n_channels x num_patch x 2]

        inputs = torch.cat(
            [inputs, concat_stats], dim=-1
        )  # [batch_size x channels x num_patch x num_features+2]
        inputs = self.inverse_transform(
            inputs
        )  # [batch_size x channels x num_patch x num_features]

        return inputs


@dataclass
class PatchTSMixerEncoderOutput(ModelOutput):
    """
    Base class for `PatchTSMixerEncoderOutput`, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, num_features)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class PatchTSMixerEncoder(PatchTSMixerPreTrainedModel):
    """
    Encoder for PatchTSMixer which inputs patched time-series and outputs patched embeddings.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        self.encoder = PatchTSMixer(config=config)

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @replace_return_docstrings(
        output_type=PatchTSMixerEncoderOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        past_values: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PatchTSMixerEncoderOutput]:
        r"""
        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, patch_len)`):
                Patched input context.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.

        Returns:
        """

        # past_values: [batch_size  x n_vars x num_patches x patch_len]
        # return: [batch_size x n_vars x num_patches x num_features]
        last_hidden_state, hidden_states = self.encoder(
            past_values, output_hidden_states=output_hidden_states
        )
        return PatchTSMixerEncoderOutput(
            last_hidden_state=last_hidden_state, hidden_states=hidden_states
        )


@dataclass
class PatchTSMixerModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor`  of shape `(batch_size, num_channels, num_patches, num_features)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
        patched_input (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_len)`):
            Patched input data to the model.
        mask: (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches)`,*optional*):
            Bool Tensor indicating True in masked patches and False otherwise.
        loc: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*):
            Gives the mean of the context window per channel. Used for revin denorm outside the model, if revin
            enabled.
        scale: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*):
            Gives the std dev of the context window per channel. Used for revin denorm outside the model, if revin
            enabled.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    patched_input: torch.FloatTensor = None
    mask: Optional[torch.FloatTensor] = None
    loc: Optional[torch.FloatTensor] = None
    scale: Optional[torch.FloatTensor] = None


@add_start_docstrings(
    "The PatchTSMixer Model for time-series forecasting.",
    PATCHTSMIXER_START_DOCSTRING,
)
class PatchTSMixerModel(PatchTSMixerPreTrainedModel):
    def __init__(self, config: PatchTSMixerConfig, mask_input: bool = False):
        super().__init__(config)

        self.encoder = PatchTSMixerEncoder(config)
        self.patching = PatchTSMixerPatchify(
            config.seq_len, patch_length=config.patch_len, stride=config.stride
        )

        if mask_input is True:
            self.masking = PatchTSMixerMasking(
                mask_type=config.mask_type,
                mask_ratio=config.mask_ratio,
                mask_patches=config.mask_patches,
                mask_patch_ratios=config.mask_patch_ratios,
                channel_consistent_masking=config.channel_consistent_masking,
                mask_value=config.mask_value,
                seed_number=config.seed_number,
            )
        else:
            self.masking = None

        if config.scaling == "mean" or config.scaling is True:
            self.scaler = PatchTSMixerMeanScaler(dim=1, keepdim=True)
        elif config.scaling == "std":
            self.scaler = PatchTSMixerStdScaler(dim=1, keepdim=True)
        else:
            self.scaler = PatchTSMixerNOPScaler(dim=1, keepdim=True)

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    # @replace_return_docstrings(output_type=PatchTSMixerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> PatchTSMixerModelOutput:
        r"""
        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
                Context values of the time series. For a pretraining task, this denotes the input time series to
                predict the masked portion. For a forecasting task, this denotes the history/past time series values.
                Similarly, for classification or regression tasks, it denotes the appropriate context values of the
                time series.

                For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series, it is
                greater than 1.
            observed_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:
                    - 1 for values that are **observed**,
                    - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Return dict.

        Returns:
            [`PatchTSMixerModelOutput`]
        """

        mask = None
        if observed_mask is None:
            observed_mask = torch.ones_like(past_values)
        past_values = torch.reshape(past_values,(past_values.shape[0],past_values.shape[1]//3,past_values.shape[2]*3))
        observed_mask = torch.reshape(observed_mask,(observed_mask.shape[0],observed_mask.shape[1]//3,observed_mask.shape[2]*3))
        scaled_past_values, loc, scale = self.scaler(past_values, observed_mask)
        scaled_past_values = torch.reshape(scaled_past_values,(scaled_past_values.shape[0],scaled_past_values.shape[1]*3,scaled_past_values.shape[2]//3))
        
        observed_mask = torch.reshape(observed_mask,(observed_mask.shape[0],observed_mask.shape[1]*3,observed_mask.shape[2]//3))
        
        patched_x = self.patching(
            scaled_past_values
        )  # [batch_size x num_input_channels x num_patch x patch_len

        enc_input = patched_x
        if self.masking is not None:
            enc_input, mask = self.masking(patched_x)
            # enc_input: [batch_size x num_input_channels x num_patch x patch_len]
            # mask: [batch_size x num_input_channels x num_patch]

        encoder_output = self.encoder(
            enc_input, output_hidden_states=output_hidden_states
        )

        return PatchTSMixerModelOutput(
            last_hidden_state=encoder_output.last_hidden_state,
            hidden_states=encoder_output.hidden_states,
            patched_input=patched_x,
            mask=mask,
            loc=loc,
            scale=scale,
        )


@dataclass
class PatchTSMixerForMaskPreTrainingOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForMaskPreTrainingOutput`].

    Args:
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, patch_len)`):
            Prediction output from the pretrain head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, num_features)`):
            Backbone embeddings before passing through the head.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss
    """

    prediction_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None


class PatchTSMixerForPretraining(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for mask pretraining.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)
        self.model = PatchTSMixerModel(config, mask_input=True)
        self.head = PatchTSMixerPretrainHead(
            config=config,
        )
        self.masked_loss = config.masked_loss
        if config.masked_loss is True:
            self.loss = torch.nn.MSELoss(reduction="none")
        else:
            self.loss = torch.nn.MSELoss(reduction="mean")

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @replace_return_docstrings(
        output_type=PatchTSMixerForMaskPreTrainingOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
    ) -> PatchTSMixerForMaskPreTrainingOutput:
        r"""
        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
                Context values of the time series. For a pretraining task, this denotes the input time series to
                predict the masked portion. For a forecasting task, this denotes the history/past time series values.
                Similarly, for classification or regression tasks, it denotes the appropriate context values of the
                time series.

                For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series, it is
                greater than 1.

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_loss (`bool`,  *optional*):
                Whether to return the loss in the `forward` call.

        Returns:

        """

        # past_values: tensor [batch_size x seq_len x num_input_channels]
        model_output = self.model(
            past_values,
            observed_mask=observed_mask,
            output_hidden_states=output_hidden_states,
        )  # x.last_hidden_state: [batch_size x nvars x num_patch x num_features]
        x_hat = self.head(
            model_output.last_hidden_state
        )  # tensor [batch_size x nvars x num_patch x patch_len]

        if return_loss is True:
            loss_val = self.loss(x_hat, model_output.patched_input)
        else:
            loss_val = None

        # calculate masked_loss
        if self.masked_loss is True and loss_val is not None:
            loss_val = (loss_val.mean(dim=-1) * model_output.mask).sum() / (
                model_output.mask.sum() + 1e-10
            )

        return PatchTSMixerForMaskPreTrainingOutput(
            prediction_logits=x_hat,  # tensor [batch_size x nvars x num_patch x patch_len]
            last_hidden_state=model_output.last_hidden_state,  # x: [batch_size x nvars x num_patch x num_features]
            hidden_states=model_output.hidden_states,
            loss=loss_val,
        )


@dataclass
class PatchTSMixerForForecastOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForForecastOutput`].

    Args:
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, forecast_len, num_input_channels)`):
            Prediction output from the forecast head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, num_features)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
        loc (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`):
            Input mean
        scale (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`):
            Input std dev

    """

    prediction_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None
    loc: torch.FloatTensor = None
    scale: torch.FloatTensor = None


@dataclass
class SamplePatchTSMixerForecastOutput(ModelOutput):
    """
    Base class for time series model's predictions outputs that contains the sampled values from the chosen
    distribution.

    Parameters:
        sequences (`torch.FloatTensor` of shape `(batch_size, num_samples, prediction_length, number_channels)`):
            Sampled values from the chosen distribution.
    """

    sequences: torch.FloatTensor = None


@dataclass
class SamplePatchTSMixerRegressionOutput(ModelOutput):
    """
    Base class for time series model's predictions outputs that contains the sampled values from the chosen
    distribution.

    Parameters:
        sequences (`torch.FloatTensor` of shape `(batch_size, num_samples, num_targets)`
                Sampled values from the chosen distribution.
    """

    sequences: torch.FloatTensor = None


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.nll
def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    return -input.log_prob(target)


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.weighted_average
def weighted_average(
    input_tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None
) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given `dim`, masking values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Args:
        input_tensor (`torch.FloatTensor`):
            Input tensor, of which the average must be computed.
        weights (`torch.FloatTensor`, *optional*):
            Weights tensor, of the same shape as `input_tensor`.
        dim (`int`, *optional*):
            The dim along which to average `input_tensor`.

    Returns:
        `torch.FloatTensor`: The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weighted_tensor = torch.where(
            weights != 0, input_tensor * weights, torch.zeros_like(input_tensor)
        )
        sum_weights = torch.clamp(
            weights.sum(dim=dim) if dim else weights.sum(), min=1.0
        )
        return (
            weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()
        ) / sum_weights
    else:
        return input_tensor.mean(dim=dim)


class PatchTSMixerForForecasting(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for forecasting application.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)
        self.config = config
        if config.loss == "mse":
            self.loss = nn.MSELoss(reduction="mean")
            self.distribution_output = None
        else:
            self.loss = nll
            if config.mode in ["common_channel", "mix_channel"]:
                dim = config.forecast_len
            else:
                dim = config.forecast_len * config.num_input_channels

            if config.distribution_output == "student_t":
                self.distribution_output = StudentTOutput(dim=dim)
            elif config.distribution_output == "normal":
                self.distribution_output = NormalOutput(dim=dim)
            elif config.distribution_output == "negative_binomial":
                self.distribution_output = NegativeBinomialOutput(dim=dim)
            else:
                raise ValueError(
                    f"Unknown distribution output {config.distribution_output}"
                )

        self.model = PatchTSMixerModel(config)
        self.head = PatchTSMixerForecastHead(
            config=config,
            distribution_output=self.distribution_output,
        )

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=PatchTSMixerForForecastOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        target_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
    ) -> PatchTSMixerForForecastOutput:
        r"""

        Returns:

        """

        # past_values: tensor [batch_size x seq_len x num_input_channels]
        model_output = self.model(
            past_values,
            observed_mask=observed_mask,
            output_hidden_states=output_hidden_states,
        )  # model_output: [batch_size x nvars x num_patch x num_features]
        y_hat = self.head(
            model_output.last_hidden_state,
        )  # tensor [batch_size x forecast_len x num_input_channels]
        
        loss_val = None
        if self.config.forecast_channel_indices is not None:
            if self.distribution_output:
                # if self.config.distribution_output == "negative_binomial" and torch.any(target_values < 0):
                #     raise Exception("target_values cannot be negative for negative_binomial distribution.")
                distribution = self.distribution_output.distribution(
                    y_hat,
                    loc=model_output.loc[..., self.config.forecast_channel_indices],
                    scale=model_output.scale[..., self.config.forecast_channel_indices],
                )
                if target_values is not None and return_loss is True:
                    loss_val = self.loss(
                        distribution,
                        target_values[..., self.config.forecast_channel_indices],
                    )
                    # take average of the loss
                    loss_val = weighted_average(loss_val)
            else:
                # y_hat = (
                #     y_hat
                #     * model_output.scale[..., self.config.forecast_channel_indices]
                #     + model_output.loc[..., self.config.forecast_channel_indices]
                # )
                if target_values is not None and return_loss is True:
                    loss_val = self.loss(
                        y_hat, target_values[..., self.config.forecast_channel_indices]
                    )

        else:
            if self.distribution_output:
                # if self.config.distribution_output == "negative_binomial" and torch.any(target_values < 0):
                #     raise Exception("target_values cannot be negative for negative_binomial distribution.")
                distribution = self.distribution_output.distribution(
                    y_hat, loc=model_output.loc, scale=model_output.scale
                )
                if target_values is not None and return_loss is True:
                    loss_val = self.loss(distribution, target_values)
                    loss_val = weighted_average(loss_val)
            else:
                y_hat = y_hat.reshape(y_hat.shape[0], y_hat.shape[1] // 3, y_hat.shape[2]*3)
                y_hat = y_hat * model_output.scale + model_output.loc
                y_hat = y_hat.reshape(y_hat.shape[0], y_hat.shape[1] * 3, y_hat.shape[2]//3)
                y_hat = y_hat.reshape(y_hat.shape[0], y_hat.shape[1] // 3, 3, y_hat.shape[2])
                y_hat = y_hat.sum(dim=2)
                if target_values is not None and return_loss is True:
                    loss_val = self.loss(y_hat, target_values)

        if self.config.forecast_channel_indices is not None:
            loc = model_output.loc[..., self.config.forecast_channel_indices]
            scale = model_output.scale[..., self.config.forecast_channel_indices]
        else:
            loc = model_output.loc
            scale = model_output.scale

        return PatchTSMixerForForecastOutput(
            prediction_logits=y_hat,  # tensor [batch_size x forecast_len x num_input_channels]
            last_hidden_state=model_output.last_hidden_state,  # x: [batch_size x nvars x num_patch x num_features]
            hidden_states=model_output.hidden_states,
            loss=loss_val,
            loc=loc,
            scale=scale,
        )

    def generate(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> SamplePatchTSMixerForecastOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Parameters:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the future.

            observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Return:
            [`SamplePatchTSMixerForecastOutput`] where the outputs `sequences` tensor will have shape `(batch_size,
            number of samples, prediction_length, num_input_channels)`.
        """
        # get number of samples
        num_parallel_samples = self.config.num_parallel_samples

        # get model output
        outputs = self(
            past_values=past_values,
            target_values=None,
            observed_mask=observed_mask,
            output_hidden_states=False,
        )

        # get distribution

        distribution = self.distribution_output.distribution(
            outputs.prediction_logits, loc=outputs.loc, scale=outputs.scale
        )

        # get samples
        samples = [
            distribution.sample() for _ in range(num_parallel_samples)
        ]  # samples: list of [batch_size x forecast_len x num_channels]
        # stack tensors
        samples = torch.stack(
            samples, dim=1
        )  # [batch_size x num_samples x forecast_len x num_channels]
        return SamplePatchTSMixerForecastOutput(sequences=samples)


@dataclass
class PatchTSMixerForClassificationOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForClassificationOutput`].

    Args:
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
            Prediction output from the classfication head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, num_features)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
    """

    prediction_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None


class PatchTSMixerForClassification(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for classification application.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        self.model = PatchTSMixerModel(config)
        self.head = PatchTSMixerLinearHead(
            config=config,
        )
        self.loss = torch.nn.CrossEntropyLoss()

        if config.scaling in ["std", "mean", True]:
            if config.mode == "flatten":
                raise ValueError(
                    "Scaling is not supported for classification task when mode == flatten"
                )
            self.inject_scale = InjectScalerStatistics4D(
                num_features=config.num_features, num_patches=config.num_patches
            )
        else:
            self.inject_scale = None

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=PatchTSMixerForClassificationOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        past_values: torch.Tensor,
        target_values: torch.Tensor = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
    ) -> PatchTSMixerForClassificationOutput:
        r"""

        Returns:

        """

        model_output = self.model(
            past_values,
            output_hidden_states=output_hidden_states,
        )  # x: [batch_size x nvars x num_patch x num_features]

        if self.inject_scale is not None:
            model_output.last_hidden_state = self.inject_scale(
                model_output.last_hidden_state,
                loc=model_output.loc,
                scale=model_output.scale,
            )  # x: [batch_size x nvars x num_patch x num_features]

        y_hat = self.head(
            model_output.last_hidden_state
        )  # tensor [batch_size x n_labels]

        if target_values is not None and return_loss is True:
            loss_val = self.loss(y_hat, target_values)
        else:
            loss_val = None

        return PatchTSMixerForClassificationOutput(
            prediction_logits=y_hat,  # tensor [batch_size x n_labels]
            last_hidden_state=model_output.last_hidden_state,  # x: [batch_size x nvars x num_patch x num_features]
            hidden_states=model_output.hidden_states,
            loss=loss_val,
        )


@dataclass
class PatchTSMixerForRegressionOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForRegressionOutput`].

    Args:
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, num_targets)`):
            Prediction output from the regression head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, num_features)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
    """

    prediction_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None


class PatchTSMixerForRegression(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for regression application.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        self.model = PatchTSMixerModel(config)
        self.config = config
        if config.loss == "mse":
            self.loss = nn.MSELoss(reduction="mean")
            self.distribution_output = None
        else:
            self.loss = nll

            if config.distribution_output == "student_t":
                self.distribution_output = StudentTOutput(dim=config.num_targets)
            elif config.distribution_output == "normal":
                self.distribution_output = NormalOutput(dim=config.num_targets)
            elif config.distribution_output == "negative_binomial":
                self.distribution_output = NegativeBinomialOutput(
                    dim=config.num_targets
                )
            else:
                raise ValueError(
                    f"Unknown distribution output {config.distribution_output}"
                )

        if config.scaling in ["std", "mean", True]:
            if config.mode == "flatten":
                raise ValueError(
                    "Scaling is not supported for classification task when mode == flatten"
                )
            self.inject_scale = InjectScalerStatistics4D(
                num_features=config.num_features, num_patches=config.num_patches
            )
        else:
            self.inject_scale = None

        self.head = PatchTSMixerLinearHead(
            config=config,
            distribution_output=self.distribution_output,
        )

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=PatchTSMixerForRegressionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        target_values: torch.Tensor = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
    ) -> PatchTSMixerForRegressionOutput:
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            target_values (`torch.Tensor` of shape `(batch_size, num_targets)`, *optional*):
                target labels associates with the `past_values`
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
            return_loss (`bool`, default to True): Whether or not to calculate the loss value.

        Returns:
            `PatchTSTForRegressionOutput` or tuple of `torch.Tensor` (if `return_dict`=False or
            `config.return_dict`=False)

        """
        model_output = self.model(
            past_values,
            output_hidden_states=output_hidden_states,
        )  # model_output: [batch_size x nvars x num_patch x num_features]

        if self.inject_scale is not None:
            model_output.last_hidden_state = self.inject_scale(
                model_output.last_hidden_state,
                loc=model_output.loc,
                scale=model_output.scale,
            )  # x: [batch_size x nvars x num_patch x num_features]

        y_hat = self.head(
            model_output.last_hidden_state
        )  # tensor [batch_size x num_targets]

        if target_values is not None and return_loss is True:
            if self.distribution_output:
                if (
                    self.config.distribution_output == "negative_binomial"
                    and torch.any(target_values < 0)
                ):
                    raise Exception(
                        "target_values cannot be negative for negative_binomial distribution."
                    )
                distribution = self.distribution_output.distribution(y_hat)
                loss_val = self.loss(distribution, target_values)
                # take average of the loss
                loss_val = weighted_average(loss_val)
            else:
                loss_val = self.loss(y_hat, target_values)
        else:
            loss_val = None

        return PatchTSMixerForRegressionOutput(
            prediction_logits=y_hat,  # tensor [batch_size x num_targets]
            last_hidden_state=model_output.last_hidden_state,  # x: [batch_size x nvars x num_patch x num_features]
            hidden_states=model_output.hidden_states,
            loss=loss_val,
        )

    def generate(
        self,
        past_values: torch.Tensor,
    ) -> SamplePatchTSMixerRegressionOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Parameters:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the future.

        Return:
            [`SamplePatchTSMixerRegressionOutput`] where the outputs `sequences` tensor will have shape `(batch_size,
            number of samples, num_targets)`.
        """
        # get number of samples
        num_parallel_samples = self.config.num_parallel_samples

        # get model output
        outputs = self(
            past_values=past_values,
            target_values=None,
            output_hidden_states=False,
        )

        # get distribution
        distribution = self.distribution_output.distribution(outputs.prediction_logits)

        # get samples
        samples = [
            distribution.sample() for _ in range(num_parallel_samples)
        ]  # samples: list of [batch_size x num_targets]
        # stack tensors
        samples = torch.stack(
            samples, dim=1
        )  # [batch_size x num_samples x num_targets]
        return SamplePatchTSMixerRegressionOutput(sequences=samples)
