import torch
import math
import copy
import warnings
import torchvision
from functools import partial
from torch import nn, Tensor
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from typing import Any, Callable, Dict, List, Optional, Sequence

# from torchvision.models.mobilenetv2 import _make_divisible #ConvBNActivation
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2d(nn.Conv2d):  # For Weight Standardization
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, x):
        weight = self.weight
        weight_mean = (
            weight.mean(dim=1, keepdim=True)
            .mean(dim=2, keepdim=True)
            .mean(dim=3, keepdim=True)
        )
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def conv3x3(
    in_planes,
    out_planes,
    kernel_size=3,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
):
    """3x3 convolution with padding"""
    return Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


class ConvBNActivation(nn.Sequential):  # using BatchNorm # NB: do not apply WS into BN
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.GELU
        super().__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_planes),  # BN
            activation_layer(),  # GELU
        )
        self.out_channels = out_planes


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class ConvGNActivation(nn.Sequential):  # using GroupNorm or LayerNorm
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.GroupNorm
        if activation_layer is None:
            activation_layer = nn.GELU
        super().__init__(
            conv3x3(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            norm_layer(8, out_planes, eps=1e-3),  # GN
            activation_layer(),  # GELU
        )
        self.out_channels = out_planes


def conv2d_init(m):  # Kaiming He initialization
    assert isinstance(m, nn.Conv2d)
    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    m.weight.data.normal_(0, math.sqrt(2.0 / n))


def gn_init(m, zero_init=False):
    assert isinstance(m, nn.GroupNorm)
    m.weight.data.fill_(0.0 if zero_init else 1.0)
    m.bias.data.zero_()


class SqueezeExcitation(nn.Module):
    # Implemented as described at Figure 4 of the MobileNetV3 paper
    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.act1 = nn.GELU()
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input: Tensor, inplace: bool) -> Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.act1(scale)
        scale = self.fc2(scale)
        return F.hardsigmoid(
            scale, inplace=inplace
        )  # --> this is helpful, better than GELU

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input, True)
        return scale * input


class InvertedResidualConfig:
    # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        use_bn: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_bn = use_bn
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        layers: List[nn.Module] = []
        activation_layer = nn.GELU

        if cnf.use_bn:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
            # expand
            if cnf.expanded_channels != cnf.input_channels:
                layers.append(
                    ConvBNActivation(
                        cnf.input_channels,
                        cnf.expanded_channels,
                        kernel_size=1,
                        norm_layer=norm_layer,
                        activation_layer=activation_layer,
                    )
                )
            # depthwise
            stride = 1 if cnf.dilation > 1 else cnf.stride
            layers.append(
                ConvBNActivation(
                    cnf.expanded_channels,
                    cnf.expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=stride,
                    dilation=cnf.dilation,
                    groups=cnf.expanded_channels,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )
            if cnf.use_se:
                layers.append(se_layer(cnf.expanded_channels))
            # project
            layers.append(
                ConvBNActivation(
                    cnf.expanded_channels,
                    cnf.out_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=nn.Identity,
                )
            )
        else:
            norm_layer = partial(nn.GroupNorm)
            # expand
            if cnf.expanded_channels != cnf.input_channels:
                layers.append(
                    ConvGNActivation(
                        cnf.input_channels,
                        cnf.expanded_channels,
                        kernel_size=1,
                        norm_layer=norm_layer,
                        activation_layer=activation_layer,
                    )
                )
            # depthwise
            stride = 1 if cnf.dilation > 1 else cnf.stride
            layers.append(
                ConvGNActivation(
                    cnf.expanded_channels,
                    cnf.expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=stride,
                    dilation=cnf.dilation,
                    groups=cnf.expanded_channels,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )
            if cnf.use_se:
                layers.append(se_layer(cnf.expanded_channels))
            # project
            layers.append(
                ConvGNActivation(
                    cnf.expanded_channels,
                    cnf.out_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=nn.Identity,
                )
            )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1960,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        output_dim=128,
        hidden_mlp=960,
        nmb_prototypes=1000,
        normalize=True,
        **kwargs: Any
    ) -> None:
        """
        MobileNet V3 main class
        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all(
                [
                    isinstance(s, InvertedResidualConfig)
                    for s in inverted_residual_setting
                ]
            )
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[InvertedResidualConfig]"
            )

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer1 = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
            norm_layer2 = partial(nn.GroupNorm)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvBNActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer1,
                activation_layer=nn.GELU,
            )
        )  # BN

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer=None))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            ConvGNActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer2,
                activation_layer=nn.GELU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Sequential(
        #    nn.Linear(lastconv_output_channels, last_channel), # 960 > 1280
        #    nn.GELU(),
        #    nn.Dropout(p=0.2, inplace=True),
        #    nn.Linear(last_channel, num_classes),
        # )

        # normalize output features
        self.l2norm = normalize
        # Add a linear classifier for classification
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, num_classes)
        )

        self.projection_head = nn.Sequential(
            nn.Linear(lastconv_output_channels, hidden_mlp),  # 960 > 960
            nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_mlp, output_dim),  # 960 > 128
        )
        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv2d_init(m)
            # elif isinstance(m, nn.Linear):
            #    #nn.init.normal_(m.weight, 0, 0.01)
            #    #nn.init.zeros_(m.bias)
            #    m.weight.data.normal_(0, 0.01)
            #    m.bias.data.zero_()

            elif isinstance(m, nn.GroupNorm):
                gn_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_backbone(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.classifier(x) # N x 1280
        return x

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in inputs]),
                return_counts=True,
            )[1],
            0,
        )
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.forward_backbone(torch.cat(inputs[start_idx:end_idx]))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx

            # Apply the linear classifier to the features
        class_scores = self.classifier(output)
        return self.forward_head(output)


def _mobilenet_v3_conf(
    arch: str,
    width_mult: float = 1.0,
    reduced_tail: bool = False,
    dilated: bool = False,
    **kwargs: Any
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(
        InvertedResidualConfig.adjust_channels, width_mult=width_mult
    )

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, True, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, True, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, True, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, True, "HS", 2, 1),  # C3
            ## above part belongs to feature extractor
            bneck_conf(80, 3, 200, 80, False, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, False, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, False, "HS", 1, 1),
            bneck_conf(
                112, 5, 672, 160 // reduce_divider, True, False, "HS", 2, dilation
            ),  # C4
            bneck_conf(
                160 // reduce_divider,
                5,
                960 // reduce_divider,
                160 // reduce_divider,
                True,
                False,
                "HS",
                1,
                dilation,
            ),
            bneck_conf(
                160 // reduce_divider,
                5,
                960 // reduce_divider,
                160 // reduce_divider,
                True,
                False,
                "HS",
                1,
                dilation,
            ),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(
                96 // reduce_divider,
                5,
                576 // reduce_divider,
                96 // reduce_divider,
                True,
                "HS",
                1,
                dilation,
            ),
            bneck_conf(
                96 // reduce_divider,
                5,
                576 // reduce_divider,
                96 // reduce_divider,
                True,
                "HS",
                1,
                dilation,
            ),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError("Unsupported model type {}".format(arch))

    return inverted_residual_setting, last_channel


def _mobilenet_v3_model(
    arch: str,
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
):
    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    if pretrained:
        print("Dont use Pretrained Checkpoints")
        # if model_urls.get(arch, None) is None:
        #    raise ValueError("No checkpoint is available for model type {}".format(arch))
        # state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        # model.load_state_dict(state_dict)
    return model


def mobilenet_v3_large(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_large"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, **kwargs)
    return _mobilenet_v3_model(
        arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs
    )


def mobilenet_v3_small(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> MobileNetV3:
    """
    Constructs a small MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_small"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, **kwargs)
    return _mobilenet_v3_model(
        arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs
    )


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out


### end ###
