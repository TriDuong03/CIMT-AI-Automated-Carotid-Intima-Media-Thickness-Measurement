from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch import nn

model = PlainConvUNet(
    input_channels=1,
    n_stages=6,
    features_per_stage=(32, 64, 128, 256, 320, 320),
    conv_op=nn.Conv2d,
    kernel_sizes=(3, 3, 3, 3, 3, 3),
    strides=(1, 2, 2, 2, 2, 2),
    n_conv_per_stage=(2, 2, 2, 2, 2, 2),
    num_classes=2,
    n_conv_per_stage_decoder=(2, 2, 2, 2, 2),
    conv_bias=True,
    norm_op=nn.InstanceNorm2d,
    norm_op_kwargs={
        "eps": 1e-5,
        "affine": True,
    },
    dropout_op=None,
    dropout_op_kwargs=None,
    nonlin=nn.LeakyReLU,
    nonlin_kwargs={
        "negative_slope": 1e-2,
        "inplace": True,
    },
    deep_supervision=False,
)