from monai.networks.nets.basic_unetplusplus import BasicUNetPlusPlus
model = BasicUNetPlusPlus(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    features=(32, 32, 64, 128, 256, 32),
)