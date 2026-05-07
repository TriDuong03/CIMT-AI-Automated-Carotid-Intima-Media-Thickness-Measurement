from monai.networks.nets.swin_unetr import SwinUNETR
model = SwinUNETR(
    in_channels=1,
    out_channels=2,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    feature_size=48,
    norm_name="instance",
    drop_rate=0.0,
    attn_drop_rate=0.0,
    use_checkpoint=False,
    spatial_dims=2,
    patch_norm = True,
    use_v2= True,
)
