norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='SDModule_',
    cfg=dict(
        type='EncoderDecoder',
    backbone=dict(
        type='mit_b1',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ),
    cfg_t=dict(
        type='EncoderDecoder',
        backbone=dict(
            type='mit_b4',
            style='pytorch'),
        decode_head=dict(
            type='SegFormerHead',
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=128,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            decoder_params=dict(embed_dim=768),
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ),
    distillation = dict(),
    s_pretrain=None,
    t_pretrain=None,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)