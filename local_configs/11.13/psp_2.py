_base_ = [
    '../_base_/datasets/ade20k_repeat.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='SDModule',
    cfg_s=dict(
        type='EncoderDecoder',
        # pretrained='pretrained/resnet50_v1c-2cccc1ad.pth',
        backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ),
    cfg_t=dict(
        type='EncoderDecoder',
        backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ),
    distillation = [
        {'student_layer':'backbone.layer2.2.bn3',
        'teacher_layer':'backbone.layer2.2.bn3',
        'loss_name':'MSE',
        'loss_config':{
            'weight':1,
            'ff_config':False,
            },
        },
    ],
    t_pretrain = './pretrained/pspnet_r101-d8_512x512_160k_ade20k_20200615_100650-967c316f.pth',  
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9,0.999), weight_decay=0.01,
                paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data = dict(samples_per_gpu=2)
evaluation = dict(interval=2000, metric='mIoU')  
work_dir = '/apdcephfs/private_inchzhang/shared_info/11.13/PSP_2'
resume_from = work_dir+'/latest.pth'
