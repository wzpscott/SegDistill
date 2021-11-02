_base_ = [
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
c = 15
s = 0
model = dict(
    type='SDModule',
    cfg_s=dict(
        type='EncoderDecoder',
        pretrained='pretrained/resnet50_v1c-2cccc1ad.pth',
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
    distillation = [
        {'student_layer':'decode_head.conv_seg',
        'teacher_layer':'decode_head.linear_pred',
        'loss_name':'KLDLoss',
        'loss_config':{
            'weight':1,
            'tau':1,
            'reshape_config':'logits',
            'resize_config':{'mode':'bilinear','align_corners':False},
            'mask_config':False,
            'transform_config':{'loss_type':'channel','group_size':c},
            'ff_config':False,
            # 'earlystop_config':120000,
            },
        },
        # {'student_layer':'decode_head.conv_seg',
        # 'teacher_layer':'decode_head.linear_pred',
        # 'loss_name':'KLDLoss',
        # 'loss_config':{
        #     'weight':1,
        #     'tau':1,
        #     'reshape_config':'logits',
        #     'resize_config':{'mode':'bilinear','align_corners':False},
        #     'mask_config':False,
        #     'transform_config':{'loss_type':'spatial','kernel_size':s,'stride':s},
        #     'ff_config':False,
        #     # 'earlystop_config':120000,
        #     },
        # },
    ],
    t_pretrain = './pretrained/segformer.b4.512x512.ade.160k.pth',  # 老师的预训练模型
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

work_dir = f'/apdcephfs/private_inchzhang/shared_info/11.2/psp_c{c}s{s}'

data = dict(samples_per_gpu=2)
evaluation = dict(interval=2000, metric='mIoU')  
resume_from = work_dir+'/latest.pth'
