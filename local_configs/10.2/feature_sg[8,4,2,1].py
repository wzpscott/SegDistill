_base_ = [
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
log_config = dict(  
    interval=50, 
    hooks=[
        # dict(type='TensorboardLoggerHook') 
        dict(type='TextLoggerHook')
    ])
work_dir = './work_dirs/feature_sg[8,4,2,1]'

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='SDModule',
    cfg_s=dict(
            type='EncoderDecoder',
            backbone=dict(
                type='mit_b0',
                style='pytorch'),
            decode_head=dict(
            type='SegFormerHead',
            in_channels=[32, 64, 160, 256],
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
    distillation = [
        # a list of dicts containing layer information you want to apply distillation .
        #     a dict contains:
        #     student_layer: the name of student layer
        #     teacher_layer: the name of teacher layer
        #     loss_name: the name of loss function you want to apply to the ouputs of those two layers.
        #     loss_config: a dict containing config for the loss function(weight, temperature for SoftMax etc)
        #     channel_nums: a tuple of (student_channel_num,teacher_channel_num). optional
        #                 If specified, using a 1x1 conv layer to upsample student layer's channel num 
        #                 when student layer and teacher layer 's outputs have different channel shapes
        {'student_layer':'backbone.block1.1.mlp.fc2',
        'teacher_layer':'backbone.block1.2.mlp.fc2',
        'loss_name':'SpatialGroupLossForFeature',
        'loss_config':{'weight':1,'tau':1,'kernel_size':8, 'dilation':1, 'padding':0,'stride':8},
        'channel_nums':[32,64],
        },
        {'student_layer':'backbone.block2.1.mlp.fc2',
        'teacher_layer':'backbone.block2.2.mlp.fc2',
        'loss_name':'SpatialGroupLossForFeature',
        'loss_config':{'weight':1,'tau':1,'kernel_size':4, 'dilation':1, 'padding':0,'stride':4},
        'channel_nums':[64,128],
        },
        {'student_layer':'backbone.block3.1.mlp.fc2',
        'teacher_layer':'backbone.block3.17.mlp.fc2',
        'loss_name':'SpatialGroupLossForFeature',
        'loss_config':{'weight':1,'tau':1,'kernel_size':2, 'dilation':1, 'padding':0,'stride':2},
        'channel_nums':[160,320],
        },
        {'student_layer':'backbone.block4.1.mlp.fc2',
        'teacher_layer':'backbone.block4.2.mlp.fc2',
        'loss_name':'SpatialGroupLossForFeature',
        'loss_config':{'weight':1,'tau':1,'kernel_size':1, 'dilation':1, 'padding':0,'stride':1},
        'channel_nums':[256,512],
        },

    ],
    s_pretrain = './pretrained/mit_b0.pth', # 学生的预训练模型
    t_pretrain = './pretrained/segformer.b4.512x512.ade.160k.pth',  # 老师的预训练模型
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
# resume_from = './work_dirs/20/attn/latest.pth'
