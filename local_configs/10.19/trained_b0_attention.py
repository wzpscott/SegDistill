_base_ = [
    '../_base_/datasets/ade20k_repeat.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)

attention_config = {
        'weight':0.1,
        'tau':1,
        'reshape_config':'attention',
        'resize_config':False,
        'mask_config':False,
        'transform_config':{'loss_type':'channel','group_size':1},
        'ff_config':False,
        },


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
        {'student_layer':'backbone.block1.0.attn.ATTN',
        'teacher_layer':'backbone.block1.0.attn.ATTN',
        'loss_name':'KLDLoss',
        'loss_config':attention_config
        },
        {'student_layer':'backbone.block1.1.attn.ATTN',
        'teacher_layer':'backbone.block1.2.attn.ATTN',
        'loss_name':'KLDLoss',
        'loss_config':attention_config
        },
        {'student_layer':'backbone.block2.0.attn.ATTN',
        'teacher_layer':'backbone.block2.0.attn.ATTN',
        'loss_name':'KLDLoss',
        'loss_config':attention_config
        },
        {'student_layer':'backbone.block2.1.attn.ATTN',
        'teacher_layer':'backbone.block2.7.attn.ATTN',
        'loss_name':'KLDLoss',
        'loss_config':attention_config
        },
        {'student_layer':'backbone.block3.0.attn.ATTN',
        'teacher_layer':'backbone.block3.0.attn.ATTN',
        'loss_name':'KLDLoss',
        'loss_config':attention_config
        },
        {'student_layer':'backbone.block3.1.attn.ATTN',
        'teacher_layer':'backbone.block3.26.attn.ATTN',
        'loss_name':'KLDLoss',
        'loss_config':attention_config
        },
        {'student_layer':'backbone.block4.0.attn.ATTN',
        'teacher_layer':'backbone.block4.0.attn.ATTN',
        'loss_name':'KLDLoss',
        'loss_config':attention_config
        },
        {'student_layer':'backbone.block4.1.attn.ATTN',
        'teacher_layer':'backbone.block4.2.attn.ATTN',
        'loss_name':'KLDLoss',
        'loss_config':attention_config
        },
    ],
    s_pretrain = 'pretrained/segformer.b0.512x512.ade.160k.pth', # 学生的预训练模型
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
evaluation = dict(interval=16000, metric='mIoU')  
work_dir = '/apdcephfs/private_inchzhang/shared_info/10.19/attention_trained'
# resume_from = ''

