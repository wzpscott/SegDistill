_base_ = [
    '../_base_/datasets/ade20k_repeat.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
weight = 1
model = dict(
    type='SDModule',
    cfg_s=dict(
        type='EncoderDecoder',
        pretrained='pretrained/mit_b0.pth',
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
            type='mit_b3',
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
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ),
    distillation = [
        {'student_layer':['backbone.block1.1.attn.ATTN','backbone.block1.1.attn.V'],
        'teacher_layer':['backbone.block1.2.attn.ATTN','backbone.block1.2.attn.V'],
        'loss_name':'StudentRE',
        'loss_config':{
            'weight':weight,
            'tau':1,
            'transform_config':{'loss_type':'channel','group_size':1},
            'proj':'backbone.block1.1.attn.proj',
            'earlystop_config':120000,
            },
        },
        {'student_layer':['backbone.block2.1.attn.ATTN','backbone.block2.1.attn.V'],
        'teacher_layer':['backbone.block2.3.attn.ATTN','backbone.block2.3.attn.V'],
        'loss_name':'StudentRE',
        'loss_config':{
            'weight':weight,
            'tau':1,
            'transform_config':{'loss_type':'channel','group_size':1},
            'proj':'backbone.block2.1.attn.proj',
            'earlystop_config':120000,
            },
        },
        {'student_layer':['backbone.block3.1.attn.ATTN','backbone.block3.1.attn.V'],
        'teacher_layer':['backbone.block3.17.attn.ATTN','backbone.block3.17.attn.V'],
        'loss_name':'StudentRE',
        'loss_config':{
            'weight':weight,
            'tau':1,
            'transform_config':{'loss_type':'channel','group_size':1},
            'proj':'backbone.block3.1.attn.proj',
            'earlystop_config':120000,
            },
        },
        {'student_layer':['backbone.block4.1.attn.ATTN','backbone.block4.1.attn.V'],
        'teacher_layer':['backbone.block4.2.attn.ATTN','backbone.block4.2.attn.V'],
        'loss_name':'StudentRE',
        'loss_config':{
            'weight':weight,
            'tau':1,
            'transform_config':{'loss_type':'channel','group_size':1},
            'proj':'backbone.block4.1.attn.proj',
            'earlystop_config':120000,
            },
        },
        {'student_layer':['backbone.block1.1.attn.ATTN','backbone.block1.1.attn.V'],
        'teacher_layer':['backbone.block1.2.attn.ATTN','backbone.block1.2.attn.V'],
        'loss_name':'TeacherRE',
        'loss_config':{
            'weight':weight,
            'tau':1,
            'transform_config':{'loss_type':'channel','group_size':1},
            'proj':'backbone.block1.2.attn.proj',
            'earlystop_config':120000,
            },
        },
        {'student_layer':['backbone.block2.1.attn.ATTN','backbone.block2.1.attn.V'],
        'teacher_layer':['backbone.block2.3.attn.ATTN','backbone.block2.3.attn.V'],
        'loss_name':'TeacherRE',
        'loss_config':{
            'weight':weight,
            'tau':1,
            'transform_config':{'loss_type':'channel','group_size':1},
            'proj':'backbone.block2.3.attn.proj',
            'earlystop_config':120000,
            },
        },
        {'student_layer':['backbone.block3.1.attn.ATTN','backbone.block3.1.attn.V'],
        'teacher_layer':['backbone.block3.17.attn.ATTN','backbone.block3.17.attn.V'],
        'loss_name':'TeacherRE',
        'loss_config':{
            'weight':weight,
            'tau':1,
            'transform_config':{'loss_type':'channel','group_size':1},
            'proj':'backbone.block3.17.attn.proj',
            'earlystop_config':120000,
            },
        },
        {'student_layer':['backbone.block4.1.attn.ATTN','backbone.block4.1.attn.V'],
        'teacher_layer':['backbone.block4.2.attn.ATTN','backbone.block4.2.attn.V'],
        'loss_name':'TeacherRE',
        'loss_config':{
            'weight':weight,
            'tau':1,
            'transform_config':{'loss_type':'channel','group_size':1},
            'proj':'backbone.block4.2.attn.proj',
            'earlystop_config':120000,
            },
        },
    ],
    t_pretrain = './pretrained/segformer.b3.512x512.ade.160k.pth',  # 老师的预训练模型
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

work_dir = '/apdcephfs/private_inchzhang/shared_info/11.1/s+t_stage1234_wol'

data = dict(samples_per_gpu=2)
evaluation = dict(interval=2000, metric='mIoU') 

resume_from = work_dir+'/latest.pth'
