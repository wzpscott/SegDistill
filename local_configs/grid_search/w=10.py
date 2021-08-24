_base_ = [
    '../_base_/models/distill.py',
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
work_dir = './work_dirs/w=10'

model = dict(
        distillation = dict(
        # layers表示要进行蒸馏的层，[teacher_layer,student_layer,[teacher_channel,student_channel],teacher_dim]
        # 其中[teacher_channel,student_channel]和teacher_dim是Adaptor的参数
        layers=[
            ['decode_head.linear_pred','decode_head.linear_pred',[150,150],4]  
        ],
        # weights_init_strategy,parse_mode,use_attn是之前实验留下的参数
        weights_init_strategy='equal',
        parse_mode='regular',
        use_attn=False,
        # selective是指对logits层蒸馏的策略，有如下取值:
        # 1. none: 不进行logits层蒸馏,就是baseline
        # 2. distill: 对 logits 蒸馏
        # 3. distill_0: 去除logits层所有结果为255的pixel之后进行蒸馏
        # 4. distill_1: 去除logits层所有结果为255的pixel+teacher预测错误的pixel+student预测正确的pixel 之后进行蒸馏
        # 5. distill_2: 去除logits层所有结果为255的pixel+teacher预测错误的pixel 之后进行蒸馏
        selective='distill',T=1,weight=10
    ),
    s_pretrain = './pretrained/mit_b1.pth', # 学生的预训练模型
    t_pretrain = './pretrained/segformer.b4.512x512.ade.160k.pth'  # 老师的预训练模型
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