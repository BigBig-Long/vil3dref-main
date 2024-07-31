from torch.optim import Adam, Adamax  # 导入必要的优化器
from .adamw import AdamW  # 从本地模块导入AdamW优化器
from .rangerlars import RangerLars  # 从本地模块导入RangerLars优化器

def build_optimizer(model, opts):
    param_optimizer = list(model.named_parameters())  # 获取模型的所有命名参数
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  # 不应用权重衰减的参数
    obj_enc_params, txt_enc_params, other_params = {}, {}, {}  # 分别存储不同部分的参数

    for n, p in param_optimizer:  # 遍历所有参数
        if not p.requires_grad:  # 如果参数不需要计算梯度，则跳过
            continue
        if 'obj_encoder' in n:  # 如果参数名中包含'obj_encoder'，则归入对象编码器参数
            obj_enc_params[n] = p
        elif 'txt_encoder' in n:  # 如果参数名中包含'txt_encoder'，则归入文本编码器参数
            txt_enc_params[n] = p
        else:  # 其他参数
            other_params[n] = p

    optimizer_grouped_parameters = []  # 用于存储优化器参数组的列表
    init_lrs = []  # 用于存储初始学习率的列表

    # 根据不同的参数类型准备参数组
    for ptype, pdict in [('obj', obj_enc_params), ('txt', txt_enc_params), ('others', other_params)]:
        if len(pdict) == 0:  # 如果参数组为空，则跳过
            continue
        init_lr = opts.learning_rate  # 从选项中获取默认学习率
        if ptype == 'obj':  # 调整对象编码器参数的学习率
            init_lr = init_lr * getattr(opts, 'obj_encoder_lr_multi', 1)
        elif ptype == 'txt':  # 调整文本编码器参数的学习率
            init_lr = init_lr * getattr(opts, 'txt_encoder_lr_multi', 1)

        # 定义不需要权重衰减的参数组
        optimizer_grouped_parameters.extend([
            {'params': [p for n, p in pdict.items() if not any(nd in n for nd in no_decay)],
             'weight_decay': opts.weight_decay, 'lr': init_lr},
            {'params': [p for n, p in pdict.items() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': init_lr}
        ])
        init_lrs.extend([init_lr] * 2)  # 添加初始学习率

    # 根据选项选择优化器类
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    elif opts.optim == 'rangerlars':
        OptimCls = RangerLars
    else:
        raise ValueError('invalid optimizer')  # 如果优化器类型不支持，抛出错误

    # 创建优化器实例
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    return optimizer, init_lrs  # 返回优化器和初始学习率列表
