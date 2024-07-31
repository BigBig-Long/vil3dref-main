import math  # 导入math库，用于数学计算

# 用于定义noam_schedule函数，该函数实现了原始Transformer中使用的学习率调度策略
def noam_schedule(step, warmup_step=4000):
    """ original Transformer schedule"""
    if step <= warmup_step:
        return step / warmup_step
    return (warmup_step ** 0.5) * (step ** -0.5)

# 用于定义warmup_linear函数，该函数实现了BERT中使用的线性预热+线性衰减的学习率调度策略
def warmup_linear(step, warmup_step, tot_step):
    """ BERT schedule """
    if step < warmup_step:
        return step / warmup_step
    return max(0, (tot_step - step) / (tot_step - warmup_step))

# 用于定义warmup_cosine函数，该函数实现了余弦退火策略，先线性预热，之后余弦衰减
def warmup_cosine(step, warmup_step, tot_step):
    if step < warmup_step:
        return step / warmup_step
    return 0.5 * (1 + math.cos((step - warmup_step) / (tot_step - warmup_step) * math.pi))

# 用于获取当前步骤的学习率，根据选定的衰减函数调整学习率
def get_lr_sched(global_step, opts):
    # 根据opts中的lr_decay选择不同的学习率调整策略
    if opts.lr_decay == 'linear':
        lr_decay_fn = warmup_linear
    elif opts.lr_decay == 'cosine':
        lr_decay_fn = warmup_cosine
    lr_this_step = opts.learning_rate * lr_decay_fn(
        global_step, opts.warmup_steps, opts.num_train_steps)
    if lr_this_step <= 0:
        lr_this_step = 1e-8
    return lr_this_step

# 类似于get_lr_sched，但返回的是衰减率而不是实际的学习率
def get_lr_sched_decay_rate(global_step, opts):
    if opts.lr_decay == 'linear':
        lr_decay_fn = warmup_linear
    elif opts.lr_decay == 'cosine':
        lr_decay_fn = warmup_cosine
    lr_decay_rate = lr_decay_fn(
        global_step, opts.warmup_steps, opts.num_train_steps)
    lr_decay_rate = max(lr_decay_rate, 1e-5)
    return lr_decay_rate
