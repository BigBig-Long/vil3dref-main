import math  # 导入math库进行数学计算
import torch  # 导入torch库，用于深度学习操作
from torch.optim.optimizer import Optimizer, required  # 从torch.optim.optimizer模块导入Optimizer类和required对象

class RAdam(Optimizer):  # 定义RAdam类，继承自Optimizer
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):  # 初始化方法，设置默认参数
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)  # 将默认参数值保存在字典中
        self.buffer = [[None, None, None] for ind in range(10)]  # 创建一个包含10个[None, None, None]列表的列表
        super(RAdam, self).__init__(params, defaults)  # 调用父类的初始化方法

    def __setstate__(self, state):  # 设置状态方法，用于序列化
        super(RAdam, self).__setstate__(state)  # 调用父类的__setstate__方法

    def step(self, closure=None):  # 定义优化步骤方法
        loss = None  # 初始化损失为None
        if closure is not None:  # 如果闭包不为None
            loss = closure()  # 调用闭包计算损失
        for group in self.param_groups:  # 遍历所有参数组
            for p in group['params']:  # 遍历参数组中的所有参数
                if p.grad is None:  # 如果参数的梯度不存在
                    continue  # 跳过当前循环
                grad = p.grad.data.float()  # 获取梯度并转换为float类型
                if grad.is_sparse:  # 如果梯度是稀疏的
                    raise RuntimeError('RAdam does not support sparse gradients')  # 抛出不支持稀疏梯度的异常
                p_data_fp32 = p.data.float()  # 获取参数数据并转换为float类型
                state = self.state[p]  # 获取参数的状态
                if len(state) == 0:  # 如果状态为空
                    state['step'] = 0  # 设置步数为0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)  # 创建和参数数据形状相同的全0张量作为一阶矩估计
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)  # 创建和参数数据形状相同的全0张量作为二阶矩估计
                else:  # 如果状态不为空
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)  # 将一阶矩估计的数据类型转换为参数数据的类型
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)  # 将二阶矩估计的数据类型转换为参数数据的类型
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']  # 获取一阶和二阶矩估计
                beta1, beta2 = group['betas']  # 获取beta1和beta2参数

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)  # 更新二阶矩估计
                exp_avg.mul_(beta1).add_(1 - beta1, grad)  # 更新一阶矩估计
                state['step'] += 1  # 步数加1

                buffered = self.buffer[int(state['step'] % 10)]  # 从buffer中获取对应的列表
                if state['step'] == buffered[0]:  # 如果步数与buffer中的步数相同
                    N_sma, step_size = buffered[1], buffered[2]  # 获取N_sma和step_size
                else:  # 如果步数不同
                    buffered[0] = state['step']  # 更新buffer中的步数
                    beta2_t = beta2 ** state['step']  # 计算beta2的state['step']次幂
                    N_sma_max = 2 / (1 - beta2) - 1  # 计算N_sma的最大值
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)  # 计算N_sma
                    buffered[1] = N_sma  # 更新buffer中的N_sma
                    # 更保守的做法，因为这是一个近似值
                    if N_sma >= 5:  # 如果N_sma大于等于5
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:  # 如果N_sma小于5
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size  # 更新buffer中的step_size
                if group['weight_decay'] != 0:  # 如果权重衰减不为0
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)  # 应用权重衰减
                # 更保守的做法，因为这是一个近似值
                if N_sma >= 5:  # 如果N_sma大于等于5
                    denom = exp_avg_sq.sqrt().add_(group['eps'])  # 计算分母
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)  # 更新参数数据
                else:  # 如果N_sma小于5
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)  # 更新参数数据
                p.data.copy_(p_data_fp32)  # 将更新后的参数数据复制回原参数
        return loss  # 返回损失

# 同样的结构用于定义PlainRAdam和AdamW类，具体实现细节略有不同，但大体框架一致
# PlainRAdam类的定义，继承自Optimizer
class PlainRAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)  # 初始化默认参数
        super(PlainRAdam, self).__init__(params, defaults)  # 调用父类构造函数，传入参数和默认值

    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)  # 用于处理pickle反序列化

    def step(self, closure=None):
        loss = None  # 初始损失设为None
        if closure is not None:  # 如果闭包不为空，则计算损失
            loss = closure()
        for group in self.param_groups:  # 遍历所有参数组
            for p in group['params']:  # 遍历组内参数
                if p.grad is None:
                    continue  # 如果参数梯度不存在，跳过此次循环
                grad = p.grad.data.float()  # 获取参数的梯度并转换为float
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')  # 不支持稀疏梯度
                p_data_fp32 = p.data.float()  # 将参数值转为float类型
                state = self.state[p]  # 获取参数状态
                if len(state) == 0:  # 如果状态为空，初始化状态
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)  # 初始化一阶动量
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)  # 初始化二阶动量
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']  # 获取一阶和二阶动量
                beta1, beta2 = group['betas']  # 获取beta参数

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)  # 更新二阶动量
                exp_avg.mul_(beta1).add_(1 - beta1, grad)  # 更新一阶动量
                state['step'] += 1  # 更新步数

                beta2_t = beta2 ** state['step']  # 计算beta2的t次幂
                N_sma_max = 2 / (1 - beta2) - 1  # 计算N_sma的最大值
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)  # 计算N_sma

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)  # 应用权重衰减

                if N_sma >= 5:  # 如果N_sma大于等于5，使用RAdam的更新公式
                    step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:  # 如果N_sma小于5，不使用二阶动量
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)  # 将更新后的参数值复制回原参数

        return loss  # 返回损失值
# AdamW类的定义，继承自Optimizer
class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, warmup=warmup)  # 设置默认参数
        super(AdamW, self).__init__(params, defaults)  # 调用父类的构造函数来初始化参数

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)  # 序列化状态时调用

    def step(self, closure=None):
        loss = None  # 初始化损失为None
        if closure is not None:  # 如果提供了闭包函数，计算损失
            loss = closure()
        for group in self.param_groups:  # 遍历所有参数组
            for p in group['params']:  # 遍历组内的所有参数
                if p.grad is None:  # 如果参数没有梯度，跳过
                    continue
                grad = p.grad.data.float()  # 获取梯度，并转换为float类型
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')  # 不支持稀疏梯度
                p_data_fp32 = p.data.float()  # 将参数转换为float类型
                state = self.state[p]  # 获取参数的状态
                if len(state) == 0:  # 如果状态为空，则初始化状态
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)  # 初始化一阶矩估计
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)  # 初始化二阶矩估计
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']  # 获取一阶和二阶矩估计
                beta1, beta2 = group['betas']  # 获取beta系数

                state['step'] += 1  # 步数增加
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)  # 更新二阶矩估计
                exp_avg.mul_(beta1).add_(1 - beta1, grad)  # 更新一阶矩估计
                denom = exp_avg_sq.sqrt().add_(group['eps'])  # 计算分母，用于更新参数

                bias_correction1 = 1 - beta1 ** state['step']  # 计算一阶矩估计的偏差修正
                bias_correction2 = 1 - beta2 ** state['step']  # 计算二阶矩估计的偏差修正
                if group['warmup'] > state['step']:  # 如果处于预热期
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']  # 计算预热期的学习率
                else:
                    scheduled_lr = group['lr']  # 否则使用正常的学习率

                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1  # 计算步长
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)  # 应用权重衰减

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)  # 根据AdamW的公式更新参数
                p.data.copy_(p_data_fp32)  # 将更新后的参数值复制回原参数

        return loss  # 返回计算的损失
