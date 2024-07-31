import torch, math  # 导入torch库和math库
from torch.optim.optimizer import Optimizer  # 从torch.optim模块导入Optimizer基类

# Ralamb类，结合了RAdam和LARS优化算法
class Ralamb(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)  # 定义默认参数
        self.buffer = [[None, None, None] for ind in range(10)]  # 创建一个缓冲区，用于存储中间计算结果
        super(Ralamb, self).__init__(params, defaults)  # 调用父类构造函数初始化优化器

    def __setstate__(self, state):
        super(Ralamb, self).__setstate__(state)  # 用于pickle序列化

    def step(self, closure=None):
        loss = None
        if closure is not None:  # 如果提供了闭包函数，调用闭包计算损失
            loss = closure()
        for group in self.param_groups:  # 遍历所有参数组
            for p in group['params']:  # 遍历组内的所有参数
                if p.grad is None:  # 如果参数没有梯度，跳过
                    continue
                grad = p.grad.data.float()  # 将梯度转换为float类型
                if grad.is_sparse:  # 如果梯度是稀疏的，抛出异常
                    raise RuntimeError('Ralamb does not support sparse gradients')
                p_data_fp32 = p.data.float()  # 将参数数据转换为float类型
                state = self.state[p]  # 获取参数的状态信息
                if len(state) == 0:  # 如果状态为空，初始化状态
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)  # 初始化一阶矩估计
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)  # 初始化二阶矩估计
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                # 更新一阶和二阶矩估计
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:  # 如果步数与缓冲区索引相同，使用缓冲区中的值
                    N_sma, radam_step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma >= 5:
                        radam_step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        radam_step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = radam_step_size
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])
                radam_step = p_data_fp32.clone()
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    radam_step.addcdiv_(-radam_step_size * group['lr'], exp_avg, denom)
                else:
                    radam_step.add_(exp_avg, alpha=-radam_step_size * group['lr'])
                radam_norm = radam_step.pow(2).sum().sqrt()
                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
                if weight_norm == 0 or radam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / radam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = radam_norm
                state['trust_ratio'] = trust_ratio
                if N_sma >= 5:
                    p_data_fp32.addcdiv_(-radam_step_size * group['lr'] * trust_ratio, exp_avg, denom)
                else:
                    p_data_fp32.add_(-radam_step_size * group['lr'] * trust_ratio, exp_avg)
                p.data.copy_(p_data_fp32)  # 将更新后的参数复制回原始参数
        return loss  # 返回损失
