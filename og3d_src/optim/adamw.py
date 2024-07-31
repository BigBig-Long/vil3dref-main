import math  # 导入math库，用于数学运算
from typing import Callable, Iterable, Tuple  # 导入类型注解功能，以便在函数定义中指定参数类型
import torch  # 导入torch库，PyTorch的核心库，用于深度学习操作
from torch.optim import Optimizer  # 从torch.optim模块导入Optimizer类，作为自定义优化器的基类

class AdamW(Optimizer):
    """
    实现了带权重衰减修正的Adam算法。
    参数:
    - params: 优化参数集合或定义参数组的字典。
    - lr: 学习率，默认为0.001。
    - betas: Adam算法的beta系数，控制一阶和二阶矩的衰减速率。
    - eps: 数值稳定性的小常数。
    - weight_decay: 权重衰减系数。
    - correct_bias: 是否校正偏差。
    """
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],  # 参数集合
        lr: float = 1e-3,  # 学习率
        betas: Tuple[float, float] = (0.9, 0.999),  # beta系数
        eps: float = 1e-6,  # 稳定性常数
        weight_decay: float = 0.0,  # 权重衰减系数
        correct_bias: bool = True,  # 是否校正偏差
    ):
        # 参数有效性检查
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        # 设置默认参数配置
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)  # 初始化父类Optimizer

    def step(self, closure: Callable = None):
        """
        执行单步优化。
        参数:
        - closure: 可调用的闭包，用于重新计算模型并返回损失值。
        """
        loss = None
        if closure is not None:
            loss = closure()  # 重新计算损失
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue  # 如果参数没有梯度，则跳过
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, consider SparseAdam instead")
                state = self.state[p]
                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                # 更新梯度的一阶和二阶矩估计
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr']
                if group['correct_bias']:  # 校正偏差
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                # 在优化的最后应用权重衰减（正确的方式）
                if group['weight_decay'] > 0.0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
        return loss  # 返回损失值
