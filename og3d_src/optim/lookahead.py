import torch  # 导入torch库，用于深度学习操作
from torch.optim.optimizer import Optimizer  # 从torch.optim模块导入Optimizer类，作为自定义优化器的基类
from torch.optim import Adam  # 导入Adam优化器
from collections import defaultdict  # 导入defaultdict，用于创建带有默认值的字典

class Lookahead(Optimizer):  # 定义Lookahead类，继承自Optimizer
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:  # 确保alpha参数在[0, 1]范围内
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:  # 确保k参数至少为1
            raise ValueError(f'Invalid lookahead steps: {k}')
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)  # 设置Lookahead的默认参数
        self.base_optimizer = base_optimizer  # 存储基础优化器（例如Adam）
        self.param_groups = self.base_optimizer.param_groups  # 引用基础优化器的参数组
        self.defaults = base_optimizer.defaults  # 从基础优化器获取默认参数
        self.defaults.update(defaults)  # 更新默认参数，添加Lookahead特有的参数
        self.state = defaultdict(dict)  # 创建状态字典，用于存储优化器状态
        # 为参数组添加Lookahead的默认参数
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_p in group["params"]:  # 遍历参数组中的参数
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if 'slow_buffer' not in param_state:  # 如果没有慢速缓冲区，则初始化
                param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                param_state['slow_buffer'].copy_(fast_p.data)
            slow = param_state['slow_buffer']
            slow.add_(group['lookahead_alpha'], fast_p.data - slow)  # 更新慢速权重
            fast_p.data.copy_(slow)  # 将更新后的慢速权重复制到参数中

    def sync_lookahead(self):
        for group in self.param_groups:  # 同步所有参数组
            self.update_slow(group)

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)  # 使用基础优化器执行优化步骤
        for group in self.param_groups:  # 更新Lookahead步骤，可能更新慢速权重
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()  # 获取基础优化器的状态字典
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            'state': state_dict['state'],
            'param_groups': state_dict['param_groups'],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)  # 将状态字典加载到基础优化器中
        # 分别处理慢速状态
        slow_state_new = False
        if 'slow_state' not in state_dict:
            print('Loading state_dict from optimizer without Lookahead applied.')
            state_dict['slow_state'] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = self.base_optimizer.param_groups  # 确保参数组引用同步
        if slow_state_new:
            # 如果慢速状态是新创建的，则重新应用默认值
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)

def LookaheadAdam(params, alpha=0.5, k=6, *args, **kwargs):
    adam = Adam(params, *args, **kwargs)  # 创建一个Adam优化器
    return Lookahead(adam, alpha, k)  # 使用Lookahead包装Adam
