import logging  # 导入Python的日志管理库
import math     # 导入数学库，用于执行数学运算
import tensorboardX  # 导入tensorboardX，用于记录训练过程中的数据，方便在TensorBoard上进行可视化

_LOG_FMT = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s'  # 定义日志的格式，包括时间、日志级别、记录器名称和消息
_DATE_FMT = '%m/%d/%Y %H:%M:%S'  # 定义时间的显示格式
# 设置基础的日志配置，包括日志格式、时间格式和日志级别
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
LOGGER = logging.getLogger('__main__')  # 获取名为'__main__'的日志记录器，用于全局日志记录

def add_log_to_file(log_path):
    fh = logging.FileHandler(log_path)  # 创建一个文件日志处理器，用于写日志到文件
    formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)  # 创建一个日志格式器
    fh.setFormatter(formatter)  # 设置文件处理器的日志格式
    LOGGER.addHandler(fh)  # 将文件处理器添加到全局日志记录器

class TensorboardLogger(object):
    def __init__(self):
        self._logger = None  # 初始化时，设置内部的logger属性为None
        self._global_step = 0  # 初始化全局步骤计数器为0

    def create(self, path):
        self._logger = tensorboardX.SummaryWriter(path)  # 创建一个SummaryWriter实例，用于写入日志到指定路径

    def noop(self, *args, **kwargs):
        return  # 空操作函数，用于在logger未初始化时的调用

    def step(self):
        self._global_step += 1  # 增加步骤计数器

    @property
    def global_step(self):
        return self._global_step  # 获取当前的步骤计数

    def log_scalar_dict(self, log_dict, prefix=''):
        """记录标量数据的函数"""
        if self._logger is None:
            return  # 如果logger未初始化，则直接返回
        if prefix:
            prefix = f'{prefix}_'  # 设置数据的前缀
        for name, value in log_dict.items():
            if isinstance(value, dict):
                self.log_scalar_dict(value, self._global_step, prefix=f'{prefix}{name}')
            else:
                self._logger.add_scalar(f'{prefix}{name}', value, self._global_step)  # 将数据记录到TensorBoard

    def __getattr__(self, name):
        if self._logger is None:
            return self.noop  # 如果logger未初始化，返回noop函数
        return self._logger.__getattribute__(name)  # 动态获取logger的属性或方法

TB_LOGGER = TensorboardLogger()  # 创建一个TensorboardLogger实例

class RunningMeter(object):
    """用于监控标量值的类，例如训练损失"""
    def __init__(self, name, val=None, smooth=0.99):
        self._name = name  # 记录指标的名称
        self._sm = smooth  # 平滑系数
        self._val = val  # 当前值，如果未设置，则为None

    def __call__(self, value):
        val = (value if self._val is None else value * (1 - self._sm) + self._val * self._sm)  # 计算平滑后的值
        if not math.isnan(val):
            self._val = val  # 更新当前值，除非计算结果为NaN

    def __str__(self):
        return f'{self._name}: {self._val:.4f}'  # 格式化输出当前指标的值

    @property
    def val(self):
        if self._val is None:
            return 0  # 如果当前值未设置，返回0
        return self._val  # 返回当前值

    @property
    def name(self):
        return self._name  # 返回指标的名称

class AverageMeter(object):
    """用于计算和存储平均值和当前值的类"""
    def __init__(self):
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 累计总和
        self.count = 0  # 计数
        self.reset()  # 重置所有统计数据

    def reset(self):
        self.val = 0  # 重置当前值
        self.avg = 0  # 重置平均值
        self.sum = 0  # 重置累计总和
        self.count = 0  # 重置计数

    def update(self, val, n=1):
        self.val = val  # 更新当前值
        self.sum += val * n  # 更新累计总和
        self.count += n  # 更新计数
        self.avg = self.sum / self.count  # 计算新的平均值
