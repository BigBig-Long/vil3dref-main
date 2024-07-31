import json  # 导入json模块，用于处理JSON数据
import os  # 导入os模块，用于处理文件和目录
import torch  # 导入torch模块，用于深度学习操作

def save_training_meta(args):
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)  # 创建日志文件夹，如果已存在则忽略
    os.makedirs(os.path.join(args.output_dir, 'ckpts'), exist_ok=True)  # 创建模型检查点文件夹，如果已存在则忽略
    with open(os.path.join(args.output_dir, 'logs', 'config.json'), 'w') as writer:  # 打开配置文件进行写操作
        json.dump(vars(args), writer, indent=4)  # 将args对象转化为字典并保存为JSON格式

class ModelSaver(object):
    def __init__(self, output_dir, prefix='model_epoch', suffix='pt'):
        self.output_dir = output_dir  # 设置模型保存的输出目录
        self.prefix = prefix  # 设置保存文件名的前缀
        self.suffix = suffix  # 设置保存文件的后缀

    def save(self, model, epoch, optimizer=None, save_latest_optim=False):
        output_model_file = os.path.join(self.output_dir, f"{self.prefix}_{epoch}.{self.suffix}")  # 生成模型文件名
        state_dict = {}
        for k, v in model.state_dict().items():  # 遍历模型的状态字典
            if k.startswith('module.'):  # 检查键是否以'module.'开头
                k = k[7:]  # 去掉'module.'前缀
            if isinstance(v, torch.Tensor):  # 检查值是否为Tensor
                state_dict[k] = v.cpu()  # 将Tensor移至CPU
            else:
                state_dict[k] = v  # 直接赋值
        torch.save(state_dict, output_model_file)  # 保存处理后的状态字典到文件
        if optimizer is not None:  # 检查是否提供了优化器
            dump = {'epoch': epoch, 'optimizer': optimizer.state_dict()}  # 创建包含epoch和优化器状态的字典
            if hasattr(optimizer, '_amp_stash'):
                pass  # TODO: 如果优化器支持自动混合精度，此处需要特殊处理
            if save_latest_optim:
                torch.save(dump, f'{self.output_dir}/train_state_lastest.pt')  # 保存最新的优化器状态
            else:
                torch.save(dump, f'{self.output_dir}/train_state_{epoch}.pt')  # 保存指定epoch的优化器状态

    def remove_previous_models(self, cur_epoch):
        for saved_model_name in os.listdir(self.output_dir):  # 遍历输出目录的所有文件
            if saved_model_name.startswith(self.prefix):  # 如果文件名以指定前缀开始
                saved_model_epoch = int(os.path.splitext(saved_model_name)[0].split('_')[-1])  # 提取文件名中的epoch数字
                if saved_model_epoch != cur_epoch:  # 如果epoch不是当前epoch
                    os.remove(os.path.join(self.output_dir, saved_model_name))  # 删除该文件
