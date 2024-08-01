import os
import sys
import json
import numpy as np
import time
from collections import defaultdict, Counter
from tqdm import tqdm
from easydict import EasyDict
import pprint
import jsonlines
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from utils.logger import LOGGER, TB_LOGGER, AverageMeter, RunningMeter, add_log_to_file
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, set_random_seed, set_cuda, wrap_model
from utils.distributed import all_gather

from optim import get_lr_sched
from optim.misc import build_optimizer

from parser import load_parser, parse_with_config

from model.obj_encoder import PcdObjEncoder
from model.referit3d_net import get_mlp_head

import torch.nn as nn  # 导入PyTorch的神经网络模块

class PcdClassifier(nn.Module):  # 定义一个名为PcdClassifier的类，继承自nn.Module
    def __init__(self, config):  # 构造函数，接收一个配置对象config
        super().__init__()  # 调用父类的构造函数
        self.obj_encoder = PcdObjEncoder(config.obj_encoder)  # 初始化点云对象编码器
        # 初始化用于分类的多层感知机（MLP）头
        # get_mlp_head是一个外部函数，用于创建MLP头，其参数包括：
        # 输入层大小、隐藏层大小、输出类别数量和dropout比率
        self.obj3d_clf_pre_head = get_mlp_head(
            config.hidden_size, config.hidden_size,  # 输入和隐藏层的大小
            config.num_obj_classes,  # 输出的类别数量
            dropout=config.dropout  # dropout比率
        )

    def forward(self, obj_pcds):  # 定义前向传播方法，接收点云数据obj_pcds
        # 使用点云对象编码器处理点云数据，得到嵌入向量
        obj_embeds = self.obj_encoder.pcd_net(obj_pcds)
        # 对嵌入向量应用dropout，防止过拟合
        obj_embeds = self.obj_encoder.dropout(obj_embeds)
        # 将嵌入向量通过MLP头得到分类的logits
        logits = self.obj3d_clf_pre_head(obj_embeds)
        return logits  # 返回分类的logits



class PcdDataset(Dataset):
    def __init__(
        self, scan_id_file, scan_dir, category_file, num_points=1024,
        cat2vec_file=None, keep_background=False, random_rotate=False,
        og3d_subset_file=None, with_rgb=True,
    ):
        # 从指定的文件中读取scan_id，并去除每行末尾的空白字符
        scan_ids = [x.strip() for x in open(scan_id_file, 'r')]
        # 如果指定了og3d_subset_file，则只保留该文件中包含的scan_id
        if og3d_subset_file is not None:
            og3d_scanids = set()
            with jsonlines.open(og3d_subset_file, 'r') as f:
                for item in f:
                    og3d_scanids.add(item['scan_id'])
            scan_ids = [scan_id for scan_id in scan_ids if scan_id in og3d_scanids]
        # 类的属性，存储scan_id列表
        self.scan_ids = scan_ids
        # 存储点云文件的目录
        self.scan_dir = scan_dir
        # 是否保留背景对象（如墙、地板、天花板）
        self.keep_background = keep_background
        # 是否对点云进行随机旋转
        self.random_rotate = random_rotate
        # 每个对象采样的点云数量
        self.num_points = num_points
        # 是否包含RGB信息
        self.with_rgb = with_rgb
        # 从文件加载类别到整数的映射
        self.int2cat = json.load(open(category_file, 'r'))
        # 创建从类别到整数的映射
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        # 初始化数据列表
        self.data = []
        # 遍历所有的scan_id
        for scan_id in self.scan_ids:
            # 加载点云数据、颜色、标签和其他信息
            pcds, colors, _, instance_labels = torch.load(
                os.path.join(self.scan_dir, 'pcd_with_global_alignment', '%s.pth'%scan_id)
            )
            # 加载实例ID到名称的映射
            obj_labels = json.load(open(
                os.path.join(self.scan_dir, 'instance_id_to_name', '%s.json'%scan_id)
            ))
            # 遍历每个对象的标签
            for i, obj_label in enumerate(obj_labels):
                # 如果不保留背景对象，则跳过这些对象
                if (not self.keep_background) and obj_label in ['wall', 'floor', 'ceiling']:
                    continue
                # 创建一个掩码，只选择当前对象的点云
                mask = instance_labels == i
                # 确保至少有一个点属于当前对象
                assert np.sum(mask) > 0, 'scan: %s, obj %d' %(scan_id, i)
                # 获取当前对象的点云和颜色
                obj_pcd = pcds[mask]
                obj_color = colors[mask]
                # 归一化点云数据
                obj_pcd = obj_pcd - obj_pcd.mean(0)
                # 计算最大距离
                max_dist = np.max(np.sqrt(np.sum(obj_pcd**2, 1)))
                # 处理非常小的点云，防止除以0
                if max_dist < 1e-6:
                    max_dist = 1
                # 归一化点云
                obj_pcd = obj_pcd / max_dist
                # 归一化颜色
                obj_color = obj_color / 127.5 - 1
                # 如果包含RGB信息，将颜色和点云数据拼接在一起，并添加到数据列表
                if self.with_rgb:
                    self.data.append((np.concatenate([obj_pcd, obj_color], 1), self.cat2int[obj_label]))
                else:
                    # 否则只添加点云数据
                    self.data.append((obj_pcd, self.cat2int[obj_label]))

    def __len__(self):
        # 返回数据集的长度，即数据列表的长度
        return len(self.data)

    def _get_augmented_pcd(self, full_obj_pcds, theta=None):
        # 从完整的点云中随机选择点，如果点云数量少于指定的采样点数，则允许重复选择
        pcd_idxs = np.random.choice(len(full_obj_pcds), size=self.num_points,
                                    replace=(len(full_obj_pcds) < self.num_points))
        # 根据索引获取选择的点云
        obj_pcds = full_obj_pcds[pcd_idxs]
        # 如果指定了旋转角度且不为0，则对点云进行旋转
        if (theta is not None) and (theta != 0):
            # 创建旋转矩阵，绕z轴旋转
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ], dtype=np.float32)
            # 应用旋转矩阵
            obj_pcds[:, :3] = np.matmul(obj_pcds[:, :3], rot_matrix.transpose())
        # 返回旋转后的点云
        return obj_pcds

    def __getitem__(self, idx):
        # 根据索引获取点云和标签
        full_obj_pcds, obj_label = self.data[idx]
        # 初始化旋转角度
        if self.random_rotate:
            # 如果启用随机旋转，从预定义的角度中选择一个
            theta = np.random.choice([0, np.pi / 2, np.pi, np.pi * 3 / 2])
        else:
            theta = 0
        # 获取增强后的点云
        obj_pcds = self._get_augmented_pcd(full_obj_pcds, theta=theta)
        # 创建输出字典，包含点云和标签
        outs = {
            'obj_pcds': torch.from_numpy(obj_pcds),  # 将numpy数组转换为PyTorch张量
            'obj_labels': obj_label,  # 对象的标签
        }
        # 返回输出字典
        return outs
def pcd_collate_fn(data):
    # 初始化输出字典
    outs = {}
    # 遍历数据样本中的所有键（例如 'obj_pcds', 'obj_labels'）
    for key in data[0].keys():
        # 对于每个键，将所有样本中的对应值收集到一个列表中
        outs[key] = [x[key] for x in data]
    # 对于点云数据，将列表中的所有张量堆叠成一个批次张量，维度为 (batch_size, num_points, feature_size)
    outs['obj_pcds'] = torch.stack(outs['obj_pcds'], 0)
    # 对于标签数据，将列表转换为长整数张量（LongTensor），维度为 (batch_size)
    outs['obj_labels'] = torch.LongTensor(outs['obj_labels'])
    # 返回处理后的批次数据
    return outs


def main(opts):
    # 设置CUDA设备，包括默认设备、GPU数量和设备对象
    default_gpu, n_gpu, device = set_cuda(opts)

    # 如果有默认GPU，打印设备信息
    if default_gpu:
        LOGGER.info(
            'device: {} n_gpu: {}, distributed training: {}'.format(
                device, n_gpu, bool(opts.local_rank != -1)
            )
        )

    # 设置随机种子
    seed = opts.seed
    # 如果是分布式训练，随机种子加上当前进程的rank
    if opts.local_rank != -1:
        seed += opts.rank
    set_random_seed(seed)

    # 如果有默认GPU且不是测试模式
    if default_gpu and not opts.test:
        # 保存训练的元数据信息
        save_training_meta(opts)
        # 创建TensorBoard日志记录器
        TB_LOGGER.create(os.path.join(opts.output_dir, 'logs'))
        # 创建模型保存器，用于保存训练过程中的检查点
        model_saver = ModelSaver(os.path.join(opts.output_dir, 'ckpts'))
        # 将日志添加到文件中
        add_log_to_file(os.path.join(opts.output_dir, 'logs', 'log.txt'))
    else:
        # 如果没有默认GPU或处于测试模式，禁用日志记录
        LOGGER.disabled = True
        # 使用无操作的进度条
        pbar = NoOp()
        # 使用无操作的模型保存器
        model_saver = NoOp()
    # 准备模型
    model_config = EasyDict(opts.model)  # 将模型配置转换为EasyDict对象，方便访问配置项
    model = PcdClassifier(model_config)  # 实例化模型
    model = wrap_model(model, device, opts.local_rank)  # 包装模型以支持分布式训练
    num_weights, num_trainable_weights = 0, 0  # 初始化权重数量和可训练权重数量
    for p in model.parameters():  # 遍历模型的所有参数
        psize = np.prod(p.size())  # 计算参数的元素数量
        num_weights += psize  # 累加总权重数量
        if p.requires_grad:  # 如果参数需要梯度更新
            num_trainable_weights += psize  # 累加可训练权重数量
    LOGGER.info('#weights: %d, #trainable weights: %d', num_weights, num_trainable_weights)  # 打印权重数量信息

    if opts.resume_files:  # 如果指定了恢复训练的文件
        checkpoint = {}  # 初始化检查点字典
        for resume_file in opts.resume_files:  # 遍历恢复文件列表
            checkpoint.update(torch.load(resume_file, map_location=lambda storage, loc: storage))  # 加载检查点
        print('resume #params:', len(checkpoint))  # 打印恢复的参数数量
        model.load_state_dict(checkpoint, strict=False)  # 加载检查点中的模型状态

    # 加载数据训练集
    data_cfg = EasyDict(opts.dataset)  # 将数据集配置转换为EasyDict对象
    trn_dataset = PcdDataset(  # 创建训练数据集
        data_cfg.trn_scan_split, data_cfg.scan_dir, data_cfg.category_file,
        num_points=data_cfg.num_points,  # 指定点云中的点数
        random_rotate=data_cfg.random_rotate if not opts.test else False,  # 是否随机旋转
        keep_background=data_cfg.keep_background,  # 是否保留背景
        og3d_subset_file=data_cfg.og3d_subset_file,  # 指定可能的3D对象子集文件
        with_rgb=data_cfg.with_rgb,  # 是否包含RGB信息
    )
    val_dataset = PcdDataset(  # 创建验证数据集
        data_cfg.val_scan_split, data_cfg.scan_dir, data_cfg.category_file,
        random_rotate=False,  # 验证集不进行随机旋转
        num_points=data_cfg.num_points,  # 指定点云中的点数
        keep_background=data_cfg.keep_background,  # 是否保留背景
        og3d_subset_file=data_cfg.og3d_subset_file,  # 指定可能的3D对象子集文件
        with_rgb=data_cfg.with_rgb,  # 是否包含RGB信息
    )
    LOGGER.info('train #scans %d, #data %d' % (len(trn_dataset.scan_ids), len(trn_dataset)))  # 打印训练集扫描次数和数据数量
    LOGGER.info('val #scans %d, #data %d' % (len(val_dataset.scan_ids), len(val_dataset)))  # 打印验证集扫描次数和数据数量
    # 构建数据加载器
    trn_dataloader = DataLoader(
        trn_dataset, batch_size=opts.batch_size, shuffle=True,  # 训练数据加载器，打乱数据顺序
        num_workers=opts.num_workers, collate_fn=pcd_collate_fn,  # 指定工作进程数和自定义的批次处理函数
        pin_memory=True, drop_last=False,  # 将数据固定到内存中，不丢弃最后不完整批次
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=opts.batch_size, shuffle=False,  # 验证数据加载器，不打乱数据顺序
        num_workers=opts.num_workers, collate_fn=pcd_collate_fn,  # 指定工作进程数和自定义的批次处理函数
        pin_memory=True, drop_last=False,  # 将数据固定到内存中，不丢弃最后不完整批次
    )
    opts.num_train_steps = len(trn_dataloader) * opts.num_epoch  # 计算总的训练步数

    # 如果是测试模式，直接进行验证
    if opts.test:
        val_log = validate(model, trn_dataloader, device)  # 在训练集上验证模型
        val_log = validate(model, val_dataloader, device)  # 在验证集上验证模型
        return  # 测试完成后返回

    # 准备优化器
    optimizer, _ = build_optimizer(model, opts)  # 构建优化器
    LOGGER.info(f"***** Running training with {opts.world_size} GPUs *****")  # 打印使用GPU数量的信息
    LOGGER.info("  Batch size = %d",
                opts.batch_size if opts.local_rank == -1 else opts.batch_size * opts.world_size)  # 打印批处理大小
    LOGGER.info("  Num epoch = %d, num steps = %d", opts.num_epoch, opts.num_train_steps)  # 打印训练轮数和步数

    # 计算训练统计信息
    avg_metrics = defaultdict(AverageMeter)  # 初始化平均值计算器
    global_step = 0  # 初始化全局步数
    model.train()  # 将模型设置为训练模式
    optimizer.zero_grad()  # 清空之前的梯度
    optimizer.step()  # 执行优化器步骤（通常在计算损失和反向传播之后调用）
    # 初始化最佳验证分数
    val_best_scores = {'epoch': -1, 'acc': -float('inf')}

    # 开始训练循环
    for epoch in tqdm(range(opts.num_epoch), desc='Epoch'):
        start_time = time.time()  # 记录当前epoch的开始时间
        # 训练集的batch处理循环
        for batch in tqdm(trn_dataloader, desc='Batch', leave=False):
            # 将数据移动到对应的设备上
            for bk, bv in batch.items():
                batch[bk] = bv.to(device)
            batch_size = len(batch['obj_pcds'])  # 获取当前批次的大小
            logits = model(batch['obj_pcds'])  # 前向传播计算logits
            loss = F.cross_entropy(logits, batch['obj_labels'])  # 计算交叉熵损失
            loss.backward()  # 反向传播计算梯度

            # 优化器更新和日志记录
            global_step += 1  # 增加全局步数
            # 学习率调度：此处为文本编码器设置学习率，可能需要根据实际情况调整
            lr_this_step = get_lr_sched(global_step, opts)
            for kp, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = lr_this_step  # 更新当前步的学习率
            TB_LOGGER.add_scalar('lr', lr_this_step, global_step)  # 记录学习率

            # 记录损失
            # 注意：为了效率，这里没有在多个GPU之间收集损失
            avg_metrics['loss'].update(loss.data.item(), n=batch_size)  # 更新损失平均值
            TB_LOGGER.log_scalar_dict({'loss': loss.data.item()})  # 记录损失
            TB_LOGGER.step()  # 步进TB_LOGGER以记录当前步的数据

            # 更新模型参数
            if opts.grad_norm != -1:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), opts.grad_norm  # 如果设置了梯度范数限制，则进行梯度裁剪
                )
                TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)  # 记录梯度范数

            optimizer.step()  # 更新模型参数
            optimizer.zero_grad()  # 清空梯度，准备下一个批次
        # 打印当前epoch的训练统计信息
        LOGGER.info(
            'Epoch %d, lr: %.6f, %s', epoch + 1,  # 打印epoch编号和当前学习率
            optimizer.param_groups[0]['lr'],  # 从优化器的第一个参数组中获取学习率
            ', '.join(['%s: %.4f' % (lk, lv.avg) for lk, lv in avg_metrics.items()])  # 打印所有平均指标
        )

        # 按照设定的间隔执行验证过程
        if (epoch + 1) % opts.val_every_epoch == 0:
            LOGGER.info(f'------Epoch {epoch + 1}: start validation------')  # 打印验证开始信息
            val_log = validate(model, val_dataloader, device)  # 执行验证过程
            TB_LOGGER.log_scalar_dict(  # 记录验证指标到TensorBoard
                {f'valid/{k}': v.avg for k, v in val_log.items()}
            )
            # model_saver.save(model, epoch+1, optimizer=optimizer)  # 注释掉的保存模型代码

        # 如果当前验证准确率高于之前最佳的准确率，则保存模型
        if val_log['acc'].avg > val_best_scores['acc']:
            output_model_file = model_saver.save(model, epoch + 1)  # 保存当前模型
            val_best_scores['acc'] = val_log['acc'].avg  # 更新最佳准确率
            val_best_scores['uw_acc'] = val_log['uw_acc'].avg  # 更新最佳未加权准确率
            val_best_scores['epoch'] = epoch + 1  # 更新最佳准确率对应的epoch

            # 删除非最佳检查点
            for ckpt_file in os.listdir(model_saver.output_dir):
                ckpt_file = os.path.join(model_saver.output_dir, ckpt_file)
                if ckpt_file != output_model_file:
                    os.remove(ckpt_file)  # 删除旧的检查点

        # 训练完成后打印最终信息
        LOGGER.info('Finished training!')
        LOGGER.info(
            'best epoch: %d, best acc %.4f, uw_acc: %.4f',
            val_best_scores['epoch'], val_best_scores['acc'], val_best_scores['uw_acc']
        )  # 打印最佳验证结果


@torch.no_grad()  # 修饰器，确保在下面的函数中不会计算梯度
def validate(model, val_dataloader, device):
    model.eval()  # 将模型设置为评估模式，这会关闭dropout和batch normalization的随机性
    avg_metrics = defaultdict(AverageMeter)  # 创建一个默认值为AverageMeter的字典，用于记录平均指标
    uw_acc_metrics = defaultdict(AverageMeter)  # 创建一个默认值为AverageMeter的字典，用于记录每个类别的未加权准确率

    # 遍历验证集的每个batch
    for batch in val_dataloader:
        batch_size = len(batch['obj_pcds'])  # 获取当前批次的大小
        # 将数据移动到对应的设备上
        for bk, bv in batch.items():
            batch[bk] = bv.to(device)

        logits = model(batch['obj_pcds'])  # 前向传播计算logits
        loss = F.cross_entropy(logits, batch['obj_labels']).data.item()  # 计算交叉熵损失
        preds = torch.argmax(logits, 1)  # 获取预测结果
        acc = torch.mean((preds == batch['obj_labels']).float()).item()  # 计算准确率

        # 更新平均损失和准确率
        avg_metrics['loss'].update(loss, n=batch_size)
        avg_metrics['acc'].update(acc, n=batch_size)

        # 更新每个类别的未加权准确率
        for pred, label in zip(preds.cpu().numpy(), batch['obj_labels'].cpu().numpy()):
            uw_acc_metrics[label].update(pred == label, n=1)

        # 计算并更新平均未加权准确率
        avg_metrics['uw_acc'].update(np.mean([x.avg for x in uw_acc_metrics.values()]), n=1)

    # 打印验证集上的平均指标
    LOGGER.info(', '.join(['%s: %.4f' % (lk, lv.avg) for lk, lv in avg_metrics.items()]))

    model.train()  # 将模型设置为训练模式
    return avg_metrics  # 返回包含验证集上平均指标的字典型对象


def build_args():  # 定义build_args函数，用于构建命令行参数
    parser = load_parser()  # 调用load_parser函数加载预定义的参数解析器
    opts = parse_with_config(parser)  # 调用parse_with_config函数解析命令行参数，并且可能和配置文件合并
    # 检查指定的输出目录是否存在且不为空
    if os.path.exists(opts.output_dir) and os.listdir(opts.output_dir):
        LOGGER.warning(  # 如果存在，使用LOGGER警告用户
            "Output directory ({}) already exists and is not empty.".format(
                opts.output_dir  # 打印输出目录的路径
            )
        )
    return opts  # 返回解析后的参数

if __name__ == '__main__':  # 如果此模块是作为主程序运行
    args = build_args()  # 构建命令行参数
    pprint.pprint(args)  # 美化打印参数，以便用户可以看到
    main(args)  # 使用解析后的参数调用主函数main

# 注意：LOGGER和main函数需要在其他地方定义，这里假设它们已经定义好了。
