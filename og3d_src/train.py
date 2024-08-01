import os
import sys
import json
import numpy as np
import time
from collections import defaultdict
from tqdm import tqdm
from easydict import EasyDict
import pprint

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.logger import LOGGER, TB_LOGGER, AverageMeter, RunningMeter, add_log_to_file
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, set_random_seed, set_cuda, wrap_model
from utils.distributed import all_gather

from optim import get_lr_sched_decay_rate
from optim.misc import build_optimizer

from parser import load_parser, parse_with_config

from data.gtlabel_dataset import GTLabelDataset, gtlabel_collate_fn
from data.gtpcd_dataset import GTPcdDataset, gtpcd_collate_fn

from model.referit3d_net import ReferIt3DNet



def build_gtlabel_datasets(data_cfg):
    # 构建训练数据集
    trn_dataset = GTLabelDataset(
        data_cfg.trn_scan_split,  # 训练数据扫描分割路径
        data_cfg.anno_file,       # 标注文件路径
        data_cfg.scan_dir,        # 扫描数据所在目录
        data_cfg.category_file,   # 类别文件路径
        cat2vec_file=data_cfg.cat2vec_file,  # 类别向量文件路径
        max_txt_len=data_cfg.max_txt_len,    # 最大文本长度限制
        max_obj_len=data_cfg.max_obj_len,    # 最大对象长度限制
        keep_background=data_cfg.keep_background,  # 是否保留背景对象
        random_rotate=data_cfg.random_rotate,       # 是否应用随机旋转
        gt_scan_dir=data_cfg.get('gt_scan_dir', None),  # 地面真实扫描数据目录，可选
        iou_replace_gt=data_cfg.get('iou_replace_gt', 0),  # 替换地面真实数据的IoU阈值
    )

    # 构建验证数据集
    val_dataset = GTLabelDataset(
        data_cfg.val_scan_split,  # 验证数据扫描分割路径
        data_cfg.anno_file,       # 标注文件路径
        data_cfg.scan_dir,        # 扫描数据所在目录
        data_cfg.category_file,   # 类别文件路径
        cat2vec_file=data_cfg.cat2vec_file,  # 类别向量文件路径
        max_txt_len=None,         # 验证集不设定最大文本长度
        max_obj_len=None,         # 验证集不设定最大对象长度
        keep_background=data_cfg.keep_background,  # 是否保留背景对象
        random_rotate=False,      # 验证集不应用随机旋转
        gt_scan_dir=data_cfg.get('gt_scan_dir', None),  # 地面真实扫描数据目录，可选
        iou_replace_gt=data_cfg.get('iou_replace_gt', 0),  # 替换地面真实数据的IoU阈值
    )

    # 返回构建的训练集和验证集
    return trn_dataset, val_dataset


def build_gtpcd_datasets(data_cfg):
    # 构建训练数据集
    trn_dataset = GTPcdDataset(
        data_cfg.trn_scan_split,  # 训练数据集扫描分割文件路径
        data_cfg.anno_file,       # 注解文件路径
        data_cfg.scan_dir,        # 扫描数据所在的目录
        data_cfg.category_file,   # 类别文件路径
        cat2vec_file=data_cfg.cat2vec_file,  # 类别向量文件路径
        random_rotate=data_cfg.random_rotate,  # 是否应用随机旋转
        max_txt_len=data_cfg.max_txt_len,    # 最大文本长度限制
        max_obj_len=data_cfg.max_obj_len,    # 最大对象长度限制
        keep_background=data_cfg.keep_background,  # 是否保留背景对象
        num_points=data_cfg.num_points,      # 从点云中采样的点数
        in_memory=True,                      # 是否将数据集完整地加载到内存中
    )

    # 构建验证数据集
    val_dataset = GTPcdDataset(
        data_cfg.val_scan_split,  # 验证数据集扫描分割文件路径
        data_cfg.anno_file,       # 注解文件路径
        data_cfg.scan_dir,        # 扫描数据所在的目录
        data_cfg.category_file,   # 类别文件路径
        cat2vec_file=data_cfg.cat2vec_file,  # 类别向量文件路径
        max_txt_len=None,         # 验证集不设置最大文本长度
        max_obj_len=None,         # 验证集不设置最大对象长度
        random_rotate=False,      # 验证集不应用随机旋转
        keep_background=data_cfg.keep_background,  # 是否保留背景对象
        num_points=data_cfg.num_points,      # 从点云中采样的点数
        in_memory=True,                      # 是否将数据集完整地加载到内存中
    )

    # 返回构建的训练集和验证集
    return trn_dataset, val_dataset



def main(opts):
    # 设置CUDA设备，返回默认GPU设备，可用的GPU数量和设备信息
    default_gpu, n_gpu, device = set_cuda(opts)
    # 如果有默认的GPU设备可用
    if default_gpu:
        # 记录设备信息，GPU数量和是否为分布式训练
        LOGGER.info(
            'device: {} n_gpu: {}, distributed training: {}'.format(
                device, n_gpu, bool(opts.local_rank != -1)
            )
        )
    # 设置随机种子，以确保训练可复现
    seed = opts.seed
    # 如果在分布式训练中，根据rank调整seed，保证每个进程的随机种子不同
    if opts.local_rank != -1:
        seed += opts.rank
    # 应用设置的随机种子
    set_random_seed(seed)
    # 如果是默认GPU并且不是在测试模式下
    if default_gpu:
        if not opts.test:
            # 保存训练的元数据
            save_training_meta(opts)
            # 创建TensorBoard日志记录器
            TB_LOGGER.create(os.path.join(opts.output_dir, 'logs'))
            # 创建模型保存器，用于定期保存训练的模型
            model_saver = ModelSaver(os.path.join(opts.output_dir, 'ckpts'))
            # 添加日志文件
            add_log_to_file(os.path.join(opts.output_dir, 'logs', 'log.txt'))
    # 如果是测试模式，则禁用日志记录
    else:
        LOGGER.disabled = True
        # 使用一个无操作的进度条，不显示训练进度
        pbar = NoOp()
        # 使用一个无操作的模型保存器，不保存模型
        model_saver = NoOp()

    # 检查是否是默认GPU:
    #
    # if default_gpu: 这行代码检查当前的进程是否在默认（主）GPU上运行。在多GPU设置中，通常只在一个GPU上执行某些操作，如日志记录或元数据保存，以避免重复操作。
    # 非测试模式:
    #
    # if not opts.test: 这个条件检查是否处于非测试模式。测试模式可能只是为了验证或演示，不需要执行所有常规训练操作。
    # save_training_meta(opts): 这个函数用于保存训练相关的元数据，比如训练开始时间、配置参数等，有助于后续分析和复现实验。
    # TB_LOGGER.create(os.path.join(opts.output_dir, 'logs')): 初始化TensorBoard日志记录器，在指定的目录下创建日志文件。这对于监控训练进度和性能非常重要。
    # model_saver = ModelSaver(os.path.join(opts.output_dir, 'ckpts')): 初始化模型保存器，设置保存模型检查点的目录。这确保了训练过程中的模型状态可以被保存和恢复。
    # add_log_to_file(os.path.join(opts.output_dir, 'logs', 'log.txt')): 设置日志文件，所有运行时的信息和错误都会被记录在这个文件中，方便调试和审查。
    # 测试模式:
    #
    # else: 如果是测试模式，执行以下操作：
    # LOGGER.disabled = True: 禁用LOGGER的日志记录功能。在测试时可能不需要记录详细的日志，以减少资源消耗和输出的干扰。
    # pbar = NoOp(): 这通常代表一个无操作的进度条。NoOp()是一个不执行任何操作的占位符，用于在不需要实际功能的地方维持代码结构。
    # model_saver = NoOp(): 类似于pbar，这也是一个无操作的模型保存器，在测试模式下，可能不需要实际保存模型。

    # Prepare model
    # 使用模型配置创建一个EasyDict对象
    model_config = EasyDict(opts.model)
    # 初始化ReferIt3DNet模型，传入配置和设备信息
    model = ReferIt3DNet(model_config, device)
    # 初始化权重计数器
    num_weights, num_trainable_weights = 0, 0
    # 遍历模型的所有参数
    for p in model.parameters():
        # 计算每个参数的元素总数
        psize = np.prod(p.size())
        # 更新总权重数
        num_weights += psize
        # 如果参数可训练，更新可训练权重数
        if p.requires_grad:
            num_trainable_weights += psize
    # 记录总权重和可训练权重的数量
    LOGGER.info('#weights: %d, #trainable weights: %d', num_weights, num_trainable_weights)
    # 如果存在恢复文件，则处理文件以恢复模型状态
    if opts.resume_files:
        # 初始化检查点字典
        checkpoint = {}
        # 遍历所有恢复文件
        for resume_file in opts.resume_files:
            # 加载检查点文件，确保加载到正确的设备上
            new_checkpoints = torch.load(resume_file, map_location=lambda storage, loc: storage)
            # 将新加载的检查点合并到总检查点字典中
            for k, v in new_checkpoints.items():
                if k not in checkpoint:
                    checkpoint[k] = v
        # 打印恢复的参数数量和匹配到模型的参数数量
        print('resume #params:', len(checkpoint), len([n for n in checkpoint.keys() if n in model.state_dict()]))
        # 检查并打印任何未在模型状态字典中找到的参数名称和尺寸
        for n in checkpoint.keys():
            if n not in model.state_dict():
                print(n, checkpoint[n].size())
        # 将检查点加载到模型中，不严格匹配所有参数
        model.load_state_dict(checkpoint, strict=False)
    # 获取模型配置
    model_cfg = model.config
    # 根据设备和本地排名包装模型，以支持可能的并行处理
    model = wrap_model(model, device, opts.local_rank)
    # 使用数据集配置创建一个EasyDict对象
    data_cfg = EasyDict(opts.dataset)
    # 根据模型类型选择合适的数据集构建函数
    if model_config.model_type == 'gtlabel':
        # 构建标签地面真实数据集
        trn_dataset, val_dataset = build_gtlabel_datasets(data_cfg)
        # 设置用于数据加载的函数
        collate_fn = gtlabel_collate_fn
    elif model_config.model_type == 'gtpcd':
        # 构建点云地面真实数据集
        trn_dataset, val_dataset = build_gtpcd_datasets(data_cfg)
        # 设置用于数据加载的函数
        collate_fn = gtpcd_collate_fn
    # 记录训练和验证数据集的扫描数量和数据点数量
    LOGGER.info('train #scans %d, #data %d' % (len(trn_dataset.scan_ids), len(trn_dataset)))
    LOGGER.info('val #scans %d, #data %d' % (len(val_dataset.scan_ids), len(val_dataset)))
    # 如果不是分布式训练，不使用sampler
    if opts.local_rank == -1:
        trn_sampler = None
        # 设置每个epoch开始时的函数，这里为空操作
        pre_epoch = lambda e: None
        # 实际批处理大小等于配置的批处理大小
        real_batch_size = opts.batch_size
    else:
        # 分布式训练设置
        size = dist.get_world_size()
        # 创建分布式采样器，用于数据加载过程中的分布式处理
        trn_sampler = DistributedSampler(
            trn_dataset, num_replicas=size, rank=dist.get_rank(), shuffle=True
        )
        # 设置每个epoch开始时重置采样器
        pre_epoch = trn_sampler.set_epoch
        # 实际批处理大小为配置的批处理大小乘以分布式训练中的进程数。

    # 使用DataLoader类创建训练数据加载器
    trn_dataloader = DataLoader(
        trn_dataset, batch_size=opts.batch_size, shuffle=True if trn_sampler is None else False,
        num_workers=opts.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False, prefetch_factor=1,
        sampler=trn_sampler
    )
    # 创建验证数据加载器
    val_dataloader = DataLoader(
        val_dataset, batch_size=opts.batch_size, shuffle=False,
        num_workers=opts.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False, prefetch_factor=1,
    )

    # 根据训练数据加载器的长度和配置的epoch数计算总的训练步骤数
    opts.num_train_steps = len(trn_dataloader) * opts.num_epoch
    # 如果处于测试模式
    if opts.test:
        # 进行模型验证，返回验证日志和预测结果
        val_log, out_preds = validate(model, model_cfg, val_dataloader, return_preds=True)
        # 设置预测结果的保存目录
        pred_dir = os.path.join(opts.output_dir, 'preds')
        # 创建目录，如果已存在则不报错
        os.makedirs(pred_dir, exist_ok=True)
        # 将预测结果保存为JSON文件
        json.dump(out_preds, open(os.path.join(pred_dir, 'val_outs.json'), 'w'))
        # 结束函数执行
        return

    # 准备优化器，返回优化器实例和初始学习率列表
    optimizer, init_lrs = build_optimizer(model, opts)
    # 如果有优化器状态文件需恢复
    if opts.resume_optimizer is not None:
        # 加载优化器状态
        optimizer_state = torch.load(opts.resume_optimizer)
        # 输出加载的优化器状态信息
        print('load optimizer epoch: %d, weights: %d' % (
            optimizer_state['epoch'], len(optimizer_state['optimizer']))
              )
        # 恢复优化器状态
        optimizer.load_state_dict(optimizer_state['optimizer'])

    # 记录运行训练的信息，包括使用的GPU数量
    LOGGER.info(f"***** Running training with {opts.world_size} GPUs *****")
    # 记录实际的批次大小，如果是分布式训练则乘以进程数
    LOGGER.info("  Batch size = %d", opts.batch_size if opts.local_rank == -1 else opts.batch_size * opts.world_size)
    # 记录总的训练周期数和步骤数
    LOGGER.info("  Num epoch = %d, num steps = %d", opts.num_epoch, opts.num_train_steps)

    # 使用defaultdict创建平均度量字典，用于计算训练统计
    avg_metrics = defaultdict(AverageMeter)
    # 初始化全局步骤计数器
    global_step = 0
    # 模型设置为训练模式
    model.train()
    # 优化器梯度清零
    optimizer.zero_grad()
    # 执行一步优化
    optimizer.step()
    # 如果是默认GPU设备
    if default_gpu:
        # 执行验证并记录日志
        val_log = validate(model, model_cfg, val_dataloader)
        # 初始化记录最好验证成绩的字典
        val_best_scores = {'epoch': -1, 'acc/og3d': -float('inf')}
        # 创建一个范围为训练周期数的迭代器
        epoch_iter = range(opts.num_epoch)
        # 如果是默认GPU，使用tqdm库创建进度条
        if default_gpu:
            epoch_iter = tqdm(epoch_iter)
        # 对每个周期进行迭代
        for epoch in epoch_iter:
            # 在分布式训练中调用pre_epoch函数
            pre_epoch(epoch)
            # 记录周期开始时间
            start_time = time.time()
            # 创建训练数据的迭代器
            batch_iter = trn_dataloader
            if default_gpu:
                batch_iter = tqdm(batch_iter)
            # 对每个批次进行迭代
            for batch in batch_iter:
                # 获取批次大小
                batch_size = len(batch['scan_ids'])
                # 计算模型输出和损失
                result, losses = model(batch, compute_loss=True)
                # 反向传播计算梯度
                losses['total'].backward()
                # 增加全局步骤计数
                global_step += 1
                # 计算学习率衰减
                lr_decay_rate = get_lr_sched_decay_rate(global_step, opts)
                for kp, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = lr_this_step = init_lrs[kp] * lr_decay_rate
                    TB_LOGGER.add_scalar('lr', lr_this_step, global_step)
                # 记录损失日志
                loss_dict = {'loss/%s' % lk: lv.data.item() for lk, lv in losses.items()}
                for lk, lv in loss_dict.items():
                    avg_metrics[lk].update(lv, n=batch_size)
                TB_LOGGER.log_scalar_dict(loss_dict)
                TB_LOGGER.step()
                # 如果设置了梯度裁剪，执行裁剪
                if opts.grad_norm != -1:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), opts.grad_norm
                    )
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                # 更新模型参数
                optimizer.step()
                optimizer.zero_grad()
                # 记录周期的学习率和损失
                LOGGER.info(
                    'Epoch %d, lr: %.6f, %s', epoch + 1,
                    optimizer.param_groups[-1]['lr'],
                    ', '.join(['%s: %.4f' % (lk, lv.avg) for lk, lv in avg_metrics.items()])
                )
                # 如果达到验证周期，执行验证
                if default_gpu and (epoch + 1) % opts.val_every_epoch == 0:
                    LOGGER.info(f'------Epoch {epoch + 1}: start validation (val)------')
                    val_log = validate(model, model_cfg, val_dataloader)
                    TB_LOGGER.log_scalar_dict(
                        {f'valid/{k}': v.avg for k, v in val_log.items()}
                    )
                    # 保存模型
                    output_model_file = model_saver.save(
                        model, epoch + 1, optimizer=optimizer, save_latest_optim=True
                    )
                    # 如果当前周期的验证结果优于之前的最好成绩，更新最好成绩
                    if val_log['acc/og3d'].avg > val_best_scores['acc/og3d']:
                        val_best_scores['acc/og3d'] = val_log['acc/og3d'].avg
                        val_best_scores['epoch'] = epoch + 1
                        model_saver.remove_previous_models(epoch + 1)
                    else:
                        os.remove(output_model_file)
        # 训练结束
        LOGGER.info('Finished training!')
        LOGGER.info(
            'best epoch: %d, best acc/og3d %.4f', val_best_scores['epoch'], val_best_scores['acc/og3d']
        )


@torch.no_grad()  # 使用此装饰器确保在此函数中不进行梯度计算，适用于评估和预测阶段
def validate(model, model_cfg, val_dataloader, niters=None, return_preds=False):
    model.eval()  # 将模型设置为评估模式，这会禁用一些特定于训练的操作，如Dropout
    output_attentions = True  # 初始化变量以决定是否输出注意力权重
    output_attentions = False  # 立即将其设置为False，表示在此验证中不输出注意力权重
    avg_metrics = defaultdict(AverageMeter)  # 创建一个用于存储平均度量值的字典
    out_preds = {}  # 初始化一个字典，用于存储输出的预测结果

    # 遍历验证数据集的每个批次
    for ib, batch in enumerate(val_dataloader):
        batch_size = len(batch['scan_ids'])  # 获取当前批次的大小
        # 使用模型对当前批次进行预测，同时计算损失
        result, losses = model(
            batch, compute_loss=True, is_test=True,
            output_attentions=output_attentions,
            output_hidden_states=False,
        )
        # 将损失信息转换为字典形式，用于记录
        loss_dict = {'loss/%s' % lk: lv.data.item() for lk, lv in losses.items()}
        # 更新平均度量字典
        for lk, lv in loss_dict.items():
            avg_metrics[lk].update(lv, n=batch_size)

        # 对'og3d_logits'进行argmax操作，获取预测结果
        og3d_preds = torch.argmax(result['og3d_logits'], dim=1).cpu()
        # 更新'og3d'准确率的度量
        avg_metrics['acc/og3d'].update(
            torch.mean((og3d_preds == batch['tgt_obj_idxs']).float()).item(),
            n=batch_size
        )
        # 更新'og3d_class'分类准确率的度量
        avg_metrics['acc/og3d_class'].update(
            torch.mean((batch['obj_classes'].gather(1, og3d_preds.unsqueeze(1)).squeeze(1) == batch[
                'tgt_obj_classes']).float()).item(),
            n=batch_size
        )

        # 如果模型配置中指定了'obj3d_clf'的损失权重大于0
        if model_cfg.losses.obj3d_clf > 0:
            # 对'obj3d_clf_logits'进行argmax操作，获取分类预测结果
            obj3d_clf_preds = torch.argmax(result['obj3d_clf_logits'], dim=2).cpu()
            # 更新'obj3d_clf'分类准确率的度量
            avg_metrics['acc/obj3d_clf'].update(
                (obj3d_clf_preds[batch['obj_masks']] == batch['obj_classes'][batch['obj_masks']]).float().mean().item(),
                n=batch['obj_masks'].sum().item()
            )

        # 如果模型配置中指定了'obj3d_clf_pre'的损失权重大于0
        if model_cfg.losses.obj3d_clf_pre > 0:
            # 对'obj3d_clf_pre_logits'进行argmax操作，获取分类预测结果
            obj3d_clf_preds = torch.argmax(result['obj3d_clf_pre_logits'], dim=2).cpu()
            # 更新'obj3d_clf_pre'分类准确率的度量
            avg_metrics['acc/obj3d_clf_pre'].update(
                (obj3d_clf_preds[batch['obj_masks']] == batch['obj_classes'][batch['obj_masks']]).float().mean().item(),
                n=batch['obj_masks'].sum().item()
            )

        # 如果模型配置中指定了'txt_clf'的损失权重大于0
        if model_cfg.losses.txt_clf > 0:
            # 对'txt_clf_logits'进行argmax操作，获取文本分类预测结果
            txt_clf_preds = torch.argmax(result['txt_clf_logits'], dim=1).cpu()
            # 更新'txt_clf'文本分类准确率的度量
            avg_metrics['acc/txt_clf'].update(
                (txt_clf_preds == batch['tgt_obj_classes']).float().mean().item(),
                n=batch_size
            )

        if return_preds:
            # 如果需要返回预测结果，则遍历当前批次中的每个数据项
            for ib in range(batch_size):
                # 存储每个数据项的ID，目标对象ID，和对应的logits
                out_preds[batch['item_ids'][ib]] = {
                    'obj_ids': batch['obj_ids'][ib],
                    'obj_logits': result['og3d_logits'][ib, :batch['obj_lens'][ib]].data.cpu().numpy().tolist(),
                }
                # 如果需要输出注意力权重
                if output_attentions:
                    # 更新每个数据项的字典，添加自注意力和交叉注意力权重
                    out_preds[batch['item_ids'][ib]].update({
                        'all_self_attns': [
                            x[:, ib, :batch['obj_lens'][ib], :batch['obj_lens'][ib]].data.cpu().numpy().tolist() for x
                            in result['all_self_attns']],
                        'all_cross_attns': [
                            x[ib, :batch['obj_lens'][ib], :batch['txt_lens'][ib]].data.cpu().numpy().tolist() for x in
                            result['all_cross_attns']],
                    })

        # 如果设置了迭代次数限制，并且已达到该限制
        if niters is not None and ib >= niters:
            break  # 终止循环

        # 记录并打印每个度量的平均值
        LOGGER.info(', '.join(['%s: %.4f' % (lk, lv.avg) for lk, lv in avg_metrics.items()]))

        # 将模型设置回训练模式
        model.train()

        # 根据是否需要返回预测结果，返回相应的数据
        if return_preds:
            return avg_metrics, out_preds  # 返回度量和预测结果
        return avg_metrics  # 否则仅返回度量


def build_args():
    # 载入参数解析器
    parser = load_parser()
    # 使用配置文件和解析器来解析命令行参数
    opts = parse_with_config(parser)
    # 检查输出目录是否存在且不为空
    if os.path.exists(opts.output_dir) and os.listdir(opts.output_dir):
        # 如果输出目录已存在且不为空，打印警告信息
        LOGGER.warning(
            "Output directory ({}) already exists and is not empty.".format(
                opts.output_dir
            )
        )
    # 返回解析得到的参数
    return opts


if __name__ == '__main__':
    args = build_args()
    pprint.pprint(args)
    main(args)
