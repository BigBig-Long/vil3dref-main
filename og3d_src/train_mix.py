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

from data.gtlabelpcd_dataset import GTLabelPcdDataset, gtlabelpcd_collate_fn

from model.referit3d_net_mix import ReferIt3DNetMix



def build_datasets(data_cfg):
    # 创建训练数据集
    trn_dataset = GTLabelPcdDataset(
        data_cfg.trn_scan_split,  # 训练集扫描数据分割
        data_cfg.anno_file,       # 注释文件路径
        data_cfg.scan_dir,        # 扫描数据目录
        data_cfg.category_file,   # 类别文件路径
        cat2vec_file=data_cfg.cat2vec_file,  # 类别向量文件路径
        random_rotate=data_cfg.get('random_rotate', False),  # 是否随机旋转，默认不旋转
        max_txt_len=data_cfg.max_txt_len,  # 最大文本长度
        max_obj_len=data_cfg.max_obj_len,  # 最大对象长度
        keep_background=data_cfg.get('keep_background', False),  # 是否保留背景，默认不保留
        num_points=data_cfg.num_points,  # 点数
        in_memory=True,  # 是否全部载入内存
        gt_scan_dir=data_cfg.get('gt_scan_dir', None),  # 地面真实扫描数据目录
        iou_replace_gt=data_cfg.get('iou_replace_gt', 0),  # 用IOU替换GT的阈值
    )
    # 创建验证数据集
    val_dataset = GTLabelPcdDataset(
        data_cfg.val_scan_split,  # 验证集扫描数据分割
        data_cfg.anno_file,       # 注释文件路径
        data_cfg.scan_dir,        # 扫描数据目录
        data_cfg.category_file,   # 类别文件路径
        cat2vec_file=data_cfg.cat2vec_file,  # 类别向量文件路径
        max_txt_len=None,  # 验证集不限制文本长度
        max_obj_len=None,  # 验证集不限制对象长度
        random_rotate=False,  # 验证集不进行随机旋转
        keep_background=data_cfg.get('keep_background', False),  # 是否保留背景，默认不保留
        num_points=data_cfg.num_points,  # 点数
        in_memory=True,  # 是否全部载入内存
        gt_scan_dir=data_cfg.get('gt_scan_dir', None),  # 地面真实扫描数据目录
        iou_replace_gt=data_cfg.get('iou_replace_gt', 0),  # 用IOU替换GT的阈值
    )
    # 返回构建的训练集和验证集
    return trn_dataset, val_dataset



def main(opts):
    # 设置CUDA设备
    default_gpu, n_gpu, device = set_cuda(opts)
    # 如果是默认GPU设备，则进行下一步配置
    if default_gpu:
        # 打印设备信息，包括设备号、GPU数量和是否分布式训练
        LOGGER.info(
            'device: {} n_gpu: {}, distributed training: {}'.format(
                device, n_gpu, bool(opts.local_rank != -1)
            )
        )
    # 设置随机种子以保证实验的可重复性
    seed = opts.seed
    # 如果是分布式训练，调整随机种子以区分不同的进程
    if opts.local_rank != -1:
        seed += opts.rank
    # 应用设置的随机种子
    set_random_seed(seed)
    # 如果是默认GPU设备，并且不是测试模式
    if default_gpu:
        if not opts.test:
            # 保存训练的元数据
            save_training_meta(opts)
            # 创建TensorBoard日志记录器
            TB_LOGGER.create(os.path.join(opts.output_dir, 'logs'))
            # 初始化模型保存器
            model_saver = ModelSaver(os.path.join(opts.output_dir, 'ckpts'))
            # 添加日志到文件
            add_log_to_file(os.path.join(opts.output_dir, 'logs', 'log.txt'))
    else:
        # 如果是测试模式，则禁用日志记录器
        LOGGER.disabled = True
        # 进度条设置为无操作，即不显示进度条
        pbar = NoOp()
        # 模型保存器设置为无操作，即不保存模型
        model_saver = NoOp()
    # 准备模型配置
    model_config = EasyDict(opts.model)
    # 初始化模型，使用配置和指定的设备
    model = ReferIt3DNetMix(model_config, device)
    # 初始化总参数数和可训练参数数
    num_weights, num_trainable_weights = 0, 0
    # 遍历模型的所有参数
    for p in model.parameters():
        # 计算每个参数的元素数
        psize = np.prod(p.size())
        # 累加所有参数的元素数
        num_weights += psize
        # 如果参数可训练，累加到可训练参数的总数
        if p.requires_grad:
            num_trainable_weights += psize
    # 记录总参数数和可训练参数数
    LOGGER.info('#weights: %d, #trainable weights: %d', num_weights, num_trainable_weights)
    # 如果存在需要恢复的文件
    if opts.resume_files:
        checkpoint = {}
        # 遍历所有的恢复文件
        for resume_file in opts.resume_files:
            # 加载每一个检查点文件
            new_checkpoints = torch.load(resume_file, map_location=lambda storage, loc: storage)
            # 合并检查点
            for k, v in new_checkpoints.items():
                if k not in checkpoint:
                    checkpoint[k] = v
        # 如果只有一个恢复文件
        if len(opts.resume_files) == 1:
            # 直接加载状态字典
            model.load_state_dict(checkpoint)
        # 打印恢复的参数信息
        print(
            'resume #params:', len(checkpoint),
            len([n for n in checkpoint.keys() if n in model.teacher_model.state_dict()]),
            len([n for n in checkpoint.keys() if n in model.student_model.state_dict()]),
        )
        # 分别加载教师和学生模型的状态字典
        model.teacher_model.load_state_dict(checkpoint, strict=False)
        if opts.resume_student:
            model.student_model.load_state_dict(checkpoint, strict=False)
        else:
            # 单独加载学生模型的检查点
            student_checkpoint = torch.load(
                opts.resume_files[0], map_location=lambda storage, loc: storage
            )
            print('resume_student', len(student_checkpoint))
            model.student_model.load_state_dict(student_checkpoint, strict=False)
    # 获取模型配置
    model_cfg = model.config
    # 将模型包装为适应指定设备和分布式设置的形式
    model = wrap_model(model, device, opts.local_rank)
    # 加载训练集数据配置
    data_cfg = EasyDict(opts.dataset)
    # 建立训练集和验证集
    trn_dataset, val_dataset = build_datasets(data_cfg)
    # 定义数据批次整合函数
    collate_fn = gtlabelpcd_collate_fn
    # 记录训练集的扫描数和数据数
    LOGGER.info('train #scans %d, #data %d' % (len(trn_dataset.scan_ids), len(trn_dataset)))
    # 记录验证集的扫描数和数据数
    LOGGER.info('val #scans %d, #data %d' % (len(val_dataset.scan_ids), len(val_dataset)))
    # 构建数据加载器
    if opts.local_rank == -1:
        # 不使用分布式采样器
        trn_sampler = None
        pre_epoch = lambda e: None
    else:
        # 获取分布式环境的大小和排名
        size = dist.get_world_size()
        # 构建分布式数据采样器
        trn_sampler = DistributedSampler(
            trn_dataset, num_replicas=size, rank=dist.get_rank(), shuffle=True
        )
        # 设置每个epoch开始时采样器的行为
        pre_epoch = trn_sampler.set_epoch
    # 创建训练数据加载器
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
    # 计算总的训练步数
    opts.num_train_steps = len(trn_dataloader) * opts.num_epoch
    # 如果是测试模式
    if opts.test:
        # 执行验证，并返回预测结果
        val_log, out_preds = validate(model, model_cfg, val_dataloader, return_preds=True)
        # 定义预测结果保存的目录
        pred_dir = os.path.join(opts.output_dir, 'preds')
        # 创建目录，如果目录已存在则不报错
        os.makedirs(pred_dir, exist_ok=True)
        # 将预测结果保存为JSON文件
        json.dump(out_preds, open(os.path.join(pred_dir, 'val_outs.json'), 'w'))
        # 测试模式下完成后直接返回
        return
    # 准备优化器
    optimizer, init_lrs = build_optimizer(model, opts)
    # 打印使用的GPU数量
    LOGGER.info(f"***** Running training with {opts.world_size} GPUs *****")
    # 打印批处理大小，如果是分布式则计算总批处理大小
    LOGGER.info("  Batch size = %d", opts.batch_size if opts.local_rank == -1 else opts.batch_size * opts.world_size)
    # 打印训练的总周期数和步数
    LOGGER.info("  Num epoch = %d, num steps = %d", opts.num_epoch, opts.num_train_steps)
    # 初始化用于计算训练统计的工具
    avg_metrics = defaultdict(AverageMeter)
    # 初始化全局步数
    global_step = 0
    # 将模型设为训练模式
    model.train()
    # 清除优化器的梯度
    optimizer.zero_grad()
    # 执行一步优化
    optimizer.step()
    # 如果使用默认GPU
    if default_gpu:
        # 执行模型验证
        val_log = validate(model, model_cfg, val_dataloader)
        # 初始化记录最佳验证得分的字典
        val_best_scores = {'epoch': -1, 'acc/og3d': -float('inf')}
        # 设置训练周期迭代器
        epoch_iter = range(opts.num_epoch)
        # 如果使用默认GPU，使用进度条显示每个周期的进度
        if default_gpu:
            epoch_iter = tqdm(epoch_iter)
        # 迭代每个训练周期
        for epoch in epoch_iter:
            # 分布式训练前的周期性调用
            pre_epoch(epoch)
            # 记录周期开始时间
            start_time = time.time()
            # 设置训练批次迭代器
            batch_iter = trn_dataloader
            # 如果使用默认GPU，使用进度条显示每个批次的进度
            if default_gpu:
                batch_iter = tqdm(batch_iter)
            # 迭代处理每个批次
            for batch in batch_iter:
                # 获取批次大小
                batch_size = len(batch['scan_ids'])
                # 在模型上执行前向计算和损失计算
                result, losses = model(batch, compute_loss=True)
                # 反向传播总损失
                losses['total'].backward()
                # 累加全局步数
                global_step += 1
                # 计算学习率衰减
                lr_decay_rate = get_lr_sched_decay_rate(global_step, opts)
                # 更新每个参数组的学习率
                for kp, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = lr_this_step = init_lrs[kp] * lr_decay_rate
                # 记录学习率到TensorBoard
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)
                # 记录每个损失项
                loss_dict = {'loss/%s' % lk: lv.data.item() for lk, lv in losses.items()}
                for lk, lv in loss_dict.items():
                    avg_metrics[lk].update(lv, n=batch_size)
                TB_LOGGER.log_scalar_dict(loss_dict)
                TB_LOGGER.step()
                # 如果指定了梯度裁剪，执行裁剪
                if opts.grad_norm != -1:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), opts.grad_norm
                    )
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                # 更新模型参数
                optimizer.step()
                optimizer.zero_grad()
            # 记录本周期的训练信息
            LOGGER.info(
                'Epoch %d, lr: %.6f, %s', epoch + 1,
                optimizer.param_groups[-1]['lr'],
                ', '.join(['%s: %.4f' % (lk, lv.avg) for lk, lv in avg_metrics.items()])
            )
            # 每隔指定周期进行一次模型验证
            if default_gpu and (epoch + 1) % opts.val_every_epoch == 0:
                LOGGER.info(f'------Epoch {epoch + 1}: start validation (val)------')
                val_log = validate(model, model_cfg, val_dataloader)
                TB_LOGGER.log_scalar_dict(
                    {f'valid/{k}': v.avg for k, v in val_log.items()}
                )
                # 保存模型和优化器的状态
                output_model_file = model_saver.save(
                    model, epoch + 1, optimizer=optimizer, save_latest_optim=True
                )
                # 检查并更新最佳验证得分
                if val_log['acc/og3d'].avg > val_best_scores['acc/og3d']:
                    val_best_scores['acc/og3d'] = val_log['acc/og3d'].avg
                    val_best_scores['epoch'] = epoch + 1
                    model_saver.remove_previous_models(epoch + 1)
                else:
                    # 删除未超过最佳得分的模型文件
                    os.remove(output_model_file)

    LOGGER.info('Finished training!')
    LOGGER.info(
        'best epoch: %d, best acc/og3d %.4f', val_best_scores['epoch'], val_best_scores['acc/og3d']
    )
@torch.no_grad()  # 不进行梯度计算，用于模型评估
def validate(model, model_cfg, val_dataloader, niters=None, return_preds=False):
    model.eval()  # 将模型设置为评估模式
    avg_metrics = defaultdict(AverageMeter)  # 初始化存储平均度量值的字典
    out_preds = {}  # 用于存储输出预测结果的字典
    for ib, batch in enumerate(val_dataloader):  # 迭代验证数据集
        batch_size = len(batch['scan_ids'])  # 获取批次大小
        # 执行模型前向传递，计算结果和损失
        result, losses = model(batch, compute_loss=True, is_test=True)
        loss_dict = {'loss/%s' % lk: lv.data.item() for lk, lv in losses.items()}  # 收集损失数据
        for lk, lv in loss_dict.items():  # 更新损失平均值
            avg_metrics[lk].update(lv, n=batch_size)
        # 评估3D对象分类准确性
        og3d_preds = torch.argmax(result['og3d_logits'], dim=1).cpu()  # 获取3D对象分类预测
        avg_metrics['acc/og3d'].update(  # 计算并更新准确率
            torch.mean((og3d_preds == batch['tgt_obj_idxs']).float()).item(),
            n=batch_size
        )
        avg_metrics['acc/og3d_class'].update(  # 计算并更新分类准确率
            torch.mean((batch['obj_classes'].gather(1, og3d_preds.unsqueeze(1)).squeeze(1) == batch['tgt_obj_classes']).float()).item(),
            n=batch_size
        )
        if model_cfg.losses.obj3d_clf:  # 如果模型配置中包含对象3D分类损失
            obj3d_clf_preds = torch.argmax(result['obj3d_clf_logits'], dim=2).cpu()
            avg_metrics['acc/obj3d_clf'].update(
                (obj3d_clf_preds[batch['obj_masks']] == batch['obj_classes'][batch['obj_masks']]).float().mean().item(),
                n=batch['obj_masks'].sum().item()
            )
        if model_cfg.losses.obj3d_clf_pre:  # 预训练的3D对象分类
            obj3d_clf_preds = torch.argmax(result['obj3d_clf_pre_logits'], dim=2).cpu()
            avg_metrics['acc/obj3d_clf_pre'].update(
                (obj3d_clf_preds[batch['obj_masks']] == batch['obj_classes'][batch['obj_masks']]).float().mean().item(),
                n=batch['obj_masks'].sum().item()
            )
        if model_cfg.losses.txt_clf:  # 文本分类
            txt_clf_preds = torch.argmax(result['txt_clf_logits'], dim=1).cpu()
            avg_metrics['acc/txt_clf'].update(
                (txt_clf_preds == batch['tgt_obj_classes']).float().mean().item(),
                n=batch_size
            )
        if model_cfg.losses.get('rot_clf', False):  # 旋转分类
            for il in model_cfg.mm_encoder.rot_layers:
                rot_clf_preds = result['all_rot_preds'][il].cpu()
                gt_views = batch['target_views']
                gt_view_mask = gt_views != -100
                avg_metrics['acc/rot_clf_%d' % il].update(
                    (rot_clf_preds == gt_views)[gt_view_mask].float().mean().item(),
                    n=torch.sum(gt_view_mask)
                )
        if model_cfg.losses.get('txt_contrast', False):  # 文本对比
            txt_ctr_preds = result['txt_pos_sims'] > torch.max(result['txt_neg_sims'], 1)[0]
            txt_ctr_preds = txt_ctr_preds.cpu().float()
            avg_metrics['acc/txt_contrast'].update(
                txt_ctr_preds.mean().item(),
                n=batch_size
            )

        if return_preds:
            # 如果需要返回预测结果，迭代每个批次中的样本
            for ib in range(batch_size):
                # 将每个样本的相关预测信息存储到out_preds字典中
                out_preds[batch['item_ids'][ib]] = {
                    'obj_ids': batch['obj_ids'][ib],  # 对象ID
                    'obj_logits': result['og3d_logits'][ib, :batch['obj_lens'][ib]].data.cpu().numpy().tolist(),
                    # 对象预测的逻辑值
                }
            # 如果设置了迭代次数限制，达到限制后中断
            if niters is not None and ib >= niters:
                break
        # 日志输出当前所有平均度量值
        LOGGER.info(', '.join(['%s: %.4f' % (lk, lv.avg) for lk, lv in avg_metrics.items()]))
        # 将模型重新设置为训练模式
        model.train()
        # 根据return_preds标志，决定返回值
        if return_preds:
            return avg_metrics, out_preds  # 返回度量值和预测结果
        return avg_metrics  # 仅返回度量值


def build_args():
    # 加载命令行解析器
    parser = load_parser()
    # 使用配置文件和命令行参数来解析选项
    opts = parse_with_config(parser)
    # 检查输出目录是否存在且不为空
    if os.path.exists(opts.output_dir) and os.listdir(opts.output_dir):
        # 如果输出目录已存在并且不为空，则发出警告
        LOGGER.warning(
            "Output directory ({}) already exists and is not empty.".format(
                opts.output_dir
            )
        )
    # 返回解析得到的选项
    return opts


if __name__ == '__main__':
    args = build_args()
    pprint.pprint(args)
    main(args)
