import logging
import os
import time
import csv
import torch
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval,R1_mAP_eval_LTCC
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from torch.cuda import amp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import gc
from utils.phase4r_telemetry import collect_component_gradient_telemetry


class _Phase2MetricWriter:
    """Forward SummaryWriter calls while collecting per-phase scalar metrics."""

    def __init__(self, writer):
        self.writer = writer
        self._records = []

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        self.writer.add_scalar(tag, scalar_value, global_step, *args, **kwargs)
        self._records.append((str(tag), float(scalar_value)))

    def reset_records(self):
        self._records = []

    def scalar_values(self):
        values = {}
        for tag, value in self._records:
            values.setdefault(tag, []).append(value)
        return values

    def close(self):
        self.writer.close()


def unfreeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = True    

def do_train(cfg,
             model,
             train_loader,
             local_rank,
             dataset,
             val_loader = None,
             val_loader_same = None):
    log_period = cfg.SOLVER.LOG_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    
    loss_fn, center_criterion = make_loss(cfg, num_classes=dataset.num_train_pids)


    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("DIFFER.train")
    logger.info('start training')
    

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    if device:
        model.to(local_rank)
        #loss_fn.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            #
            print('Using {} GPUs for training, current GPU number {}'.format(torch.cuda.device_count(),local_rank))
            # rank = dist.get_rank()
            # device_id = rank % torch.cuda.device_count()
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)
            #loss_fn = torch.nn.parallel.DistributedDataParallel(loss_fn, device_ids=[local_rank])
    train_writer = _Phase2MetricWriter(SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'train')))
    is_main_process = (not cfg.MODEL.DIST_TRAIN) or dist.get_rank() == 0
    telemetry_enabled = bool(getattr(cfg.TRAIN, 'GRADIENT_TELEMETRY', False))
    telemetry_interval = int(getattr(cfg.TRAIN, 'GRADIENT_TELEMETRY_INTERVAL', 100))
    telemetry_file = None
    telemetry_csv = None
    if telemetry_enabled:
        telemetry_file = open(
            os.path.join(cfg.OUTPUT_DIR, f'gradient_telemetry_rank{local_rank}.csv'),
            'w',
            newline='',
        )
        telemetry_csv = csv.DictWriter(telemetry_file, fieldnames=[
            'epoch', 'iteration', 'rank', 'group', 'scaler_scale',
            'core_norm_pre_scaled', 'hair_norm_pre_scaled',
            'clothing_norm_pre_scaled', 'reverse_norm_pre_scaled',
            'core_norm_scaled', 'hair_norm_scaled', 'clothing_norm_scaled',
            'reverse_norm_scaled', 'core_norm_unscaled', 'hair_norm_unscaled',
            'clothing_norm_unscaled', 'reverse_norm_unscaled',
            'cos_core_hair', 'cos_core_clothing', 'cos_hair_clothing',
            'cos_core_reverse', 'hair_to_core_norm_ratio',
            'clothing_to_core_norm_ratio', 'reverse_to_core_norm_ratio',
            'hair_nonzero', 'clothing_nonzero', 'reverse_nonzero',
            'hair_conflict', 'clothing_conflict', 'reverse_conflict',
            'parameter_count',
        ])
        telemetry_csv.writeheader()
        telemetry_file.flush()
        logger = logging.getLogger("DIFFER.train")
        logger.info(
            "Phase 4R gradient telemetry enabled: interval=%d, path=%s",
            telemetry_interval,
            telemetry_file.name,
        )
    metrics_file = None
    metrics_csv = None
    if is_main_process:
        metrics_file = open(os.path.join(cfg.OUTPUT_DIR, 'metrics_per_epoch.csv'), 'w', newline='')
        metrics_csv = csv.DictWriter(metrics_file, fieldnames=[
            'epoch', 'train_loss', 'ce_loss', 'triplet_loss', 'bio_loss',
            'reverse_hair_loss', 'reverse_cloth_loss', 'lr', 'lr_bias',
            'bio_accuracy', 'reverse_hair_accuracy', 'reverse_cloth_accuracy',
            'amp_scale_min', 'amp_scale_max', 'amp_scale_end',
            'skipped_steps', 'found_inf_count', 'nonfinite_events',
            'train_accuracy', 'cc_rank1', 'cc_rank5', 'cc_rank10', 'cc_mAP',
            'cc_mINP', 'general_rank1', 'general_rank5', 'general_rank10',
            'general_mAP', 'general_mINP',
        ])
        metrics_csv.writeheader()
        metrics_file.flush()
   
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    if cfg.DATA.DATASET == 'ltcc':
        evaluator_diff = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)  # ltcc
        evaluator_general = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)  # prcc
        evaluator_same = R1_mAP_eval(dataset.num_query_imgs_same, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,ignore_cam=True)


    ppu_initial_scale = 512 if cfg.DATA.DATASET == 'prcc' else 1024
    scaler = amp.GradScaler(init_scale=ppu_initial_scale)
    best_rank1 = -np.inf
    best_epoch = 0
    start_train_time = time.time()
    
    model.eval()
    #evaluator.reset()
    epoch=0
    if cfg.DATA.DATASET == 'ltcc':
        evaluator_diff.reset()
        evaluator_general.reset()

    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff.reset()
        evaluator_same.reset()
    else:
        evaluator.reset()

    pretrainModelTest=False
    if pretrainModelTest:
        logger.info("Test the model at the beginning of training")
        if cfg.DATA.DATASET == 'prcc':
            logger.info("Clothes changing setting")
            rank1= test(cfg, model, evaluator_diff, val_loader, logger, device,epoch, train_writer)
            logger.info("Standard setting")
            test(cfg, model, evaluator_same, val_loader_same, logger, device, epoch,  train_writer,test=True)
        elif cfg.DATA.DATASET == 'ltcc':
            logger.info("Clothes changing setting")
            rank1 = test(cfg, model, evaluator_diff, val_loader, logger, device,epoch,train_writer, ltcc=True)
            logger.info("Standard setting")
            test(cfg, model, evaluator_general, val_loader, logger, device, epoch, train_writer,test=True)
        else:
            rank1= test(cfg, model, evaluator, val_loader, logger, device,epoch,train_writer)   
    

    for epoch in range(cfg.TRAIN.START_EPOCH, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        train_writer.reset_records()
        amp_scale_min = float(scaler.get_scale())
        amp_scale_max = float(scaler.get_scale())
        amp_scale_end = float(scaler.get_scale())
        amp_skipped_steps = 0
        amp_found_inf_count = 0
        amp_nonfinite_events = 0


        if cfg.DATA.DATASET == 'ltcc':
            evaluator_diff.reset()
            evaluator_general.reset()

        elif cfg.DATA.DATASET == 'prcc':
            evaluator_diff.reset()
            evaluator_same.reset()
        else:
            evaluator.reset()
        
        scheduler.step(epoch)
        model.train()
      
        for idx, data in enumerate(train_loader):
            step=idx+len(train_loader)*epoch
            
            samples=data['image']
            targets=data['pid']
            camids=data['camid']
            clothes=data['clothes_id']
            caption_feature=data['caption_feature'] if 'caption_feature' in data else None
                   
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
           
            with amp.autocast(enabled=True):
              
                score, feat = model(samples, cam_label=camids)
                

            loss = loss_fn(score, feat, targets, camids,caption_feature,clothes_ids=clothes,train_writer=train_writer,step=step)

            if telemetry_enabled and ((idx + 1) % telemetry_interval == 0):
                telemetry_rows = collect_component_gradient_telemetry(
                    model=model,
                    components=getattr(loss_fn, 'last_components', {}),
                    scaler_scale=float(scaler.get_scale()),
                    epoch=epoch,
                    iteration=(epoch - 1) * len(train_loader) + idx + 1,
                    rank=local_rank,
                )
                for telemetry_row in telemetry_rows:
                    telemetry_csv.writerow(telemetry_row)
                telemetry_file.flush()

            loss_is_finite = bool(torch.isfinite(loss.detach()))
            amp_nonfinite_events += int(not loss_is_finite)
            
            train_writer.add_scalar('loss/total', loss.item(), step)

            current_lr=scheduler._get_lr(epoch)[0]
            train_writer.add_scalar('lr',current_lr , step)
            
            scaler.scale(loss).backward()

            amp_scale_before_step = float(scaler.get_scale())
            scaler.step(optimizer)
            found_inf_state = scaler._per_optimizer_states[id(optimizer)]['found_inf_per_device']
            found_inf = sum(float(value.item()) for value in found_inf_state.values())
            amp_found_inf_count += int(found_inf > 0)
            amp_skipped_steps += int(found_inf > 0)
            amp_nonfinite_events += int(found_inf > 0)
            if found_inf > 0:
                logger.info(
                    "AMP_SKIP epoch=%d iteration=%d scale_before_step=%.1f",
                    epoch,
                    idx + 1,
                    amp_scale_before_step,
                )
            scaler.update()
            amp_scale_end = float(scaler.get_scale())
            amp_scale_min = min(amp_scale_min, float(scaler.get_scale()))
            amp_scale_max = max(amp_scale_max, float(scaler.get_scale()))
            if 'center' in cfg.MODEL.LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == targets).float().mean()
            elif isinstance(score, dict):
                if isinstance(score['cls_score'], list):
                    acc = (score['cls_score'][0].max(1)[1] == targets).float().mean()
                else:
                    acc = (score['cls_score'].max(1)[1] == targets).float().mean()
            else:
                acc = (score.max(1)[1] == targets).float().mean()

            loss_meter.update(loss.item(), samples.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (idx + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (idx + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, current_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (idx + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        train_scalars = train_writer.scalar_values()

        def _mean_scalar(tag, missing='NA'):
            values = train_scalars.get(tag, [])
            return sum(values) / len(values) if values else missing

        if epoch % eval_period == 0 :#and cfg.MODEL.DIST_TRAIN and dist.get_rank() == 0:
            model.eval()
            if cfg.DATA.DATASET == 'prcc':
                logger.info("Clothes changing setting")
                train_writer.reset_records()
                rank1 = test(cfg, model, evaluator_diff, val_loader, logger, device, epoch,
                             train_writer, metric_prefix='cc')
                cc_scalars = train_writer.scalar_values()
                logger.info("Standard setting")
                train_writer.reset_records()
                test(cfg, model, evaluator_same, val_loader_same, logger, device, epoch,
                     train_writer, test=True, metric_prefix='general')
                general_scalars = train_writer.scalar_values()
            elif cfg.DATA.DATASET == 'ltcc':
                logger.info("Clothes changing setting")
                train_writer.reset_records()
                rank1 = test(cfg, model, evaluator_diff, val_loader, logger, device, epoch,
                             train_writer, ltcc=True, metric_prefix='cc')
                cc_scalars = train_writer.scalar_values()
                logger.info("Standard setting")
                train_writer.reset_records()
                test(cfg, model, evaluator_general, val_loader, logger, device, epoch,
                     train_writer, test=True, metric_prefix='general')
                general_scalars = train_writer.scalar_values()
            else:
                train_writer.reset_records()
                rank1 = test(cfg, model, evaluator, val_loader, logger, device, epoch,
                             train_writer, metric_prefix='cc')
                cc_scalars = train_writer.scalar_values()
                general_scalars = {}

            def _eval_value(values, tag):
                entries = values.get(tag, [])
                return entries[-1] if entries else 'NA'

            if is_main_process:
                lr_values = sorted(float(group['lr']) for group in optimizer.param_groups)
                metrics_csv.writerow({
                    'epoch': epoch,
                    'train_loss': _mean_scalar('loss/total'),
                    'ce_loss': _mean_scalar('loss/id'),
                    'triplet_loss': _mean_scalar('loss/triplet'),
                    'bio_loss': _mean_scalar('loss/clip_bio_0'),
                    'reverse_hair_loss': _mean_scalar('loss/clip_bio_reverse_0'),
                    'reverse_cloth_loss': _mean_scalar('loss/clip_bio_reverse_1'),
                    'lr': lr_values[0],
                    'lr_bias': lr_values[-1],
                    'bio_accuracy': _mean_scalar('acc/clip_bio_0'),
                    'reverse_hair_accuracy': _mean_scalar('acc/clip_bio_reverse_0'),
                    'reverse_cloth_accuracy': _mean_scalar('acc/clip_bio_reverse_1'),
                    'amp_scale_min': amp_scale_min,
                    'amp_scale_max': amp_scale_max,
                    'amp_scale_end': amp_scale_end,
                    'skipped_steps': amp_skipped_steps,
                    'found_inf_count': amp_found_inf_count,
                    'nonfinite_events': amp_nonfinite_events,
                    'train_accuracy': float(acc_meter.avg),
                    'cc_rank1': _eval_value(cc_scalars, 'cc/rank1'),
                    'cc_rank5': _eval_value(cc_scalars, 'cc/rank5'),
                    'cc_rank10': _eval_value(cc_scalars, 'cc/rank10'),
                    'cc_mAP': _eval_value(cc_scalars, 'cc/mAP'),
                    'cc_mINP': 'NA',
                    'general_rank1': _eval_value(general_scalars, 'general/rank1'),
                    'general_rank5': _eval_value(general_scalars, 'general/rank5'),
                    'general_rank10': _eval_value(general_scalars, 'general/rank10'),
                    'general_mAP': _eval_value(general_scalars, 'general/mAP'),
                    'general_mINP': 'NA',
                })
                metrics_file.flush()

            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch
                logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))
                if is_main_process:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                    logger.info("Save the best model")

    if is_main_process:
        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_final.pth'))
        logger.info("Save the final model at epoch {}".format(epochs))
        logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))
        if metrics_file is not None:
            metrics_file.close()

    if telemetry_file is not None:
        telemetry_file.close()

    train_writer.close()

    total_time = time.time() - start_train_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def do_inference(cfg,
                 model,
                 dataset,
                 val_loader = None,
                 val_loader_same = None):
    logger = logging.getLogger("DIFFER.test")
    logger.info("Enter inferencing")

    logger.info("transreid inferencing")
    device = "cuda"
    if cfg.DATA.DATASET == 'ltcc':
        evaluator_diff = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)  # ltcc
        evaluator_general = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)  # prcc
        evaluator_same = R1_mAP_eval(dataset.num_query_imgs_same, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,ignore_cam=True)
    if cfg.DATA.DATASET == 'ltcc':
        evaluator_diff.reset()
        evaluator_general.reset()

    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff.reset()
        evaluator_same.reset()
    else:
        evaluator.reset()
    model.to(device)
    model.eval()
    if cfg.DATA.DATASET == 'prcc':
        logger.info("Clothes changing setting")
        test(cfg, model, evaluator_diff, val_loader, logger, device, test=True,clothChanging=True)
        logger.info("Standard setting")
        test(cfg, model, evaluator_same, val_loader_same, logger, device, test=True)
    elif cfg.DATA.DATASET == 'ltcc':
        logger.info("Clothes changing setting")
        test(cfg, model, evaluator_diff, val_loader, logger, device, test=True,ltcc=True,clothChanging=True)
        logger.info("Standard setting")
        test(cfg, model, evaluator_general, val_loader, logger, device, test=True)
    else:
        test(cfg, model, evaluator, val_loader, logger, device, test=True,clothChanging=True)

def test(cfg, model, evaluator, val_loader, logger, device, epoch=None, test_writer=None,
         test=False,ltcc=False,clothChanging=False,metric_prefix=None):
        
    for n_iter, data in enumerate(val_loader):
        if n_iter%100==0:
            logger.info(f"processing batch {n_iter}/{len(val_loader)}")
    
        imgs=data['image']
        pids=data['pid']
        camids=data['camid']
        clothes_id=data['clothes_id']
        img_paths=data['image_path']
        
        with torch.no_grad():
            
            imgs = imgs.to(device) 
         
            camids0=camids.to(device) 
           
            feat = model(imgs, cam_label=camids0)
            
            if ltcc:
                evaluator.update((feat, pids, camids, clothes_id,img_paths))
            else:
                evaluator.update((feat, pids, camids,clothes_id,img_paths))
                
   
    cmc, mAP, _, _, _, queryFeature, galleryFeature,imgpath = evaluator.compute()
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("mAP-:{:.1%}".format(mAP))
    if test_writer is not None:
        prefix = f'{metric_prefix}/' if metric_prefix else ''
        test_writer.add_scalar(prefix + 'rank1', float(cmc[0]), epoch)
        test_writer.add_scalar(prefix + 'rank5', float(cmc[4]), epoch)
        test_writer.add_scalar(prefix + 'rank10', float(cmc[9]), epoch)
        test_writer.add_scalar(prefix + 'mAP', float(mAP), epoch)

    rank1 = cmc[0] 
    if test :
        torch.cuda.empty_cache()
        return rank1  
   
    
    logger.info("Validation Results - Epoch: {}".format(epoch))
    #rank1 = cmc[0]
    
    torch.cuda.empty_cache()
    return rank1
