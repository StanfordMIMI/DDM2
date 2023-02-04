import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
# from tensorboardX import SummaryWriter
import os
import numpy as np

print(torch.__version__, torch.version.cuda)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args, stage=1)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info('[Phase 1] Training noise model!')

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    trainer = Model.create_noise_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = trainer.begin_step
    current_epoch = trainer.begin_epoch
    n_iter = opt['noise_model']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                trainer.feed_data(train_data)
                trainer.optimize_parameters()
                # log
                if current_step % opt['noise_model']['print_freq'] == 0:
                    logs = trainer.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                    logger.info(message)

                # validation
                if current_step % opt['noise_model']['val_freq'] == 0:
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        trainer.feed_data(val_data)
                        trainer.test(continous=True)
                        
                        visuals = trainer.get_current_visuals()

                        denoised_img = Metrics.tensor2img(visuals['denoised'])  # uint8
                        input_img = Metrics.tensor2img(visuals['X'])  # uint8

                        Metrics.save_img(
                            denoised_img[:,:], '{}/{}_{}_denoised.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            input_img[:,:], '{}/{}_{}_input.png'.format(result_path, current_step, idx))

                if current_step % opt['noise_model']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    trainer.save_network(current_epoch, current_step, save_last_only=True)
        # save model
        logger.info('End of training.')
