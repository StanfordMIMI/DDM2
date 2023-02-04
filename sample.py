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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sample_sr3_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args, suffix='_sample')
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    opt['phase'] = 'val'

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    # logger.info(Logger.dict2str(opt))
    # tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test' or phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']
    sample_sum = opt['datasets']['val']['data_len']
    sample_sum = 20

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    # sample starts
    logger.info('Begin Model Evaluation.')

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)

    for _,  val_data in enumerate(val_loader):
        diffusion.feed_data(val_data)
        break

    print(sample_sum)
    for idx in range(sample_sum):   
        diffusion.sample(continous=False)
        visuals = diffusion.get_current_visuals(sample=True)

        show_img_mode = 'single'
        if show_img_mode == 'single':
            # single img series
            sample_img = visuals['denoised']  # uint8
            sample_num = sample_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(sample_img[iter]), '{}/{}_{}_sample_{}.png'.format(result_path, current_step, idx, iter))
        else:
            # grid img
            sample_img = Metrics.tensor2img(visuals['denoised'])  # uint8
            Metrics.save_img(
                sample_img, '{}/{}_{}_sample_process.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                Metrics.tensor2img(visuals['denoised'][-1]), '{}/{}_{}_sample.png'.format(result_path, current_step, idx))
