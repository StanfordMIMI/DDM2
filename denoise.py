import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
import numpy as np
from dipy.io.image import save_nifti, load_nifti
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_64_512.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--align_mean', action='store_true')
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args, stage='denoise')
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test' or phase == 'val':
            
            #### evaluation slice ####
            if args.save:
                dataset_opt['val_volume_idx'] = 32 # save only the 32th volume
                dataset_opt['val_slice_idx'] = 'all' #save all slices
            ##########################

            val_set = Data.create_dataset(dataset_opt, phase, stage2_file=opt['stage2_file'])
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)

    if args.save:
        imgs = []
        denoised_volumes = []
        denoised_imgs = []


    for step,  val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=False)
        visuals = diffusion.get_current_visuals(need_LR=False)

        if not args.save:
            denoised_img = Metrics.tensor2img(visuals['denoised'])  # uint8
            input_img = Metrics.tensor2img(visuals['Y'])  # uint8

            # save img
            Metrics.save_img(
                denoised_img, '{}/{}_{}_denoised.png'.format(result_path, step, idx))
            Metrics.save_img(
                input_img[:,:], '{}/{}_{}_input.png'.format(result_path, step, idx))
        else:
            denoised_img = Metrics.tensor2img(visuals['denoised'], out_type=np.float32) # w, h, 1 
            denoised_volumes.append(denoised_img[...,None,None]) 
            if idx % len(val_set.val_slice_idx) == 0:
                idx = 0
                denoised_imgs.append(np.concatenate(denoised_volumes, axis=-2)) # w, h, N, 1
                denoised_volumes = []

        print('%d done %d to go!!' % (step, len(val_loader)))

    if args.save:
        denoised_imgs = np.concatenate(denoised_imgs, axis=-1) # w, h, N*L
        denoised_imgs = np.clip(denoised_imgs, 0., 1.)
        denoised_imgs = np.reshape(denoised_imgs, (denoised_imgs.shape[0], denoised_imgs.shape[1], len(val_set.val_slice_idx), len(val_set.val_volume_idx)))
        if args.align_mean:
            raw_normalized = val_set.raw_data.astype(np.float32) - np.min(val_set.raw_data, axis=(0,1), keepdims=True)
            raw_normalized = (raw_normalized.astype(np.float32) / np.max(raw_normalized, axis=(0,1), keepdims=True))
            raw_normalized_mean = np.mean(raw_normalized, axis=(0,1), keepdims=True)

            denoised_imgs -= np.min(denoised_imgs, axis=(0,1), keepdims=True)
            denoised_imgs = np.clip(denoised_imgs, 0., 1.)

        print('saving size:', denoised_imgs.shape)
        save_nifti('{}/{}_denoised.nii.gz'.format(result_path, opt['name']), denoised_imgs, affine=np.eye(4))
