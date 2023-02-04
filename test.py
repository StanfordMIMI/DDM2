import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from tensorboardX import SummaryWriter
import os
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_64_512.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

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
            dataset_opt['val_volume_idx'] = 'all'
            dataset_opt['val_slice_idx'] = 'all'
            val_set = Data.create_dataset(dataset_opt, phase)
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

    # print(diffusion.netG.gmm.mu)
    # print(diffusion.netG.gmm.var)

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)

    # plot gmm
    # mu = diffusion.netG.gmm.mu.data.cpu().numpy()
    # var = diffusion.netG.gmm.var.data.cpu().numpy()
    # print(diffusion.netG.initial_stage)

    # print(mu.shape, var.shape)

    # print(mu)
    # print(var)

    # sigma = np.sqrt(var)

    # x = np.linspace(-1., 1., 100)
    # colors = ['r', 'g', 'b']

    # plt.plot(x, stats.norm.pdf(x, 0., 1.), c='gray')
    # for i in range(3):
    #     plt.plot(x, stats.norm.pdf(x, mu[0,i,0], sigma[0,i,0]), c=colors[i])

    # plt.show()



    for _,  val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)


        # results = diffusion.netG(val_data, False, False, debug=False)
        # debug_results = results['debug_results']
        # noise = debug_results['noise'].detach().cpu().numpy()[0,0,:,:]
        # recon = debug_results['recon'].detach().cpu().numpy()[0,0,:,:]
        # resample_noise = diffusion.netG.gmm.sample_noise(list(noise.shape)).detach().cpu().numpy()
        # # plt.imshow(np.hstack((val_data['X'].detach().cpu().numpy()[0,0,:,:], noise, resample_noise, recon)), cmap='gray')
        # plt.imshow(np.hstack((val_data['X'].detach().cpu().numpy()[0,0,:,:], recon, noise, np.random.randn(*noise.shape))), cmap='gray')
        # plt.show()
        # break

        diffusion.test(continous=False)
        visuals = diffusion.get_current_visuals(need_LR=False)
        #print(torch.max(visuals['denoised']), torch.min(visuals['denoised']))
        #break
        denoised_img = Metrics.tensor2img(visuals['denoised'], out_type=np.float32)  # uint8
        # input_img = Metrics.tensor2img(visuals['Y'])  # uint8

        # # save img
        # # Metrics.save_img(
        # #     denoised_img[:,:], '{}/{}_{}_denoised.png'.format(result_path, current_step, idx))
        # # Metrics.save_img(
        # #     input_img[:,:], '{}/{}_{}_input.png'.format(result_path, current_step, idx))

        # # save np
        volume_idx = (idx - 1) // val_set.raw_data.shape[-2]
        slice_idx = (idx - 1) % val_set.raw_data.shape[-2]
        if not os.path.exists(os.path.join(result_path, str(volume_idx))):
            os.mkdir(os.path.join(result_path, str(volume_idx)))
        Metrics.save_np(
            denoised_img[:,:], '{}/{}/{}.npy'.format(result_path, volume_idx, slice_idx))