import logging
from collections import OrderedDict
import copy
import torch
import torch.nn as nn
import os
import model.networks as networks
from model.mri_modules import unet
from model.mri_modules import train_noise_model
from .base_model import BaseModel
logger = logging.getLogger('base')

torch.set_printoptions(precision=10)

TWO_NETWORK = True


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        self.opt = opt

        model_opt = opt['noise_model']
        denoisor_opt = model_opt['unet']

        # basic uent
        self.denoisor = unet.UNet(
            in_channel=denoisor_opt['in_channel'],
            out_channel=denoisor_opt['out_channel'],
            norm_groups=denoisor_opt['norm_groups'],
            inner_channel=denoisor_opt['inner_channel'],
            channel_mults=denoisor_opt['channel_multiplier'],
            attn_res=denoisor_opt['attn_res'],
            res_blocks=denoisor_opt['res_blocks'],
            dropout=0.,
            image_size=opt['model']['diffusion']['image_size'],
            version=denoisor_opt['version'],
            with_noise_level_emb=False,   
        )

        self.netG = train_noise_model.N2N(
            self.denoisor
        )

        self.netG = self.set_device(self.netG)

        self.loss_type = opt['model']['loss_type']

        self.optG = torch.optim.Adam(
            self.netG.parameters(), lr=opt['train']["optimizer"]["lr"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optG, opt['train']['n_iter'], eta_min=opt['train']["optimizer"]["lr"]*0.01)

        self.log_dict = OrderedDict()
        self.load_network()
        self.counter = 0

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        
        self.optG.zero_grad()

        outputs = self.netG(self.data)
        
        l_pix = outputs['total_loss']
        l_pix.backward()


        self.optG.step()
        self.scheduler.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        
    def test(self, continous=False):
        self.netG.eval()
        #with torch.no_grad(): # TTT
        if isinstance(self.netG, nn.DataParallel):
            self.denoised = self.netG.module.denoise(
                self.data)
        else:
            self.denoised = self.netG.denoise(
                self.data)
        self.netG.train()

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['denoised'] = self.denoised.detach().float().cpu()
            out_dict['Y'] = self.data['Y'].detach().float().cpu()

        return out_dict

    def print_network(self):
        pass

    def save_network(self, epoch, iter_step, save_last_only=False):

        if not save_last_only:
            gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
            opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        else:
            gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'latest_gen.pth'.format(iter_step, epoch))
            opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'latest_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['noise_model']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
