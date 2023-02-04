import logging
from collections import OrderedDict
import copy
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel


logger = logging.getLogger('base')

torch.set_printoptions(precision=10)


class DDM2(BaseModel):
    def __init__(self, opt):
        super(DDM2, self).__init__(opt)
        self.opt = opt
        
        # TTT
        if 'TTT' in opt:
            self.use_ttt = True
        else:
            self.use_ttt = False

        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))

        self.schedule_phase = None

        self.loss_type = opt['model']['loss_type']

        # set loss and load resume state
        self.set_loss()

        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')

        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if k.find('denoise_fn') >= 0:
                        continue
                    if k.find('noise_model_variance') >= 0:
                        continue
                    optim_params.append(v)
                    #if k.find('noise_model_variance') >= 0:
                #optim_params = list(self.netG.parameters())
            print('Optimizing: '+str(len(optim_params))+' params')
            
            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])

            self.log_dict = OrderedDict()
        
        #self.print_network()
        self.load_network()
        self.counter = 0

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()

        outputs = self.netG(self.data)
        
        if torch.is_tensor(outputs):
            l_pix = outputs
            l_pix.backward()
            self.optG.step()

        elif type(outputs) is dict:
            l_pix = outputs['total_loss']

            total_loss = l_pix
            total_loss.backward()
            self.optG.step()
    
        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False):
        if self.use_ttt:
            optim_params = []
            for k, v in self.netG.named_parameters():
                if k.find('denoise_fn') >= 0:
                    continue
                optim_params.append(v)
            
            ttt_opt = torch.optim.Adam(
                optim_params, lr=self.opt['TTT']["optimizer"]["lr"])
        else:
            self.netG.eval()
            ttt_opt = None

        if isinstance(self.netG, nn.DataParallel):
            self.denoised = self.netG.module.denoise(
                self.data, continous, ttt_opt=ttt_opt)
        else:
            self.denoised = self.netG.denoise(
                self.data, continous, ttt_opt=ttt_opt)
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.denoised = self.netG.module.sample(self.data, continous)
            else:
                self.denoised = self.netG.sample(self.data, continous)
        self.netG.train()

    def interpolate(self, continous=False, lams=[0.5]):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.denoised = self.netG.module.interpolate(self.data, continous,lams=lams)
            else:
                self.denoised = self.netG.interpolate(self.data, continous,lams=lams)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

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

    def get_current_visuals(self, need_LR=True, sample=False, interpolate=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['denoised'] = self.denoised.detach().float().cpu()
        elif interpolate:
            out_dict['img1_input']=self.denoised[0].detach().float().cpu()
            out_dict['img2_input']=self.denoised[1].detach().float().cpu()
            out_dict['img1_denoised']=self.denoised[2].detach().float().cpu()
            out_dict['img2_denoised']=self.denoised[3].detach().float().cpu()
            out_dict['interpolated'] = []
            for idx, img in enumerate(self.denoised[-1]):
                out_dict['interpolated'].append(img.detach().float().cpu())
        else:
            out_dict['denoised'] = self.denoised.detach().float().cpu()
            out_dict['X'] = self.data['X'].detach().float().cpu()

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
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading stage2 pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            # network.load_state_dict(torch.load(
            #     gen_path), strict=(not self.opt['model']['finetune_norm']))
            network.load_state_dict(torch.load(
                gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
        elif self.opt['noise_model']['resume_state'] is not None:
            load_path = self.opt['noise_model']['resume_state']
            gen_path = '{}_gen.pth'.format(load_path)
            state_dict = torch.load(gen_path)
            logger.info(
                'Loading stage1 pretrained model for G [{:s}] ...'.format(gen_path))
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.load_state_dict(state_dict, strict=False)
            else:
                self.netG.load_state_dict(state_dict, strict=False)
