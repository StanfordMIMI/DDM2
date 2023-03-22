'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase, stage2_file=None):
    '''create dataset'''

    # unified data loader
    from data.mri_dataset import MRIDataset

    dataset = MRIDataset(dataroot=dataset_opt['dataroot'],
                valid_mask=dataset_opt['valid_mask'],
                phase=dataset_opt['phase'],
                val_volume_idx=dataset_opt['val_volume_idx'],
                val_slice_idx=dataset_opt['val_slice_idx'],
                padding=dataset_opt['padding'],
                in_channel=dataset_opt['in_channel'],
                image_size=dataset_opt['image_size'] if 'image_size' in dataset_opt else None,
                stage2_file=stage2_file
                )

    logger = logging.getLogger('base')
    logger.info('MRI dataset [{:s}] is created.'.format(dataset_opt['name']))
    return dataset
