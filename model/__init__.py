import logging
logger = logging.getLogger('base')


def create_model(opt):
    from .model import DDM2 as M
    m = M(opt)
    logger.info('[{:s}] is created.'.format(m.__class__.__name__))
    return m

def create_noise_model(opt):
    from .model_stage1 import DDM2Stage1 as M
    m = M(opt)
    logger.info('Noise Model is created.')
    return m
