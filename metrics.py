import dipy.reconst.dti as dti
import dipy.reconst.csdeconv as csd
import numpy as np
from dipy.segment.mask import median_otsu
import dipy.reconst.cross_validation as xval
import copy
import scipy.stats as stats
import os
from joblib import Parallel, delayed # this is for parallelization
from matplotlib import pyplot as plt
from dipy.io.image import save_nifti, load_nifti
import dipy.data as dpd
from dipy.core.gradients import gradient_table, unique_bvals_tolerance
import pandas as pd


class MRIMetrics():
    def __init__(self, gtab):
        self.gtab = gtab
        self.dti_model = dti.TensorModel(gtab)

    def fit_model(self, data):
        #response, ratio = csd.auto_response_ssst(self.gtab, data, roi_radius=10, fa_thr=0.7)
        response, ratio = csd.auto_response_ssst(self.gtab, data, roi_radii=10, fa_thr=0.7)
        csd_model = csd.ConstrainedSphericalDeconvModel(self.gtab, response)
        return csd_model, response

    def pearsonr(self, data, dti):
        return stats.pearsonr(data, dti)[0] ** 2

    def eval(self, data_slice, csd_model, response):

        dti_slice = xval.kfold_xval(self.dti_model, data_slice, 2)
        csd_slice = xval.kfold_xval(csd_model, data_slice, 2, response)
        print(data_slice.shape, dti_slice.shape)

        r2s_dti = []
        for i in range(0, dti_slice.shape[0]):
            for j in range(0, dti_slice.shape[1]):
                for k in range(0, dti_slice.shape[2]):
                    dti_r2 = stats.pearsonr(data_slice[i, j, k], dti_slice[i, j, k])[0] ** 2
                    r2s_dti.append(dti_r2)

        # r2s_dti = Parallel(n_jobs=8)(delayed(self.pearsonr)(data_slice[i, j, k], dti_slice[i, j, k]) for i in range(0, dti_slice.shape[0]) for j in range(0, dti_slice.shape[1]) for k in range(0, dti_slice.shape[2]))


        r2s_dti = np.array(r2s_dti)
        r2s_dti = r2s_dti[~np.isnan(r2s_dti)]

        r2s_csd = []
        for i in range(0, csd_slice.shape[0]):
            for j in range(0, csd_slice.shape[1]):
                for k in range(0, csd_slice.shape[2]):
                    csd_r2_mp = stats.pearsonr(data_slice[i, j, k], csd_slice[i, j, k])[0] ** 2

        # r2s_csd = Parallel(n_jobs=8)(delayed(self.pearsonr)(data_slice[i, j, k], csd_slice[i, j, k]) for i in range(0, csd_slice.shape[0]) for j in range(0, csd_slice.shape[1]) for k in range(0, csd_slice.shape[2]))

        r2s_csd = np.array(r2s_csd)
        r2s_csd = r2s_csd[~np.isnan(r2s_csd)]

        return dti_slice, csd_slice

    def calc(self, data, slice=38):
        csd_model, response = self.fit_model(data)
        
        # mask with otsu
        _, mask = median_otsu(data, vol_idx=[0, 1])
        data_masked = copy.deepcopy(data)
        
        if slice is not None:
            data_masked = data_masked[..., slice:slice+1, :]
            data_masked[mask[..., slice:slice+1] == 0] = 0

        dti, csd = self.eval(data_masked, csd_model, response)

        return dti, csd


class DTIMetrics():
    def __init__(self, gtab):
        self.gtab = gtab
        self.dti_model = dti.TensorModel(gtab)

    def eval(self, data_slice):

        dti_slice = xval.kfold_xval(self.dti_model, data_slice, 2)

        r2s_dti = []
        for i in range(0, dti_slice.shape[0]):
            for j in range(0, dti_slice.shape[1]):
                for k in range(0, dti_slice.shape[2]):
                    dti_r2 = stats.pearsonr(data_slice[i, j, k], dti_slice[i, j, k])[0] ** 2
                    r2s_dti.append(dti_r2)

        # r2s_dti = Parallel(n_jobs=8)(delayed(self.pearsonr)(data_slice[i, j, k], dti_slice[i, j, k]) for i in range(0, dti_slice.shape[0]) for j in range(0, dti_slice.shape[1]) for k in range(0, dti_slice.shape[2]))

        r2s_dti = np.array(r2s_dti)
        r2s_dti = r2s_dti[~np.isnan(r2s_dti)]

        return dti_slice

    def calc(self, data, slice=38):
        # mask with otsu
        # _, mask = median_otsu(data, vol_idx=[0, 1])
        # data_masked = copy.deepcopy(data)
        return self.eval(data[..., slice:slice+1, :])
        

def load_ours_single_stage(path):
    
    volumes = []
    for volume_idx in range(0, 64):
        slices = []
        for slice_idx in range(0, 60):
            slices.append(np.load(os.path.join(path, str(volume_idx), str(slice_idx)+'.npy')))
        volumes.append(np.array(slices))
    volumes = np.array(volumes).transpose((2, 3, 1, 0))
    print(volumes.shape)
    np.save('/media/administrator/1305D8BDB8D46DEE/stanford/ours_slices_v25/stage2.npy', volumes)
    return volumes


def load_ours():
    #stage0 = load_ours_single_stage('/media/administrator/1305D8BDB8D46DEE/stanford/MRI/experiments/sb_stage0_results/results')
    #stage1 = load_ours_single_stage('/media/administrator/1305D8BDB8D46DEE/stanford/MRI/experiments/sb_stage1_results/results')
    #stage2 = load_ours_single_stage('/media/administrator/1305D8BDB8D46DEE/stanford/MRI/experiments/sb_stage2_results/results')
    stage0 = np.load('/media/administrator/1305D8BDB8D46DEE/stanford/ours_slices/stage0.npy').astype(np.float32)
    stage1 = np.load('/media/administrator/1305D8BDB8D46DEE/stanford/ours_slices/stage1.npy').astype(np.float32)
    stage2 = np.load('/media/administrator/1305D8BDB8D46DEE/stanford/ours_slices/stage2.npy').astype(np.float32)
    #print(np.max(stage0), np.max(stage1), np.max(stage2))
    return stage0 / 255., stage1 / 255., stage2 / 255.

def plot(data_dti, mp_dti, p2s_dti, our_dti):
    import seaborn as sns
    from statannot import add_stat_annotation

    df_diff = pd.DataFrame({'(MP) DTI':mp_dti - data_dti,
                        '(P2S) DTI':p2s_dti - data_dti,
                        '(Our) DTI':our_dti - data_dti})
    
    
    sns.set(style="whitegrid")
    ax = sns.boxplot(x="variable", y="value", data=pd.melt(df_diff), fliersize=0, sym='', palette="Set2")

    
    add_stat_annotation(ax, data=pd.melt(df_diff), x="variable", y="value",
                        box_pairs=[('(MP - Noisy) DTI', '(P2S - Noisy) DTI', '(Our - Noisy) DTI')],
                                test='t-test_ind', text_format='star', loc='outside', verbose=2)

if __name__ == '__main__':

    # load results
    #load_ours_single_stage('/media/administrator/1305D8BDB8D46DEE/stanford/MRI/experiments/v25_220326_084934/results')
    #exit()
    
    # loading gtab
    data_root = '/media/administrator/1305D8BDB8D46DEE/stanford/sr3/scripts/data/'
    _, gtab = dpd.read_sherbrooke_3shell()

    bvals = gtab.bvals
    bvecs = gtab.bvecs

    sel_b = np.logical_or(bvals == 0, bvals == 2000)

    gtab = gradient_table(bvals[sel_b], bvecs[sel_b])

    # loading original datla
    data, _ = load_nifti(os.path.join(data_root, 'HARDI193.nii.gz'))
    
    #data = data.astype(np.float32) / max_data
    data = data[..., sel_b]
    max_data = np.max(data, axis=(0,1,2), keepdims=True)

    # loading our data
    stage1 = np.load('/media/administrator/1305D8BDB8D46DEE/stanford/ours_slices_v25/stage1.npy').astype(np.float32)
    data_ours = np.concatenate((data[:,:,:,[0]], stage1), axis=-1)
    data_ours[:,:,:,1:] = data_ours[:,:,:,1:] * max_data[:,:,:,1:]

    # loading p2s
    data_p2s, _ = load_nifti('/home/administrator/stanford/patch2self-master/notebooks/denoised_hardi193_p2s_mlp.nii.gz')
    #data_p2s = data_p2s.astype(np.float32) / max_data
    data_p2s[:,:,:,0] = data[:,:,:,0]
    data_p2s = data_p2s[..., sel_b]

    # loading mp
    data_mp, _ = load_nifti(os.path.join(data_root, 's3sh_mp.nii.gz'))
    #data_mp = data_mp.astype(np.float32) / max_data
    data_mp[:,:,:,0] = data[:,:,:,0]
    data_mp = data_mp[..., sel_b]

    # plt.imshow(np.hstack((data[:,:,40,40], data_mp[:,:,40,40], data_p2s[:,:,40,40], data_ours[:,:,40,40])), cmap='gray')
    # plt.show()

    # exit()

    # DTI calculation
    M = DTIMetrics(gtab)

    dti_raw = M.calc(data, slice=38)

    dti_mp = M.calc(data_mp, slice=38)

    print('MP:', np.mean(dti_mp - dti_raw))

    dti_p2s = M.calc(data_p2s, slice=38)

    print('P2S:', np.mean(dti_p2s - dti_raw))

    dti_ours = M.calc(data_ours, slice=38)

    print('Ours:', np.mean(dti_ours - dti_raw))

    # plot
    plot(dti_raw, dti_mp, dti_p2s, dti_ours)


    # CSD calculation TODO

    # M = CSDMetrics(gtab)

    # csd_raw = M.calc(data, slice=38)

    # csd_mp = M.calc(data_mp, slice=38)

    # print('MP:', np.mean(csd_mp - csd_raw))

    # csd_p2s = M.calc(data_p2s, slice=30)

    # print('P2S:', np.mean(csd_p2s - csd_raw))

    # csd_ours = M.calc(data_ours, slice=None)

    # print('Ours:', np.mean(csd_ours - csd_raw))

