#!/usr/bin/env python3

"""
first level analysis:
wrapper for nilearn first level GLM 
"""
from glob import glob
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn.glm.first_level.design_matrix import _convolve_regressors
from nilearn.glm.first_level import (
    make_first_level_design_matrix,
    FirstLevelModel,
)
from nilearn.glm.contrasts import compute_fixed_effects
from nilearn.image import index_img
from nilearn.plotting import plot_design_matrix
import numpy as np
from scipy.linalg import block_diag


class FirstLevel:
    """
    Wrapper for first level analysis
    """

    def __init__(
        self,
        func_dir,
        events_dir,
        tr,
        reg_names=None,
        n_dummy=None,
        hrf="spm + derivative",
        drift_model="polynomial",
        drift_order=1,
        high_pass=0.01,
        fir_delays=[0],
        n_scans=None,
        param_mod=None,
    ):
        self.func_dir = func_dir
        self.events_dir = events_dir
        self.hrf = hrf
        self.tr = tr
        self.reg_names = reg_names
        self.n_dummy = n_dummy
        self.drift_model = drift_model
        self.drift_order = drift_order
        self.high_pass = high_pass
        self.fir_delays = fir_delays
        self.n_scans = n_scans
        self.param_mod = param_mod
        self.events = self._get_events()
        self.confounds = self._get_confounds()
        self.func = sorted(glob(join(self.func_dir, "*_bold.nii.gz")))
        self.mask = self._get_mask()

    def _get_mask(self):
        """
        Load all the masks present in the func directory.
        It requires BIDS forma
        """
        return [mask for mask in sorted(glob(join(self.func_dir, "*_mask.nii.gz")))]

    def _get_events(self):
        """
        Load events according to BIDS nomenclature
        """
        if self.param_mod is not None:
            rel_reg = ["onset", "duration", "trial_type", "modulation"]
            return [
                pd.read_csv(ev, sep="\t").rename(
                    columns={self.param_mod: "modulation"}
                )[rel_reg]
                for ev in sorted(glob(join(self.events_dir, "*_events.tsv")))
            ]
        else:
            rel_regr = ["onset", "duration", "trial_type"]
            return [
                pd.read_csv(ev, sep="\t")[rel_regr]
                for ev in sorted(glob(join(self.events_dir, "*_events.tsv")))
            ]

    def _get_confounds(self):
        """
        Load fmriprep confounds, get rid of
        counfounds relative to the dummy scans
        """
        fmriprep_conf = sorted(glob(join(self.func_dir, "*_timeseries.tsv")))
        return [
            pd.read_csv(conf, sep="\t")[self.reg_names]
            .loc[self.n_dummy :]
            .reset_index(drop=True)
            for conf in fmriprep_conf
        ]

    def mk_design_mat(self, events, confounds):
        """
        Computes design matrix per run
        """
        frame_times = np.arange(self.n_scans - self.n_dummy) * self.tr
        if self.param_mod is not None:
            # check whether NANs are present
            # if any, make a new "no_response" column
            if any(events.modulation.isna()):
                events.loc[events.modulation.isna(), "trial_type"] = "no_response"
            idx_nan = events.modulation.isna()
            # de-mean
            events.modulation = events.modulation - events.modulation.mean()
            # check this one!
            events.loc[idx_nan, "modulation"] = 1
            mod_conv = _convolve_regressors(
                events, hrf_model=self.hrf, frame_times=frame_times, fir_delays=[0]
            )
            p_mod = pd.DataFrame(
                mod_conv[0], columns=["_".join(["par_mod", reg]) for reg in mod_conv[1]]
            )
            confounds = pd.concat([p_mod, confounds], axis=1).reset_index(drop=True)
        # get rid of the "modulation" column, no longer needed
        events = events[["onset", "duration", "trial_type"]]
        return make_first_level_design_matrix(
            frame_times,
            events,
            hrf_model=self.hrf,
            add_regs=confounds,
            add_reg_names=self.reg_names,
            drift_order=self.drift_order,
            drift_model=self.drift_model,
            high_pass=self.high_pass,
            fir_delays=self.fir_delays,
        )

    def diag_design_matrix(self):
        """
        It produces an SPM-like design matrix
        with the design matrices of all the
        runs in the diagonal.
        """
        des_matrices, cols = [], []
        for ev, conf in zip(self.events, self.confounds):
            d_mat = self.mk_design_mat(ev, conf)
            des_matrices.append(d_mat)
            cols.extend(list(d_mat.columns))
        return pd.DataFrame(block_diag(*(des_matrices)), columns=cols)

    def plot_dm(self, design_matrix):
        """
        Plots the design matrix.
        """
        plot_design_matrix(design_matrix, rescale=True, ax=None, output_file=None)
        plt.show()

    def _remove_dummies(self, img):
        """
        This method removes superfluous dummy scans
        and checks whether the number of scans
        is what was expected.
        """
        bold_img = index_img(img, slice(self.n_dummy, self.n_scans))
        if bold_img.shape[-1] != (self.n_scans - self.n_dummy):
            raise ValueError(
                f"Original scans are {nib.load(img).shape[-1]},"
                f"there are {bold_img.shape[-1]} scans after dummy removal,"
                f"but according to your input you should have {(self.n_scans - self.n_dummy)}"
            )
        else:
            return bold_img

    def _fixed_effects(self, stats, output=None):
        """
        Put together the fixed effects runs.
        Parameters
        ----------
        stats : dictionary containing stats from the
                contrasts.
        Returns
        -------
        glm_stats : dictionary containing one Niimage
                    per contrast
        """
        glm_stats = {}
        for key, _ in stats.items():
            contrast_imgs = [i["effect_size"] for i in stats[key]]
            variance_imgs = [i["effect_variance"] for i in stats[key]]
            fx_contr, fx_var, fx_stat = compute_fixed_effects(
                contrast_imgs, variance_imgs
            )
            glm_stats[key] = fx_stat
            if output is not None:
                fx_stat.to_filename(f"{output}_{key}.nii.gz")
        return glm_stats

    def __call__(
        self,
        contrasts,
        noise_model="ar1",
        des_mat_model="multiple",
        univar=True,
        output=None,
    ):
        """
        Fit the first level over all the runs and saves data
        """
        fl_model = FirstLevelModel(
            self.tr, hrf_model=self.hrf, noise_model=noise_model, mask_img=self.mask[0]
        )  # do you want to use a unique mask for all the runs?
        if isinstance(contrasts, tuple):
            stats = {key: [] for key in contrasts[0].keys()}
        else:
            stats = {key: [] for key in contrasts.keys()}
        if des_mat_model == "unique":

            # remove dummies
            bold_img = [self._remove_dummies(img) for img in self.func]
            # create diagonal design matrix
            design_mat = self.diag_design_matrix()
            glm = fl_model.fit(bold_img, design_matrices=design_mat)
            for contrast_id, contrast_value in contrasts.items():
                contr_stats = glm.compute_contrast(contrast_value, output_type="all")
                stats[contrast_id].append(contr_stats)

        elif des_mat_model == "multiple":
            # loop over the runs
            for run, (func_img, ev, conf) in enumerate(
                zip(self.func, self.events, self.confounds)
            ):
                if isinstance(contrasts, tuple):
                    contr = contrasts[run]
                else:
                    contr = contrasts

                bold_img = self._remove_dummies(func_img)
                design_mat = self.mk_design_mat(ev, conf)
                glm = fl_model.fit(bold_img, design_matrices=design_mat)
                for contrast_id, contrast_value in contr.items():
                    contr_stats = glm.compute_contrast(
                        contrast_value, output_type="all"
                    )
                    if univar:
                        stats[contrast_id].append(contr_stats)
                    else:
                        contr_stats["z_score"].to_filename(
                            f"{output}_{contrast_id}_run_{run+1}.nii.gz"
                        )
        else:
            raise ValueError(
                f"{des_mat_model} does not exist, choose beetween 'separate' and 'unique'"
            )
        if univar:
            return self._fixed_effects(stats, output)


def mk_contrasts(design_matrix):
    """
    Basic contrast dictionary.
    use this one to create more complex
    contrasts.
    """
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = {
        column: contrast_matrix[i] for i, column in enumerate(design_matrix.columns)
    }
    return contrasts
