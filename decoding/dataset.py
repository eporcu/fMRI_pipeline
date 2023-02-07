from glob import glob
from os.path import join
import numpy as np


class GetData:
    """Creates a dataset for running
    sklearn analyses. It needs BIDS compliant data.
    Parameters
    ----------

    tasks : list of strings
    labels : dict keys are labels names,
             values are int (one for each distinct label)
    groups : list of strings

    Returns
    -------

    dataset : dict containing as keys tasks, labels and groups
              as values the corresponding lists.
    """

    def __init__(self, tasks, labels, group_filter=None):
        self.tasks = tasks
        self.labels = labels
        self.group_filter = group_filter

    def _get_bold(self, data_dir):
        """gets the functional data, by leveraging on tasks"""
        files = []
        for task in self.tasks:
            files.extend(glob(join(data_dir, f"*{task}*.nii.gz")))
        return np.array(files)

    def _get_labels(self, files):
        """organizes labels"""

        return np.array(
            [
                lab_value
                for img in files
                for label, lab_value in self.labels.items()
                if label in img
            ]
        )

    def _get_groups(self, files):
        """organizes groups"""

        return np.array(
            [
                n + 1
                for img in files
                for n, filt in enumerate(self.group_filter)
                if filt in img
            ]
        )

    def __call__(self, data_dir):
        """returns the dataset dictionary"""

        files = self._get_bold(data_dir)
        labels = self._get_labels(files)
        groups = self._get_groups(files)
        return {"data": files, "labels": labels, "groups": groups}
