#!/usr/bin/env python3
"""
This class permutes labels 
for following permutation tests
"""
import numpy as np


class PermLabels:
    """
    Permutes labels and checks
    whether the permuted labels
    are accetable. 
    Parameters
    ----------
    labels : list of labels
    cv : sklearn crossvalidation 
    groups : list of groups 
    	     (check groups sklearn crossvalidation)
    random_state : int, to set a new randomization seed
    thresh : float, range 0-1 to ensure that a specific 
             portion of the data has been actually randomized 
    """

    def __init__(self, labels, cv, groups=None, random_state=None, thresh=0.9):
        self.labels = labels
        self.cv = cv
        self.groups = groups
        self.random_state = random_state
        self.thresh = thresh

    def _check_permutation(self, orig_labels, perm_labels):
        """ sanity check, to verify whether data
            has actually been permuted
        """
        
        ratio = np.sum(orig_labels != perm_labels) / len(orig_labels)
        return ratio > self.thresh

    def permute(self, labels):
        """ it does the actual permutation """
        
        r_state = np.random.RandomState(self.random_state)
        rand = r_state.permutation(len(labels))
        perm = labels[rand]
        while not self._check_permutation(labels, perm):
            rand = r_state.permutation(len(labels))
            perm = labels[rand]
        return perm

    def __call__(self):
        """
        Permute labels taking into account
        group subdivisions if necessary
        """
        perm_labels = []
        for train, _ in self.cv:
            fold = []
            if self.groups is not None:
                group = self.groups[train]
                for group_id in np.unique(group):
                    grp_mask = group == group_id
                    sel_labels = self.labels[train[grp_mask]]
                    fold.extend(list(self.permute(sel_labels)))
            else:
                sel_labels = self.labels[train]
                fold.extend(self.permute(sel_labels))
            perm_labels.append(fold)
        return perm_labels
