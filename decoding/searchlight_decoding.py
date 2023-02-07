#!/usr/bin/env python3
"""
Script for running nilearn searchlight for classification (sklearn).
It allows to run permutations on training data, which by default
is not possible on nilearn searchlight.
"""
import argparse
from os.path import join
from dataset import GetData
import nibabel as nib
import nilearn.decoding
import numpy as np
from permute_labels import PermLabels
from sklearn.model_selection import LeaveOneGroupOut


def np2nii(img, scores, filename):
    """
    It saves data into a nifti file
    by leveraging on another image
    of the same size.
    Parameters
    ----------
    img : nifti file (e.g a mask)
    scores : numpy array containing decoding scores
    filename : string name of the new nifti file
    Returns
    -------
    nii_file : Nifti1Image
    """
    img = nib.load(img)
    header = nib.Nifti1Header()
    affine = img.affine
    nii_file = nib.Nifti1Image(scores, affine, header)
    nib.save(nii_file, filename)
    return nii_file


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-d", "--data_dir", metavar="PATH", required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=False)
    parser.add_argument("-t", "--tasks", type=str, nargs="+", required=True)
    parser.add_argument("-l", "--labels", type=str, nargs="+", required=True)
    parser.add_argument("-g", "--groups", type=str, nargs="+", required=True)
    parser.add_argument("-m", "--mask", type=str, required=True)
    parser.add_argument("-r", "--radius", type=float, required=True)
    parser.add_argument("-s", "--subj", type=str, required=True)
    parser.add_argument("-p", "--permutation", type=int, required=False)
    parser.add_argument(
        "-e", "--estimator", const="svc", nargs="?", type=str, default="svc"
    )
    args = parser.parse_args()

    # get the dataset
    dataset = GetData(
        tasks=args.tasks,
        labels={l: n + 1 for n, l in enumerate(args.labels)},
        group_filter=args.groups,
    )
    data = dataset(args.data_dir)
    print(data)

    # instantiate the cross validation
    cv = LeaveOneGroupOut()

    if args.permutation:
        print("you are now running the searchlight for the chance maps")
        cv = list(cv.split(data["data"], data["labels"], data["groups"]))

        # permute labels
        permutation = PermLabels(
            data["labels"], cv, data["groups"], random_state=int(args.subj)
        )

        for fold, ((train, test), perm) in enumerate(zip(cv, permutation())):
            print(f"fold number: {fold + 1}")
            print(f"train indices: {train} - test indices: {test}")
            print(
                (
                    f"non-permuted train labels: {data['labels'][train]}"
                    f" permuted train labels: {perm}"
                )
            )

            labels = np.copy(data["labels"])
            print(data["labels"])
            print("length train", len(train), "length perm", len(perm))
            labels[train] = perm
            print(labels)

            SL = nilearn.decoding.SearchLight(
                mask_img=args.mask,
                # process_mask_img=process_mask_img,
                radius=args.radius,
                estimator=args.estimator,
                n_jobs=1,
                verbose=1,
                cv=[(train, test)],
                scoring="accuracy",
            )
            SL.fit(data["data"], labels, data["groups"])
            scores = SL.scores_
            print(np.mean(scores[scores > 0]))
            output = join(
                f"{args.output_dir}",
                (
                    f"sub-{args.subj}_{args.tasks[0]}_{args.tasks[1]}_"
                    f"radius_{int(args.radius)}_perm_"
                    f"{str(args.permutation)}_fold_{fold+1}"
                ),
            )
            np.save(f"{output}.npy", scores)
    else:
        SL = nilearn.decoding.SearchLight(
            mask_img=args.mask,
            # process_mask_img=process_mask_img,
            radius=args.radius,
            estimator=args.estimator,
            n_jobs=1,
            verbose=1,
            cv=cv,
            scoring="accuracy",
        )
        SL.fit(data["data"], data["labels"], data["groups"])
        scores = SL.scores_
        output = join(
            f"{args.output_dir}",
            (
                f"sub-{args.subj}_{args.tasks[0]}_"
                f"{args.tasks[1]}_radius_{int(args.radius)}"
            ),
        )
        print(scores[scores > 0])
        np.save(f"{output}.npy", scores)
        np2nii(args.mask, scores, f"{output}.nii.gz")
