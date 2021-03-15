#!/usr/bin/env python
"""
External job for fitting FSC curves and finding the value at which they cross 0.5
"""

import argparse
import json
import os
import os.path
import shutil
import time
import numpy as np
from scipy.optimize import curve_fit, fsolve, brentq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gemmi


RELION_JOB_FAILURE_FILENAME = "RELION_JOB_EXIT_FAILURE"
RELION_JOB_SUCCESS_FILENAME = "RELION_JOB_EXIT_SUCCESS"


def run_job(project_dir, out_dir, fscs_files, args_list):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--j", dest="threads", help="Number of threads to run (ignored)"
    )
    args = parser.parse_args(args_list)

    resolutions = []

    for i, starin in enumerate(fscs_files):
        if not starin.endswith(".star"):
            raise ValueError("Input files containing FSC curves must end with .star")
        fsc_in = gemmi.cif.read_file(os.path.join(project_dir, starin))
        data_as_dict = json.loads(fsc_in.as_json())["fsc"]
        invres = [1 / x for x in data_as_dict["_rlnangstromresolution"]]
        fsc = data_as_dict["_rlnfouriershellcorrelationcorrected"]
        res = fsc_res(invres, fsc, out_dir, i + 1)
        resolutions.append(res)

    print(resolutions)
    class_index = resolutions.index(min(resolutions))

    return class_index


def fsc_res(invres, fsc, out_dir, iclass):
    # placeholder function
    def fitfcn(x, a, b, eps=0):
        return 1 / (1 + a * np.exp(-b * x ** 2)) - eps

    fitres = curve_fit(fitfcn, invres, fsc, p0=[1, 1])
    # solution = fsolve(fitfcnsolve, 0.1, args=(fitres[0][0], fitres[0][1]))
    solution = brentq(fitfcn, 0.001, 1, args=(fitres[0][0], fitres[0][1], 0.5))

    fitrange = np.linspace(0, invres[-1])
    fitline = [fitfcn(x, fitres[0][0], fitres[0][1]) for x in fitrange]

    plt.plot(invres, fsc, "o")
    plt.plot(fitrange, fitline, "--")
    plt.savefig(out_dir + f"fsc_curve{iclass}.pdf")
    plt.close()

    return solution


def main():
    """Change to the job working directory, then call run_job()"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir", dest="out_dir", help="Directory for the FSC fitting External job"
    )
    parser.add_argument(
        "--i",
        nargs="+",
        dest="fscs",
        help="Input star file names containing FSC curves for comparison",
    )
    parser.add_argument(
        "--pipeline_control", help="Directory for pipeline control files"
    )
    known_args, other_args = parser.parse_known_args()
    project_dir = os.getcwd()
    os.makedirs(known_args.out_dir, exist_ok=True)
    os.chdir(known_args.out_dir)
    if os.path.isfile(RELION_JOB_FAILURE_FILENAME):
        print(" fsc_fitting_external_job: Removing previous failure indicator file")
        os.remove(RELION_JOB_FAILURE_FILENAME)
    if os.path.isfile(RELION_JOB_SUCCESS_FILENAME):
        print(" fsc_fitting_external_job: Removing previous success indicator file")
        os.remove(RELION_JOB_SUCCESS_FILENAME)
    try:
        os.chdir("../..")
        class_index = run_job(
            project_dir, known_args.out_dir, known_args.fscs, other_args
        )
        with open(
            os.path.join(project_dir, known_args.out_dir, "BestClass.txt"), "w"
        ) as f:
            f.write(str(class_index))
    except:
        os.chdir(known_args.out_dir)
        open(RELION_JOB_FAILURE_FILENAME, "w").close()
        raise
    else:
        os.chdir(known_args.out_dir)
        open(RELION_JOB_SUCCESS_FILENAME, "w").close()


if __name__ == "__main__":
    main()