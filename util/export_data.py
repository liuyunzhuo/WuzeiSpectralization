import numpy as np
import os
from scipy.io import savemat
import ntpath


def export_spectral_mat(webpage, img_path, spectral_cube, max_radiance):
    """Export Spectral Cube

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        img_path (str)           -- the string is used to create image paths
        spectral_cube(dict)      -- {cube: 0~1 spectral_cube, bands: wavelengths list}
        max_radiance (int)             -- turn 0~1 spectral_cube to radiance
    """
    web_dir = webpage.web_dir
    short_path = ntpath.basename(img_path[0])
    name = os.path.splitext(short_path)[0]

    save_folder = os.path.join(web_dir, 'mats')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    filename = os.path.join(save_folder, 'rec_' + name + '.mat')
    spectral_cube['cube'] *= max_radiance
    savemat(filename, spectral_cube)
