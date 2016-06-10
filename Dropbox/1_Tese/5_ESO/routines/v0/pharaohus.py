# ==============================================================================
# !/usr/bin/env python
# -*- coding:utf-8 -*-


# import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
import matplotlib.font_manager as fm
import os
import csv
import sys
from matplotlib import cm

sys.path.append('/home/sysadmin/.local/lib/python2.7/site-packages/'
                'pyhdust-0.98-py2.7.egg/')
sys.path.append('/scratch/home/sysadmin/lmfit-py/lmfit/__init__.py')
sys.path.append('/scratch/home/sysadmin/.local/lib/python2.7/site-packages/'
                'lmfit-0.9.3rc1_32_g9a6f751-py2.7.egg')
sys.path.append('/scratch/home/sysadmin/Downloads/lineid_plot-master')
sys.path.append('/scratch/home/sysadmin/.local/lib/python2.7/site-packages/'
                'lineid_plot.py to lineid_plot.pyc')
sys.path.append('/scratch/home/sysadmin/.local/lib/python2.7/site-packages/'
                'lineid_plot-0.3-py2.7.egg-info')
sys.path.append('/scratch/home/sysadmin/.local/lib/python2.7/site-packages/'
                'lmfit-0.9.3rc1_32_g9a6f751-py2.7.egg/lmfit/__init__.pyc')
sys.path.append('/scratch/home/sysadmin/.local/lib/python2.7/site-packages/'
                'lmfit-0.9.3rc1_32_g9a6f751-py2.7.egg/lmfit/__init__.py')
sys.path.append('/scratch/home/sysadmin//.local/lib/python2.7/site-packages/'
                'matplotlib-1.5.1+1425.g378bfd5-py2.7-linux-x86_64.egg/'
                'matplotlib/')
sys.path.insert(1, '/scratch/home/sysadmin/.local/lib/python2.7/'
                'site-packages/')

import lineid_plot
import pyhdust.phc as phc
import pyhdust.spectools as spt
import pyhdust.bcd as bcd
from pyhdust.interftools import imshowl
from scipy.interpolate import griddata
from itertools import product as prod

mpl.rcParams.update({'font.size': 18})
mpl.rcParams['lines.linewidth'] = 2
font = fm.FontProperties(size=17)
mpl.rc('xtick', labelsize=17)
mpl.rc('ytick', labelsize=17)
fontsize_label = 18  # 'x-large'

num_spa = 87  # 176

__version__ = "0.0.1"
__author__ = "Ahmed Shoky, Bruno Mota, Daniel Moser"


# ==============================================================================
class data_object:
    '''
    Class of the stellar objects. Using this class, we can store
    for each star a sort of variables, which can be easily accessed.

    '''

    kind = 'star'     # class variable shared by all instances

    def __init__(self, name, lamb0, D):
        # instance variable unique to each instance
        self.name = name
        self.lamb0 = lamb0
        self.D = D


# ==============================================================================
def create_list_files(list_name, folder, folder_table):
    '''
    Creates a list of the names inside a given folder.

    :param list_name: list's name (string)
    :param folder: files' folder
    :return: creates a txt file, with the files' paths
    '''

    a = open(folder_table + list_name + ".txt", "w")
    for path, subdirs, files in os.walk(folder):
        for filename in files:
            f = os.path.join(path, filename)
            a.write(str(f) + os.linesep)
    return


# ==============================================================================
def read_list_files_all(table_name, folder_table):
    '''
    Read list of files in a table, and returns all fits file in an array.

    :param table_name: Table's name (string)
    :param reflex: If you want to unzip the files to run reflex
    :return: list of files (txt file)
    '''

    file_data = folder_table + table_name

    files = open(file_data, 'r')
    lines = files.readlines()

    list_files = []
    for i in range(len(lines)):
        list_files.append(lines[i][:-1])

    files.close()

    return list_files


# ==============================================================================
def find_nearest(array, value):
    '''
    Find the nearest value inside an array.

    :param array: array
    :param value: desired value (float)
    :return: nearest value and its index (float)
    '''

    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


# ==============================================================================
def create_txt_file(x, y, w, z, h, q, file_name):
    '''
    Create a txt file.

    :param x: array with n elements (array)
    :param y: array with n elements (array)
    :param file_name: file's name (string)
    :return: txt file
    '''

    writer = open(file_name, 'w')
    writer.write('#D_o    lb0_o   te_ip   lgg_ip  Tp_ip       Ms_ip\n')
    writer.close()

    with open(file_name, 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(x, y, w, z, h, q))

    return


# ==============================================================================
def read_txt(table_name):
    '''
    Read a simple txt file.

    :param table_name: name of the table
    :param ncols: number of columns
    :return: x, y (arrays) or x, y, z (arrays)
    '''

    data = np.loadtxt(table_name)

    return data


# ==============================================================================
def read_txt_2(table_name, ncols):
    '''
    Read a simple txt file.

    :param table_name: name of the table
    :param ncols: number of columns
    :return: x, y (arrays)
    '''

    if ncols == 2:
        type = (0, 1)
        a = np.loadtxt(table_name, dtype='float', comments='#',
                       delimiter='\t', skiprows=0, usecols=type,
                       unpack=True, ndmin=0)
        x, y = a[0], a[1]
        return x, y

    if ncols == 4:
        type = (0, 1, 2, 3)
        a = np.loadtxt(table_name, dtype='float', comments='#',
                       delimiter='\t', skiprows=0, usecols=type,
                       unpack=True, ndmin=0)
        x, y, w, z = a[0], a[1], a[2], a[3]
        return x, y, w, z


# ==============================================================================
def create_class_object(list_files):
    '''
    Create object class for all targets in a list.

    :param list_files: list of XShooter fits files (array)
    :return: data (class object)
    '''

    data = []
    for i in range(len(list_files)):
        # Plotting UBV only
        # Rewrite the original fits
        string = str(list_files[i][:])
        rule = '(?!.*\/)(.*)'
        match = re.search(rule, string)

        star = match.group(1)

        print('\nStar: %s' % star)
        print('File: %s' % list_files[i])

        fo = open(list_files[i], "r")
        nlines = len(fo.readlines())

        if nlines != 0:

            if len(list_files) != 1:
                D, lamb0 = read_txt_2(table_name=list_files[i], ncols=2)
            else:
                D, lamb0 = read_txt_2(table_name=list_files, ncols=2)

            data.append(data_object(name=star, lamb0=lamb0, D=D))

        else:
            data.append(data_object(name=star, lamb0=0, D=0))

    return data


# ==============================================================================
def rotate(matrix, degree):

    '''
    Function to rotate a matrix.

    :param matrix: matrix (array)
    :param degree: rotation (float)
    :return: matrix (array)
    '''

    if degree == 0:
        return matrix
    elif degree > 0:
        return rotate(zip(*matrix[::-1]), degree - 90)
    else:
        return rotate(zip(*matrix)[::-1], degree + 90)


# ==============================================================================
def main():
    global data, name_ffit, D, lamb0, Teff, Tpol, Mass, logg,\
        models_data, grid_x, grid_y, input_obs_data_1,\
        input_obs_data_2, grid_z1

    print(num_spa * '=')
    print("\n                                       PHARAOHUS \n")

# ------------------------------------------------------------------------------
    # You can define these parameters
    resolution = 1000.  # resolution of the color maps
    read_files = True
    interpol = True
    plot3d = False
    plot_labels = False  # Plot points data alongside
    cmap = 'inferno'  # 'nipy_spectral' 'gist_rainbow' 'plasma'
    interpol_method = 'linear'  # 'nearest' 'cubic' 'linear'
    list_models = '1.0_all'  # '0.5_all' models' data
    commum_folder = '/scratch/home/sysadmin/MEGAsync/'

    folder_table = commum_folder + 'XShooter/tables/'
    folder_results = commum_folder + 'XShooter/results/'

# ------------------------------------------------------------------------------
    if read_files is True:
        print(num_spa * '=')
        print('\nReading files...\n')

        # Read data models
        data_models = folder_table + list_models
        models_data = read_txt(table_name=data_models)

        # Read observational data
        print(num_spa * '-')
        print('\nCreating list of files...\n')

        create_list_files(list_name='targets', folder=folder_table + 'stars/',
                          folder_table=folder_table)
        list_stars = read_list_files_all(table_name='targets.txt',
                                         folder_table=folder_table)

        # Saving observational data
        print(num_spa * '-')
        print('\nSaving observational data...\n')

        data = create_class_object(list_stars)


# ------------------------------------------------------------------------------
    if interpol is True:
        D = models_data[:, 0]
        lamb0 = models_data[:, 1]
        Teff = models_data[:, 2]
        logg = models_data[:, 3]
        Tpol = models_data[:, 4]
        Mass = models_data[:, 5]

        fig, ax = plt.subplots()
        fig2, ay = plt.subplots()
        fig3, aw = plt.subplots()
        fig4, az = plt.subplots()

# ------------------------------------------------------------------------------
        # grid teff
        grid_x, grid_y = \
            np.mgrid[np.min(D):np.max(D):(np.max(D) - np.min(D)) / resolution,
                     np.min(lamb0):np.max(lamb0):(np.max(lamb0) -
                     np.min(lamb0)) / resolution]

        aspect = (np.max(D) - np.min(D)) / (np.max(lamb0) - np.min(lamb0))

        grid_z1 = griddata(models_data[:, 0:2], Teff,
                           (grid_x, grid_y), method=interpol_method)
        grid_z1 = rotate(matrix=grid_z1, degree=-90)

        cax = ax.imshow(grid_z1, cmap=cmap, extent=[grid_x[0, 0],
                        grid_x[-1, -1], grid_y[0, 0], grid_y[-1, -1]],
                        aspect=aspect)
        fig.colorbar(cax)

        X = np.arange(np.min(D), np.max(D), (np.max(D) -
                      np.min(D)) / resolution)
        Y = np.arange(np.min(lamb0), np.max(lamb0),
                      (np.max(lamb0) - np.min(lamb0)) / resolution)
        X, Y = np.meshgrid(X, Y[::-1])

# ------------------------------------------------------------------------------
        # grid logg
        grid_z2 = griddata(models_data[:, 0:2], logg,
                           (grid_x, grid_y), method=interpol_method)
        grid_z2 = rotate(matrix=grid_z2, degree=-90)

        cay = ay.imshow(grid_z2, cmap=cmap, extent=[grid_x[0, 0],
                        grid_x[-1, -1], grid_y[0, 0], grid_y[-1, -1]],
                        aspect=aspect)
        fig2.colorbar(cay)

# ------------------------------------------------------------------------------
        # grid Tpol
        grid_z3 = griddata(models_data[:, 0:2], Tpol, (grid_x, grid_y),
                           method=interpol_method)
        grid_z3 = rotate(matrix=grid_z3, degree=-90)

        caw = aw.imshow(grid_z3, cmap=cmap, extent=[grid_x[0, 0],
                        grid_x[-1, -1], grid_y[0, 0], grid_y[-1, -1]],
                        aspect=aspect)
        fig3.colorbar(caw)

# ------------------------------------------------------------------------------
        # grid Mass
        grid_z4 = griddata(models_data[:, 0:2], Mass, (grid_x, grid_y),
                           method=interpol_method)
        grid_z4 = rotate(matrix=grid_z4, degree=-90)

        caz = az.imshow(grid_z4, cmap=cmap, extent=[grid_x[0, 0],
                        grid_x[-1, -1], grid_y[0, 0], grid_y[-1, -1]],
                        aspect=aspect)
        fig4.colorbar(caz)

# ------------------------------------------------------------------------------
        print(num_spa * '-')
        print('\nInterpolating and saving the results ...\n')

        for i in range(len(data)):
            results_teff = []
            results_logg = []
            results_Tpol = []
            results_Mass = []
            D_obs = []
            lamb0_obs = []

            ax.cla()
            ay.cla()
            aw.cla()
            az.cla()
            if np.size(data[i].D) != 1:
                for j in range(len(data[i].D)):
                    # grid teff
                    input_obs_data_1 = np.array([data[i].D[j],
                                                data[i].lamb0[j]])
                    interpoled_values_teff = griddata(models_data[:, 0:2],
                                                      models_data[:, 2],
                                                      input_obs_data_1)

                    results_teff.append('%0.0f' % interpoled_values_teff[0])
                    D_obs.append('%0.2f' % data[i].D[j])
                    lamb0_obs.append('%0.2f' % data[i].lamb0[j])

                    ax.plot(data[i].D[j], data[i].lamb0[j], 'o',
                            markeredgecolor='k', markerfacecolor='gray',
                            markeredgewidth=2, alpha=0.5)

                    if plot_labels is True:
                        ax.text(data[i].D[j] + 0.01, data[i].lamb0[j] + 0.01,
                                '%s' % str(interpoled_values_teff[:4]),
                                fontsize=7)

                    # grid logg
                    input_obs_data_2 = np.array([data[i].D[j],
                                                data[i].lamb0[j]])
                    interpoled_values_logg = griddata(models_data[:, 0:2],
                                                      models_data[:, 3],
                                                      input_obs_data_2)
                    results_logg.append('%0.2f' % interpoled_values_logg[0])

                    ay.plot(data[i].D[j], data[i].lamb0[j], 'o',
                            markeredgecolor='k', markerfacecolor='grey',
                            markeredgewidth=2, alpha=0.5)

                    if plot_labels is True:
                        ay.text(data[i].D[j] + 0.01, data[i].lamb0[j] + 0.01,
                                ('%s' % str(interpoled_values_logg[:4])),
                                fontsize=7)

                    # grid Tpol
                    input_obs_data_3 = np.array([data[i].D[j],
                                                data[i].lamb0[j]])
                    interpoled_values_Tpol = griddata(models_data[:, 0:2],
                                                      models_data[:, 4],
                                                      input_obs_data_3)
                    results_Tpol.append('%0.2f' % interpoled_values_Tpol[0])

                    aw.plot(data[i].D[j], data[i].lamb0[j], 'o',
                            markeredgecolor='k', markerfacecolor='grey',
                            markeredgewidth=2, alpha=0.5)

                    if plot_labels is True:
                        aw.text(data[i].D[j] + 0.01, data[i].lamb0[j] + 0.01,
                                ('%s' % str(interpoled_values_Tpol)),
                                fontsize=7)

                    # grid Mass
                    input_obs_data_4 = np.array([data[i].D[j],
                                                data[i].lamb0[j]])
                    interpoled_values_Mass = griddata(models_data[:, 0:2],
                                                      models_data[:, 5],
                                                      input_obs_data_4)
                    results_Mass.append('%0.2f' % interpoled_values_Mass[0])

                    az.plot(data[i].D[j], data[i].lamb0[j], 'o',
                            markeredgecolor='k', markerfacecolor='grey',
                            markeredgewidth=2, alpha=0.5)

                    if plot_labels is True:
                        az.text(data[i].D[j] + 0.01, data[i].lamb0[j] + 0.01,
                                ('%s' % str(interpoled_values_Mass)),
                                fontsize=7)

                cax = ax.imshow(grid_z1, cmap=cmap, extent=[grid_x[0, 0],
                                grid_x[-1, -1], grid_y[0, 0], grid_y[-1, -1]],
                                aspect=aspect)
                cay = ay.imshow(grid_z2, cmap=cmap, extent=[grid_x[0, 0],
                                grid_x[-1, -1], grid_y[0, 0], grid_y[-1, -1]],
                                aspect=aspect)
                caw = aw.imshow(grid_z3, cmap=cmap, extent=[grid_x[0, 0],
                                grid_x[-1, -1], grid_y[0, 0], grid_y[-1, -1]],
                                aspect=aspect)
                caz = az.imshow(grid_z4, cmap=cmap, extent=[grid_x[0, 0],
                                grid_x[-1, -1], grid_y[0, 0], grid_y[-1, -1]],
                                aspect=aspect)

                CS = ax.contour(X, Y, grid_z1)
                ax.clabel(CS, fontsize=9, inline=1, fontcolor='black')
                CS = ay.contour(X, Y, grid_z2)
                ay.clabel(CS, fontsize=9, inline=1, fontcolor='black')
                CS = aw.contour(X, Y, grid_z3)
                aw.clabel(CS, fontsize=9, inline=1, fontcolor='black')
                CS = az.contour(X, Y, grid_z4)
                az.clabel(CS, fontsize=9, inline=1, fontcolor='black')

                ax.set_ylabel(r'$\lambda_0$')
                ax.set_xlabel(r'D')
                ax.minorticks_on()
                ay.set_ylabel(r'$\lambda_0$')
                ay.set_xlabel(r'D')
                ay.minorticks_on()
                aw.set_ylabel(r'$\lambda_0$')
                aw.set_xlabel(r'D')
                aw.minorticks_on()
                az.set_ylabel(r'$\lambda_0$')
                az.set_xlabel(r'D')
                az.minorticks_on()

                folder_fig_2 = folder_results + data[i].name[:-4] + '/bcd/'

                create_txt_file(x=D_obs, y=lamb0_obs, w=results_teff,
                                z=results_logg, h=results_Tpol,
                                q=results_Mass,
                                file_name=str(folder_fig_2 + data[i].name))

                print('\nStar: %s' % data[i].name[:-4])
                print('Saved files:\n%s' %
                      folder_fig_2 + data[i].name[:-4] + '_teff.png')
                print(folder_fig_2 + data[i].name[:-4] + '_logg.png')

                fig.tight_layout()
                fig.savefig(folder_fig_2 + data[i].name + '_teff.png')
                fig2.tight_layout()
                fig2.savefig(folder_fig_2 + data[i].name + '_logg.png')
                fig3.tight_layout()
                fig3.savefig(folder_fig_2 + data[i].name + '_Tpol.png')
                fig4.tight_layout()
                fig4.savefig(folder_fig_2 + data[i].name + '_Mass.png')

            else:
                    # grid teff
                    input_obs_data_1 = np.array([data[i].D, data[i].lamb0])
                    interpoled_values_teff = griddata(models_data[:, 0:2],
                                                      models_data[:, 2],
                                                      input_obs_data_1)

                    results_teff.append('%0.0f' % interpoled_values_teff)
                    D_obs.append('%0.2f' % data[i].D)
                    lamb0_obs.append('%0.2f' % data[i].lamb0)

                    ax.plot(data[i].D, data[i].lamb0, 'o', markeredgecolor='k',
                            markerfacecolor='gray', markeredgewidth=2,
                            alpha=0.5)

                    if plot_labels is True:
                        ax.text(data[i].D + 0.01, data[i].lamb0 + 0.01, '%s'
                                % str(interpoled_values_teff[:4]), fontsize=7)

                    # grid logg
                    input_obs_data_2 = np.array([data[i].D, data[i].lamb0])
                    interpoled_values_logg = griddata(models_data[:, 0:2],
                                                      models_data[:, 3],
                                                      input_obs_data_2)
                    results_logg.append('%0.2f' % interpoled_values_logg)

                    ay.plot(data[i].D, data[i].lamb0, 'o', markeredgecolor='k',
                            markerfacecolor='gray', markeredgewidth=2,
                            alpha=0.5)

                    if plot_labels is True:
                        ay.text(data[i].D + 0.01, data[i].lamb0 + 0.01,
                                ('%s' % str(interpoled_values_logg[:4])),
                                fontsize=7)

                    # grid Tpol
                    input_obs_data_3 = np.array([data[i].D, data[i].lamb0])
                    interpoled_values_Tpol = griddata(models_data[:, 0:2],
                                                      models_data[:, 4],
                                                      input_obs_data_3)
                    results_Tpol.append('%0.2f' % interpoled_values_Tpol)

                    aw.plot(data[i].D, data[i].lamb0, 'o', markeredgecolor='k',
                            markerfacecolor='grey', markeredgewidth=2,
                            alpha=0.5)

                    if plot_labels is True:
                        aw.text(data[i].D + 0.01, data[i].lamb0 + 0.01,
                                ('%s' % str(interpoled_values_Tpol)),
                                fontsize=7)

                    # grid Mass
                    input_obs_data_4 = np.array([data[i].D, data[i].lamb0])
                    interpoled_values_Mass = griddata(models_data[:, 0:2],
                                                      models_data[:, 3],
                                                      input_obs_data_4)
                    results_logg.append('%0.2f' % interpoled_values_Mass)

                    az.plot(data[i].D, data[i].lamb0, 'o', markeredgecolor='k',
                            markerfacecolor='grey', markeredgewidth=2,
                            alpha=0.5)

                    if plot_labels is True:
                        az.text(data[i].D + 0.01, data[i].lamb0 + 0.01, ('%s'
                                % str(interpoled_values_Mass[:4])), fontsize=7)

                    cax = ax.imshow(grid_z1, cmap=cmap, extent=[grid_x[0, 0],
                                    grid_x[-1, -1], grid_y[0, 0],
                                    grid_y[-1, -1]], aspect=aspect)
                    cay = ay.imshow(grid_z2, cmap=cmap, extent=[grid_x[0, 0],
                                    grid_x[-1, -1], grid_y[0, 0],
                                    grid_y[-1, -1]], aspect=aspect)
                    caw = aw.imshow(grid_z3, cmap=cmap, extent=[grid_x[0, 0],
                                    grid_x[-1, -1], grid_y[0, 0],
                                    grid_y[-1, -1]], aspect=aspect)
                    caz = az.imshow(grid_z4, cmap=cmap, extent=[grid_x[0, 0],
                                    grid_x[-1, -1], grid_y[0, 0],
                                    grid_y[-1, -1]], aspect=aspect)

                    CS = ax.contour(X, Y, grid_z1)
                    ax.clabel(CS, fontsize=9, inline=1, fontcolor='black')
                    CS = ay.contour(X, Y, grid_z2)
                    ay.clabel(CS, fontsize=9, inline=1, fontcolor='black')
                    CS = aw.contour(X, Y, grid_z3)
                    aw.clabel(CS, fontsize=9, inline=1, fontcolor='black')
                    CS = az.contour(X, Y, grid_z4)
                    az.clabel(CS, fontsize=9, inline=1, fontcolor='black')

                    ax.set_ylabel(r'$\lambda_0$')
                    ax.set_xlabel(r'D')
                    ax.minorticks_on()
                    ay.set_ylabel(r'$\lambda_0$')
                    ay.set_xlabel(r'D')
                    ay.minorticks_on()
                    aw.set_ylabel(r'$\lambda_0$')
                    aw.set_xlabel(r'D')
                    aw.minorticks_on()
                    az.set_ylabel(r'$\lambda_0$')
                    az.set_xlabel(r'D')
                    az.minorticks_on()

                    folder_fig_2 = folder_results + data[i].name[:-4] + '/bcd/'

                    create_txt_file(x=D_obs, y=lamb0_obs, w=results_teff,
                                    z=results_logg, h=results_Tpol,
                                    q=results_Mass,
                                    file_name=str(folder_fig_2 + data[i].name))

                    print('\nStar: %s' % data[i].name[:-4])
                    print('Saved files:\n%s:' %
                          folder_fig_2 + data[i].name[:-4] + '_teff.png')
                    print(folder_fig_2 + data[i].name[:-4] + '_logg.png')

                    plt.tight_layout()
                    fig.tight_layout()
                    fig.savefig(folder_fig_2 + data[i].name + '_teff.png')
                    fig2.tight_layout()
                    fig2.savefig(folder_fig_2 + data[i].name + '_logg.png')
                    fig3.tight_layout()
                    fig3.savefig(folder_fig_2 + data[i].name + '_Tpol.png')
                    fig4.tight_layout()
                    fig4.savefig(folder_fig_2 + data[i].name + '_Mass.png')

        print(num_spa * '-')
        print('\nThe End ...\n')
        print(num_spa * '-')

# ------------------------------------------------------------------------------
    if plot3d is True:
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        ax.plot_surface(grid_x, grid_y, grid_z1, rstride=8,
                        cstride=8, alpha=0.3)
        ax.contourf(grid_x, grid_y, grid_z1, zdir='z',
                    offset=-100, cmap=cm.coolwarm)
        ax.contourf(grid_x, grid_y, grid_z1, zdir='z',
                    cmap=cm.coolwarm)
        ax.contourf(grid_x, grid_y, grid_z1, zdir='z',
                    offset=-4, cmap=cm.coolwarm)
        ax.contourf(grid_x, grid_y, grid_z1, zdir='z',
                    offset=-40, cmap=cm.coolwarm)
        ax.set_zlim(-100, 100)
        plt.clf()

# ------------------------------------------------------------------------------

    return

# ==============================================================================
if __name__ == '__main__':
    main()
