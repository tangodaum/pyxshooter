# ==============================================================================
# !/usr/bin/env python
# -*- coding:utf-8 -*-

# Created by B. Mota 2016-02-16 to present...

# import packages
# from pyraf import iraf

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import gzip
import re
import matplotlib.font_manager as fm
import os
import math
from astropy import units as u
import csv
import sys
import pyfits
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages/')
import matplotlib
from astroquery.simbad import Simbad
import aplpy
import lineid_plot
import pyhdust.phc as phc
import pyhdust.spectools as spt
import pyhdust.bcd as bcd
from pyhdust.interftools import imshowl
import pyraf.iraf as iraf
# import pyraf
mpl.rcParams.update({'font.size': 18})
mpl.rcParams['lines.linewidth'] = 2
font = fm.FontProperties(size=17)
mpl.rc('xtick', labelsize=17)
mpl.rc('ytick', labelsize=17)
fontsize_label = 18  # 'x-large'
num_spa = 74  # 151 #176 #176

# from telfit import TelluricFitter, DataStructures

__version__ = "0.0.1"
__author__ = "Bruno Mota, Daniel Moser, Ahmed Shoky"

# ==============================================================================
# Define some constants

light_speed = phc.c.SI  # m/s


# ==============================================================================
class data_object:
    '''
    Class of the stellar objects. Using this class, we can store
    for each star a sort of variables, which can be easily accessed.

    '''

    kind = 'star'           # class variable shared by all instances

    def __init__(self, name, arm, mjd, list_files, name_ffit,
                 wave=None, flux=None, sigm=None, wave_vis=None,
                 flux_vis=None, sigm_vis=None, wave_nir=None,
                 flux_nir=None, sigm_nir=None):
        # instance variable unique to each instance
        self.name = name
        self.wave = wave
        self.flux = flux
        self.sigm = sigm
        self.wave_vis = wave_vis
        self.flux_vis = flux_vis
        self.sigm_vis = sigm_vis
        self.wave_nir = wave_nir
        self.flux_nir = flux_nir
        self.sigm_nir = sigm_nir
        self.arm = arm
        self.mjd = mjd
        self.list_files = list_files
        self.name_ffit = name_ffit


# ==============================================================================
def ger_tbdata(file_name):
    '''
    Creates tbdata from hdulist and check if there is fields.

    :param file_name: fits file (string)
    :return: tbdata
    '''

    hdulist = pyfits.open(file_name)
    tbdata = hdulist[1].data

    return tbdata


# ==============================================================================
def read_zipfile(folder, zip_file):
    '''
    Read the content of a zip file.

    :param folder: folder with the files (string)
    :param zip_file: name of the file
    :return: file content
    '''

    zip_file = folder + zip_file

    with gzip.open(zip_file, 'rb') as f:
        file_content = f.read()
    return file_content


# ==============================================================================
def unzip_file(folder, folder_table, zip_file=None, list=None):
    '''
    This routine unzip a file or a list of files in a given folder.

    :param folder: file's folder (string)
    :param zip_file: file's names (string)
    :param list: to unzipped more than one file put list=True, otherwise False
    :return: unzipped file

    Example:

        folder = '/run/media/sysadmin/SAMSUNG/reduzindo/runC/'
        folder_table = '/run/media/sysadmin/SAMSUNG/reduzindo/'
        unzip_file(folder, folder_table, list=True)

    '''

    if list is True:
        create_list_files('list_zip', folder, folder_table)
        list_files = read_list_files_all(table_name='list_zip.txt',
                                         folder_table=folder_table)
        os.chdir(folder)

        for i in range(len(list_files)):
            if list_files[i][-2:] == '.Z':
                os.system("patool extract " + list_files[i])

    else:
        zip_file = folder + zip_file
        os.system("patool extract " + zip_file)

    return


# ==============================================================================
def create_list_files(list_name, folder, folder_table):
    '''
    Creates a list of the files inside a given folder.

    :param list_name: list's name (string)
    :param folder: files' folder (string)
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

    :param folder_table: table's folder (string)
    :param table_name: Table's name (string)
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
def read_list_files_star(table_name, folder_table):
    '''
    Read list of files listed in a table, and returns the star's
    fits names in an array.

    :param table_name: Table's name (string)
    :return: list of files (txt file)
    '''

    file_data = folder_table + table_name

    files = open(file_data).read()
    rule = r'/scratch.*HD.*'
    regex = re.compile(rule)
    matches = [m.group(0) for m in regex.finditer(files)]
    list_files = matches[:]

    for i in range(len(list_files)):
        identify_object(list_files[i])

    return list_files


# ==============================================================================
def read_norm_fits(file_name):
    '''
    Read XShooter's normalized fits files.

    :param file_name: name of the fits file (string)
    :return: normalized flux (array)
    '''

    hdulist = pyfits.open(file_name)
    flux_norm = hdulist[0].data

    return flux_norm


# ==============================================================================
def read_header_aquisition(file_name):
    '''
    Read header of the aquisition images.

    :param file_name: name of the fits file (string)
    :return: object name (string), MJD (float), xshooter arm (string)
    '''

    # Openning the fits file
    hdulist = pyfits.open(file_name)
    header = hdulist[0].header

    try:
        obj = header['OBJECT']
        mjd = header['MJD-OBS']
        arm = header['HIERARCH ESO SEQ ARM']

        return obj, mjd, arm
    except:
        try:
            obj = header['HIERARCH ESO OBS TARG NAME']
            mjd = header['MJD-OBS']
            arm = header['HIERARCH ESO SEQ ARM']

            return obj, mjd, arm
        except:
            pass

            return False, False, False


# ==============================================================================
def read_header_simple(file_name):
    '''
    Read header of a simple fits file.

    :param file_name: name of the fits file (string)
    :return: header_1 (string), header_2 (string)

    Note:
        Somethimes, some fits file exhibit two headers.
    '''

    # Openning the fits file
    hdulist = pyfits.open(file_name)
    header = hdulist[0].header

    # This function shows the type of file
    header_2 = hdulist[1].header

    return header, header_2


# ==============================================================================
def read_fits_xshooter(file_name, print_obj):
    '''
    Read XShooter's fits files.

    :param file_name: name of the file in fit extension (string)
    :return: parameters (obj, obs_date, mjd, arm, wave, flux,
    sigma, qual, snr, flux_red, sigm_red)
    '''

    hdulist = pyfits.open(file_name)
    header = hdulist[0].header
    tbdata = hdulist[1].data
    obj = header['OBJECT']
    mjd = header['MJD-OBS']
    obs_date = header['DATE-OBS']
    arm = header['ESO SEQ ARM']

    if print_obj is True:
        print("OBJECT: %s " % obj)

    header_2 = hdulist[1]
    xtension = header_2.header['XTENSION']

    if hasattr(tbdata, 'field') is True and \
       xtension != 'IMAGE' and ('WAVE' in tbdata.names) is True:
        wave = tbdata.field('WAVE')
        flux = tbdata.field('FLUX')
        sigma = tbdata.field('ERR')
        qual = tbdata.field('QUAL')
        snr = tbdata.field('SNR')
        boolop = True

        return boolop, obj, obs_date, mjd, arm, wave, flux,\
            sigma, qual, snr

    else:
        boolop = False

        return boolop


# ==============================================================================
def read_fits_star_name(file_name, typ):
    '''
    Read XShooter's fits files.

    :param file_name: name of the file in fit extension (string)
    :param typ: define if it will be used to read a image (typ='img')or a
    bintable (typ='data').
    :return: parameters (object if typ='data', file_name if typ='img' )
    '''

    if file_name[-3:] != 'txt' and file_name[-2:] != '.Z' \
       and file_name[-3:] != 'xml' and file_name[-2:] != 'db' \
       and file_name[-3:] != 'log':

        string = str(file_name)
        rule = '(?!.*\/)(.*?)(?=\.fits)'
        match = re.search(rule, string)
        name_fits = match.group() + '.fits'

        if name_fits[0:2] != 'M.':

            hdulist = pyfits.open(file_name)
            header = hdulist[0].header

            if 'OBJECT' in header:

                obj = header['OBJECT']

                if typ == 'data':
                    return obj
                if typ == 'img':
                    return file_name


# ==============================================================================
def create_list_stars(list_files):
    '''
    Create list of the observed stars.

    :param list_files: text list with the files' paths
    :return: list of stars (array)
    '''

    stars = []
    for i in range(len(list_files)):
        if list_files[i][-6:] != 'README':
            star = read_fits_star_name(file_name=list_files[i], typ='data')
            stars.append(star)

    cond = np.unique(stars)
    if len(cond) != 1:
        list_stars = []
        for i in range(len(stars)):
            if stars[i] != stars[i - 1]:
                list_stars.append(stars[i])

        new_list_stars = []
        for i in range(len(list_stars)):
            if list_stars[i] != 'STD,TELLURIC':
                new_list_stars.append(list_stars[i])

        list_stars = np.unique(new_list_stars)

    else:

        list_stars = np.copy(cond[0])

    return list_stars


# ==============================================================================
def create_list_stars_img(list_files):

    '''
    Create list of the observed stars.

    :param list_files: text list with the files' paths
    :return: list of stars (array)
    '''

    stars = []
    for i in range(len(list_files)):
        if list_files[i][-6:] != 'README':

            star = read_fits_star_name(file_name=list_files[i], typ='img')
            stars.append(star)

    list_stars = []
    for i in range(len(stars)):
        if stars[i] != stars[i - 1]:
            list_stars.append(stars[i])

    new_list_stars = []
    for i in range(len(list_stars)):
        if list_stars[i] != 'STD,TELLURIC':
            new_list_stars.append(list_stars[i])

    list_stars = np.unique(new_list_stars)

    return list_stars


# ==============================================================================
def create_class_object(list_files, obj):
    '''
    Create object class for all targets in a list.

    :param list_files: list of XShooter fits files (array)
    :return: data (class object)
    '''

    data = []

    for i in range(len(list_files)):
        if list_files[i][-6:] != 'README':
            star = read_fits_star_name(list_files[i], typ='data')

            if star == obj:
                params = read_fits_xshooter(list_files[i], print_obj=False)

                if params is not False:
                    boolop = params[0]
                else:
                    boolop = params

                if boolop is True:

                    boolop, obj, obs_date, mjd, arm, wave, flux, \
                        sigma, qual, snr = \
                        read_fits_xshooter(list_files[i], print_obj=False)
                    wave = 10. * wave[:]  # Convert to angstrom
                    wave = np.array(wave)

                    # Plotting UBV only
                    # Rewrite the original fits
                    string = str(list_files[i][:])
                    rule = '(?!.*\/)(.*?)(?=\.fits)'
                    match = re.search(rule, string)
                    name_ffit = match.group()

                    data.append(data_object(name=obj, wave=wave, flux=flux,
                                            sigm=sigma, arm=arm, mjd=mjd,
                                            list_files=list_files[i],
                                            name_ffit=name_ffit))

    return data


# ==============================================================================
def plot_sed(data, folder_fig, folder_temp, star_name):

    '''
    Plot the SED for each star inside the data structure.

    :param data: list of XShooter fits files (array)
    :param folder_fig: folder where the figures will be saved (string)
    :param folder_temp: folder of the temporary files (string)
    :star_name: star name (string)
    :return: data (class object)
    '''

    global conc_wave, conc_flux, conc_sigm, conc_mjd, conc_arm,\
        conc_flux_norm, conc_name_ffit

# ------------------------------------------------------------------------------
    print(num_spa * '=')
    print('\nPlotting SEDs\n')
    print("OBJECT: %s \n" % star_name)
    print('Normalizing spectra...\n')

    conc_wave, conc_flux, conc_sigm, conc_mjd, conc_arm, conc_flux_norm,\
        conc_name_ffit = [], [], [], [], [], [], []

    for i in range(len(data)):
        star = data[i].name

        if star_name == star:

            wave = data[i].wave[0]
            flux = data[i].flux[0]
            sigm = data[i].sigm[0]
            mjd = data[i].mjd
            arm = data[i].arm
            list_files = data[i].list_files
            name_ffit = data[i].name_ffit

            flux_smoothed = smooth_spectrum(wave, flux)

            new_norm_fits, flux_norm = \
                normalize_spectrum(wave, flux_smoothed, input_file=list_files,
                                   output_folder=folder_temp, arm=arm)

            conc_flux_norm.append(flux_norm)
            conc_wave.append(wave)
            conc_flux.append(flux)
            conc_sigm.append(sigm)
            conc_mjd.append(mjd)
            conc_arm.append(arm)
            conc_name_ffit.append(name_ffit)

# ------------------------------------------------------------------------------
    if len(conc_wave) != 0:

        plt.figure()
        if len(conc_arm) != 1:
            cor = phc.gradColor(np.arange(len(conc_arm)), cmapn='rainbow')
        else:
            cor = 'black'

        for i in range(len(conc_arm)):
            plt.errorbar(x=conc_wave[i], y=conc_flux[i], yerr=conc_sigm[i],
                         fmt='.', label="{:.2f} , {}".format(conc_mjd[i],
                         conc_name_ffit[i]), color=cor[i])

        plt.legend(loc='best', fontsize=8)
        plt.xlabel(r'$\lambda$ $[\AA ]$')
        plt.ylabel(r'$F_{\lambda}$ $\rm [erg/cm^2/s/\AA]$')
        plt.yscale('log')

        plt.autoscale()
        plt.minorticks_on()
        plt.xlim(1000., 30000.)

        if os.path.isdir(folder_fig + star_name + '/sed/') is False:
            os.mkdir(folder_fig + star_name + '/sed/')
        folder_fig_2 = folder_fig + star_name + '/sed/'

        name_fig = folder_fig_2 + star_name + '.png'

        plt.tight_layout()
        plt.savefig(name_fig)
        plt.clf()
        plt.close()

# ------------------------------------------------------------------------------

        ak = lineid_plot.initial_annotate_kwargs()
        ak['arrowprops']['arrowstyle'] = "-"
        pk = lineid_plot.initial_plot_kwargs()
        pk['color'] = "red"

        fig = plt.figure()
        baxsp = 0.08
        fontsize = 4

        if len(conc_arm) != 1:
            cor = phc.gradColor(np.arange(len(conc_arm)), cmapn='cool')

            avg_conc_flux_arr_uvb, flux_smoth_arr_uvb, cont_uvb = [], [], []
            avg_conc_flux_arr_vis, flux_smoth_arr_vis, cont_vis = [], [], []
            avg_conc_flux_arr_nir, flux_smoth_arr_nir, cont_nir = [], [], []
            for i in range(len(conc_arm)):
                if conc_arm[i] == 'UVB':
                    flux_smoth = smooth_spectrum(conc_wave[i], conc_flux[i],
                                                 doplot=False)
                    flux_smoth = np.log10(flux_smoth)
                    conc_wave_uvb = np.copy(conc_wave[i])
                    avg_conc_flux = np.median(flux_smoth)
                    nans, x = phc.nan_helper(avg_conc_flux)
                    avg_conc_flux = avg_conc_flux[~nans]
                    avg_conc_flux_arr_uvb.append(avg_conc_flux)
                    flux_smoth_arr_uvb.append(flux_smoth)
                    cont_uvb.append(np.size(i))
                    ax = plt.subplot(311)
                    ax.plot(conc_wave[i], flux_smoth, '-', color=cor[i])

                if conc_arm[i] == 'VIS':
                    flux_smoth = smooth_spectrum(conc_wave[i], conc_flux[i],
                                                 doplot=False)
                    flux_smoth = np.log10(flux_smoth)
                    conc_wave_vis = np.copy(conc_wave[i])
                    avg_conc_flux = np.median(flux_smoth)
                    nans, x = phc.nan_helper(avg_conc_flux)
                    avg_conc_flux = avg_conc_flux[~nans]
                    avg_conc_flux_arr_vis.append(avg_conc_flux)
                    flux_smoth_arr_vis.append(flux_smoth)
                    cont_vis.append(np.size(i))
                    ax1 = plt.subplot(312)
                    ax1.plot(conc_wave[i], flux_smoth, '-', color=cor[i])

                if conc_arm[i] == 'NIR':
                    flux_smoth = smooth_spectrum(conc_wave[i], conc_flux[i],
                                                 doplot=False)
                    flux_smoth = np.log10(flux_smoth)
                    conc_wave_nir = np.copy(conc_wave[i])
                    avg_conc_flux = np.median(flux_smoth)
                    nans, x = phc.nan_helper(avg_conc_flux)
                    avg_conc_flux = avg_conc_flux[~nans]
                    avg_conc_flux_arr_nir.append(avg_conc_flux)
                    flux_smoth_arr_nir.append(flux_smoth)
                    cont_nir.append(np.size(i))
                    ax2 = plt.subplot(313)
                    ax2.plot(conc_wave[i], flux_smoth, '-', color=cor[i])

# Plotting line IDs:
            # UVB
            if len(avg_conc_flux_arr_uvb) != 0:
                if np.size(cont_uvb) == 1:
                    indx_max = 0
                    cont_uvb = 0
                else:
                    indx_max = avg_conc_flux_arr_uvb.\
                        index(np.max(avg_conc_flux_arr_uvb))
                    cont_uvb = cont_uvb[indx_max]

                ax = plt.subplot(311)
                lineid_plot.plot_line_ids(conc_wave_uvb,
                                          flux_smoth_arr_uvb[cont_uvb],
                                          line_wave_uvb,
                                          line_label_uvb,
                                          ax=ax,
                                          annotate_kwargs=ak,
                                          plot_kwargs=pk,
                                          box_axes_space=1.1 * baxsp,
                                          fontsize_label=fontsize)

            # VIS
            if len(avg_conc_flux_arr_vis) != 0:
                if np.size(cont_vis) == 1:
                    indx_max = 0
                    cont_vis = 0
                else:
                    indx_max = avg_conc_flux_arr_vis.\
                        index(np.max(avg_conc_flux_arr_vis))
                    cont_vis = cont_vis[indx_max]

                ax1 = plt.subplot(312)
                lineid_plot.plot_line_ids(conc_wave_vis,
                                          flux_smoth_arr_vis[cont_vis],
                                          line_wave_vis,
                                          line_label_vis,
                                          ax=ax1,
                                          annotate_kwargs=ak,
                                          plot_kwargs=pk,
                                          box_axes_space=baxsp,
                                          fontsize_label=fontsize)

            # NIR
            if len(avg_conc_flux_arr_nir) != 0:
                if np.size(cont_nir) == 1:
                    indx_max = 0
                    cont_nir = 0
                else:
                    indx_max = avg_conc_flux_arr_nir.\
                        index(np.max(avg_conc_flux_arr_nir))
                    cont_nir = cont_nir[indx_max]

                ax2 = plt.subplot(313)
                lineid_plot.plot_line_ids(conc_wave_nir,
                                          flux_smoth_arr_nir[cont_nir],
                                          line_wave_nir,
                                          line_label_nir,
                                          ax=ax2,
                                          annotate_kwargs=ak,
                                          plot_kwargs=pk,
                                          box_axes_space=1.3 * baxsp,
                                          fontsize_label=fontsize)

        else:
            if len(conc_arm) == 1:
                cor = 'black'
                indexes = np.where(conc_flux[0] > 0)
                conc_wave = np.copy(conc_wave[0])
                conc_wave = conc_wave[indexes]
                conc_flux = np.copy(conc_flux[0])
                conc_flux = conc_flux[indexes]

                flux_smoth = smooth_spectrum(conc_wave, conc_flux,
                                             doplot=False)
                flux_smoth = np.log10(flux_smoth)

                if conc_arm[0] == 'UVB':
                    ax = plt.subplot(311)
                    ax.plot(conc_wave, flux_smoth, '-', color=cor)
                    lineid_plot.plot_line_ids(conc_wave, flux_smoth,
                                              line_wave_uvb, line_label_uvb,
                                              ax=ax, annotate_kwargs=ak,
                                              plot_kwargs=pk,
                                              box_axes_space=baxsp,
                                              fontsize_label=fontsize)
                if conc_arm[0] == 'VIS':
                    ax1 = plt.subplot(312)
                    ax1.plot(conc_wave, flux_smoth, '-', color=cor)
                    lineid_plot.plot_line_ids(conc_wave, flux_smoth,
                                              line_wave_vis,
                                              line_label_vis, ax=ax1,
                                              annotate_kwargs=ak,
                                              plot_kwargs=pk,
                                              box_axes_space=baxsp,
                                              fontsize_label=fontsize)
                if conc_arm[0] == 'NIR':
                    ax2 = plt.subplot(313)
                    ax2.plot(conc_wave, flux_smoth, '-', color=cor)
                    lineid_plot.plot_line_ids(conc_wave, flux_smoth,
                                              line_wave_nir,
                                              line_label_nir, ax=ax2,
                                              annotate_kwargs=ak,
                                              plot_kwargs=pk,
                                              box_axes_space=baxsp,
                                              fontsize_label=fontsize)

        ax = plt.subplot(311)
        ax.set_xlim(3000., 5600.)
        # ax.set_ylim(1e-15, 1e-9)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.locator_params(nbins=5, axis='y')

        ax1 = plt.subplot(312)
        ax1.set_xlim(5200., 10200.)
        # ax1.set_ylim(1e-15, 1e-9)
        ax1.minorticks_on()
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.locator_params(nbins=5, axis='y')

        ax2 = plt.subplot(313)
        ax2.set_xlim(9700., 25000.)
        # ax2.set_ylim(1e-15, 1e-9)
        ax2.minorticks_on()
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.locator_params(nbins=5, axis='y')

        ax1.set_ylabel(r'$\log F$  $[\rm erg/cm^2/s/$ $\AA]$')
        ax2.set_xlabel(r'$\lambda$ $[\AA ]$')

        fig.subplots_adjust(hspace=0.0001)

        name_fig = folder_fig_2 + star_name + '_smoothed' + '.png'
        fig.tight_layout()
        plt.locator_params(nlins=5, axis='y')
        plt.savefig(name_fig)

        plt.clf()
        plt.close()

    return


# ==============================================================================
def plot_arm_lines(data, folder_fig, folder_temp, delta, star_name):
    '''
    Plot the lines for each star inside the data structure.

    :param data: list of XShooter fits files (array)
    :param folder_fig: folder where the figures will be saved (string)
    :param delta: plot parameter (float)
    :param folder_temp: folder of the temporary files (string)
    :star_name: star name (string)
    :return: figures and tables
    '''

    global conc_wave, conc_flux, conc_sigm, conc_mjd, conc_arm,\
        conc_flux_norm, conc_name_ffit, folder_fig_2

# ------------------------------------------------------------------------------
    print(num_spa * '=')
    print('\nPlotting Line Profiles and calculating BCD parameters\n')
    print("OBJECT: %s \n" % star_name)

    conc_wave, conc_flux, conc_sigm, conc_mjd, conc_arm, conc_flux_norm,\
        conc_name_ffit, D_obs, lamb0_obs = [], [], [], [], [], [], [], [], []

    for i in range(len(data)):
        star = data[i].name

        if star_name == star:

            wave = data[i].wave[0]
            flux = data[i].flux[0]
            sigm = data[i].sigm[0]
            mjd = data[i].mjd
            arm = data[i].arm

            list_files = data[i].list_files
            name_ffit = data[i].name_ffit

            # Normalizing the spectrum
            new_norm_fits, flux_norm = \
                normalize_spectrum(wave, flux, input_file=list_files,
                                   output_folder=folder_temp, arm=arm)

            conc_flux_norm.append(flux_norm)
            conc_wave.append(wave)
            conc_flux.append(flux)
            conc_sigm.append(sigm)
            conc_mjd.append(mjd)
            conc_arm.append(arm)
            conc_name_ffit.append(name_ffit)

    if os.path.isdir(folder_fig + star_name) is False:
        os.mkdir(folder_fig + star_name)
    folder_fig_2 = folder_fig + star_name + '/'

    if os.path.isdir(folder_fig_2 + '/bcd/') is False:
        os.mkdir(folder_fig_2 + '/bcd/')

# ------------------------------------------------------------------------------
    # BCD Analysis
    for z in range(len(conc_arm)):

        if conc_arm[z] == 'UVB':

            D, lambda_0 = \
                bcd_analysis(obj=star_name, wave=conc_wave[z],
                             flux=conc_flux[z],
                             flux_norm=conc_flux_norm[z],
                             folder_fig=folder_fig_2 + '/bcd/')

            D_obs.append('%0.2f' % D)
            lamb0_obs.append('%0.2f' % lambda_0)

    create_txt_file(x=D_obs, y=lamb0_obs,
                    file_name=folder_bcd + star_name + '.txt')

# ------------------------------------------------------------------------------
    # Plotting lines
    delta_lambda_arr = []
    print('\n')
    print(num_spa * '-')
    print('\nPlotting lines by arm')

    print('\nARM: UVB')
    for v in range(len(line_label_uvb_2)):
        fig, ax = plt.subplots()
        fig2, ay = plt.subplots()
        fig3, az = plt.subplots()
        ew = []
        cont = 0

        for z in range(len(conc_arm)):
            if conc_arm[z] == 'UVB':

                central_wave = 4861.  # AA hbeta

                delta_lambda = \
                    calc_dlamb(wave=conc_wave[z], flux=conc_flux_norm[z],
                               center_wave=central_wave, delta=delta)
                delta_lambda_arr.append(delta_lambda)

                plot_line(wave=conc_wave[z] + delta_lambda,
                          flux=conc_flux_norm[z],
                          center_wave=line_wave_uvb[v], delta=delta,
                          vel=400., label="{}".format(conc_mjd[z]),
                          ax=ax, calc_central_wave=True,
                          save_fig=False, gauss_fit=False)

                # Plotting EW
                vel, norm_flux = \
                    fluxXvel(wave=conc_wave[z], flux=conc_flux_norm[z],
                             flux_plus_i=conc_flux_norm[z] + 0.1 * cont,
                             central_wave=central_wave, delta=1000,
                             label=str(conc_mjd[z]), ax=ay)

                val, indx = find_nearest(conc_wave[z], central_wave)

                vel = np.array(vel)
                norm_flux = np.array(norm_flux)

                # Determining EW
                ew.append((spt.EWcalc(vels=vel, flux=norm_flux, vw=1000),
                          conc_mjd[z]))

                cont = cont + 1

        line_label = np.copy(line_label_uvb_2[v])
        center_wave = np.copy(line_wave_uvb[v])

# ------------------------------------------------------------------------------
        # Saving Line profile (Flux norm x lambda)
        ax.minorticks_on()
        ax.legend(loc='best')
        ax.set_title(line_label_uvb[v])
        wave_min = center_wave - delta
        wave_max = center_wave + delta
        ax.set_xlim(wave_min, wave_max)
        ax.set_xlabel(r'$\lambda$ $[\AA]$')
        ax.set_ylabel(r'$F_{\lambda}/F_c$')
        ax.set_ylim(0., 2.)

        if os.path.isdir(folder_fig_2 + 'lines/') is False:
            os.mkdir(folder_fig_2 + 'lines/')

        if os.path.isdir(folder_fig_2 + 'lines/' +
                         str(line_label)) is False:
            os.mkdir(folder_fig_2 + 'lines/' + str(line_label))

        fig_name = folder_fig_2 + 'lines/' + str(line_label) + '/'\
            + star_name + '_' + str(line_label) + '.png'
        fig.tight_layout()
        fig.savefig(fig_name)
        fig.clf()

# ------------------------------------------------------------------------------
        # Saving FLux vs Velocity
        ay.minorticks_on()
        ay.legend(loc='best')
        ay.set_title(line_label_uvb[v])
        ay.set_xlabel(r'$V$ $[km/s]$')
        ay.set_ylabel(r'$F_{\lambda}/F_c$')
        ay.set_ylim(0., 2.)

        fig_name = folder_fig_2 + 'lines/' + str(line_label) + '/' +\
            star_name + '_' + str(line_label) + '_velXflux' + '.png'
        plt.tight_layout()
        fig2.savefig(fig_name)
        fig2.clf()

# ------------------------------------------------------------------------------
        # Plotting EW x time
        for i in range(len(ew)):
            mjd = ew[i][1]
            ewi = ew[i][0]
            if ewi != 0:
                az.plot(mjd, ewi, 'o')

        az.minorticks_on()
        az.legend(loc='best')
        az.set_xlabel(r'MJD-OBS')
        az.set_ylabel(r'EW')

        fig_name = folder_fig_2 + 'lines/' + str(line_label) + '/' +\
            star_name + '_' + str(line_label) + '_ew' + '.png'
        plt.tight_layout()
        fig3.savefig(fig_name)
        fig3.clf()

        plt.close()
        plt.close()
        plt.close()

        if len(delta_lambda_arr) != 0:
            delta_lambda = np.mean(delta_lambda_arr)
        else:
            delta_lambda = 0.

# ------------------------------------------------------------------------------
    print('\nARM: VIS')
    for v in range(len(line_label_vis_2)):
        fig, ax = plt.subplots()
        fig2, ay = plt.subplots()
        fig3, az = plt.subplots()
        ew = []
        cont = 0
        for z in range(len(conc_arm)):
            if conc_arm[z] == 'VIS':

                central_wave = np.copy(line_wave_vis[v])

                delta_lambda = \
                    calc_dlamb(wave=conc_wave[z], flux=conc_flux_norm[z],
                               center_wave=central_wave, delta=delta)

                plot_line(wave=conc_wave[z] + delta_lambda,
                          flux=conc_flux_norm[z] + 0.1 * cont,
                          center_wave=line_wave_vis[v],
                          delta=delta, vel=400.,
                          label="{}".format(conc_mjd[z]),
                          ax=ax, calc_central_wave=True,
                          save_fig=False, gauss_fit=False)

                # Plotting EW
                vel, norm_flux = \
                    fluxXvel(wave=conc_wave[z], flux=conc_flux_norm[z],
                             flux_plus_i=conc_flux_norm[z] + 0.1 * cont,
                             central_wave=central_wave, delta=1000,
                             label=str(conc_mjd[z]), ax=ay)

                val, indx = find_nearest(conc_wave[z], central_wave)

                vel = np.array(vel)
                norm_flux = np.array(norm_flux)

                # Determining EW
                ew.append((spt.EWcalc(vels=vel, flux=norm_flux,
                          vw=1000), conc_mjd[z]))

                cont = cont + 1

        line_label = np.copy(line_label_vis_2[v])
        center_wave = np.copy(line_wave_vis[v])

        if os.path.isdir(folder_fig + star_name) is False:
            os.mkdir(folder_fig + star_name)
        folder_fig_2 = folder_fig + star_name + '/'

        if os.path.isdir(folder_fig_2 + 'lines/') is False:
            os.mkdir(folder_fig_2 + 'lines/')

        if os.path.isdir(folder_fig_2 + 'lines/' + str(line_label)) \
           is False:
            os.mkdir(folder_fig_2 + 'lines/' + str(line_label))

        ax.legend(loc='best')

        ax.minorticks_on()
        ax.set_title(line_label_vis[v])
        wave_min = center_wave - delta
        wave_max = center_wave + delta
        ax.set_xlim(wave_min, wave_max)
        ax.set_ylim(0., 2.)

        fig_name = folder_fig_2 + 'lines/' + str(line_label) +\
            '/' + star_name + '_' + str(line_label) + '.png'
        ax.set_xlabel(r'$\lambda$ $[\AA]$')
        ax.set_ylabel(r'$F_{\lambda}/F_c$')
        fig.tight_layout()
        fig.savefig(fig_name)
        fig.clf()

# ------------------------------------------------------------------------------
        # Saving FLux vs Velocity
        ay.minorticks_on()
        ay.legend(loc='best')
        ay.set_title(line_label_vis[v])
        ay.set_xlabel(r'$V$ $[km/s]$')
        ay.set_ylabel(r'$F_{\lambda}/F_c$')
        ay.set_ylim(0., 2.)

        fig_name = folder_fig_2 + 'lines/' + str(line_label) + '/' +\
            star_name + '_' + str(line_label) + '_velXflux' + '.png'
        plt.tight_layout()
        fig2.savefig(fig_name)
        fig2.clf()

# ------------------------------------------------------------------------------
        # Plotting EW x time
        for i in range(len(ew)):
            mjd = ew[i][1]
            ewi = ew[i][0]
            if ewi != 0:
                az.plot(mjd, ewi, 'o')

        az.minorticks_on()
        az.legend(loc='best')
        az.set_xlabel(r'MJD-OBS')
        az.set_ylabel(r'EW')
        fig_name = folder_fig_2 + 'lines/' + str(line_label) + '/' +\
            star_name + '_' + str(line_label) + '_ew' + '.png'

        plt.tight_layout()
        fig3.savefig(fig_name)
        fig3.clf()

        plt.close()
        plt.close()
        plt.close()

# ------------------------------------------------------------------------------
    print('\nARM: NIR')
    for v in range(len(line_label_nir_2)):
        fig, ax = plt.subplots()
        fig2, ay = plt.subplots()
        fig3, az = plt.subplots()
        ew = []
        cont = 0
        for z in range(len(conc_arm)):
            if conc_arm[z] == 'NIR':

                central_wave = np.copy(line_wave_nir[v])

                delta_lambda = calc_dlamb(wave=conc_wave[z],
                                          flux=conc_flux_norm[z],
                                          center_wave=central_wave,
                                          delta=delta)

                plot_line(wave=conc_wave[z] + delta_lambda,
                          flux=conc_flux_norm[z] + 0.1 * cont,
                          center_wave=line_wave_nir[v],
                          delta=delta, vel=400.,
                          label="{}".format(conc_mjd[z]),
                          ax=ax, calc_central_wave=True,
                          save_fig=False, gauss_fit=False)

                # Plotting EW
                vel, norm_flux = \
                    fluxXvel(wave=conc_wave[z], flux=conc_flux_norm[z],
                             flux_plus_i=conc_flux_norm[z] + 0.1 * z,
                             central_wave=central_wave, delta=1000,
                             label=str(conc_mjd[z]), ax=ay)

                val, indx = find_nearest(conc_wave[z], central_wave)

                vel = np.array(vel)
                norm_flux = np.array(norm_flux)

                # Determining EW
                ew.append((spt.EWcalc(vels=vel, flux=norm_flux, vw=1000),
                          conc_mjd[z]))

                cont = cont + 1

        line_label = np.copy(line_label_nir_2[v])
        center_wave = np.copy(line_wave_nir[v])

        if os.path.isdir(folder_fig + star_name) is False:
            os.mkdir(folder_fig + star_name)
        folder_fig_2 = folder_fig + star_name + '/'

        if os.path.isdir(folder_fig_2 + 'lines/') is False:
            os.mkdir(folder_fig_2 + 'lines/')

        if os.path.isdir(folder_fig_2 + 'lines/' + str(line_label)) \
           is False:
            os.mkdir(folder_fig_2 + 'lines/' + str(line_label))

        ax.legend(loc='best')
        ax.minorticks_on()
        ax.set_title(line_label_nir[v])
        wave_min = center_wave - delta
        wave_max = center_wave + delta
        ax.set_xlim(wave_min, wave_max)
        ax.set_ylim(0., 2.)

        fig_name = folder_fig_2 + 'lines/' + str(line_label) +\
            '/' + star_name + '_' + str(line_label) + '.png'
        ax.set_xlabel(r'$\lambda$ $[\AA]$')
        ax.set_ylabel(r'$F_{\lambda}/F_c$')
        fig.tight_layout()
        fig.savefig(fig_name)
        fig.clf()

# ------------------------------------------------------------------------------
        # Saving FLux vs Velocity
        ay.minorticks_on()
        ay.legend(loc='best')
        ay.set_title(line_label_nir[v])
        ay.set_xlabel(r'$V$ $[km/s]$')
        ay.set_ylabel(r'$F_{\lambda}/F_c$')
        ay.set_ylim(0., 2.)

        fig_name = folder_fig_2 + 'lines/' + str(line_label) + '/' +\
            star_name + '_' + str(line_label) + '_velXflux' + '.png'
        plt.tight_layout()
        fig2.savefig(fig_name)
        fig2.clf()

# ------------------------------------------------------------------------------
        # Plotting EW x time
        for i in range(len(ew)):
            mjd = ew[i][1]
            ewi = ew[i][0]
            if ewi != 0:
                az.plot(mjd, ewi, 'o')

        az.minorticks_on()
        az.legend(loc='best')
        az.set_xlabel(r'MJD-OBS')
        az.set_ylabel(r'EW')
        fig_name = folder_fig_2 + 'lines/' + str(line_label) + '/' + \
            star_name + '_' + str(line_label) + '_ew' + '.png'
        plt.tight_layout()
        fig3.savefig(fig_name)
        fig3.clf()

        plt.close()
        plt.close()
        plt.close()

    return


# ==============================================================================
def create_images(list_files, folder_fig):
    '''
    Plot the images for each observed star from the aquisition images.

    :param list_files: list of XShooter fits files (array)
    :param folder_fig: folder where the figures will be saved (string)
    :return: images
    '''

    list_files = list_files[list_files != np.array(None)]
    for i in range(len(list_files)):

        filename, file_extension = os.path.splitext(list_files[i])

        if file_extension == '.fits':
            # Plotting UBV only
            # Rewrite the original fits
            string = str(list_files[i][:])
            rule = '(?!.*\/)(.*?)(?=\.fits)'

            match = re.search(rule, string)
            name_fits = match.group() + '.fits'

            if name_fits[0:2] != 'M.':
                obj, mjd, arm = read_header_aquisition(list_files[i])
                if obj is not False and obj != 'LAMP,FLAT' and \
                    obj != 'BIAS' and obj != 'LAMP,AFC' \
                   and obj != 'STD,TELLURIC' and obj != 'LAMP,DFLAT' \
                   and obj != 'LAMP,ORDERDEF' and obj != 'LAMP,QFLAT' \
                   and obj != 'DARK' and obj is not 'LAMP,FMTCHK' \
                   and obj != 'LAMP,DORDERDEF' and obj != 'LAMP,WAVE' \
                   and obj != 'STD,FLUX' and obj != 'LAMP,QORDERDEF' \
                   and obj != 'FLAT,LINEARITY,DETCHAR' and arm == 'AGC':
                    if obj != 'OBJECT':

                        if os.path.isdir(folder_fig + str(obj)) is False:
                            os.mkdir(folder_fig + str(obj))
                        folder_img = folder_fig + str(obj)

                        if os.path.isdir(folder_img + '/images/') is False:
                            os.mkdir(folder_img + '/images/')
                        folder_img = folder_img + '/images/'

                        print('\nOBJECT: %s \n' % obj)

                        plot_fits_image(folder_fig=folder_img,
                                        fits_file=list_files[i],
                                        obj=obj, mjd=mjd, scale='log',
                                        zoom=True)

                    else:

                        hdulist = pyfits.open(list_files[i])
                        header = hdulist[0].header
                        obj = header['HIERARCH ESO OBS TARG NAME']
                        mjd = header['MJD-OBS']

                        print('\nOBJECT: %s \n' % obj)

                        if os.path.isdir(folder_fig + str(obj)) is False:
                            os.mkdir(folder_fig + str(obj))
                        folder_img = folder_fig + str(obj)

                        if os.path.isdir(folder_img + '/images/') is False:
                            os.mkdir(folder_img + '/images/')
                        folder_img = folder_img + '/images/'

                        plot_fits_image(folder_fig=folder_img,
                                        fits_file=list_files[i],
                                        obj=obj, mjd=mjd, scale='log',
                                        zoom=True)

    return


# ==============================================================================
def bcd_analysis(obj, wave, flux, flux_norm, folder_fig):
    '''
    This function performs the BCD analysis.

    :param obj: star name (string)
    :param wave: Array with the wavelenghts in Angstrom (numpy array)
    :param flux: Array with the fluxes  (numpy array)
    :param flux_norm: Array with the normalized fluxes (numpy array)
    :param folder_fig: folder where the figures will be saved (string)
    :star_name: star name (string)
    :return: D and lambda_0 (BCD parameters)
    '''

    new_flux = []
    new_lambda = []
    new_flux_norm = []
    for i in range(len(flux)):
        if flux[i] >= 0.:
            new_flux.append(flux[i])
            new_flux_norm.append(flux_norm[i])
            new_lambda.append(wave[i])

    if new_lambda[-1] <= 5560. and new_lambda[0] >= 2989.2:

        logflux = np.array(np.log10(new_flux))
        D, lambda_1 = bcd.bcd(obj=obj, wav=new_lambda,
                              nflx=new_flux_norm, logflx=logflux, label='Spec',
                              folder=folder_fig, doplot=True)
        lambda_0 = lambda_1 - 3700.

    return D, lambda_0


# ==============================================================================
def calc_dlamb(wave, flux, center_wave, delta):

    '''
    This function calculates the shift of the line.

    :param wave: Observed wavelenght of the line (float, ex: 6562.81 #AA)
    :param flux: Array with the fluxes (numpy array)
    :param center_wave: Lab central wave (float)
    :param delta: range to be considered around the lamb_obs (float)
    :return: delta lambda (float)
    '''

    # Searching the central line
    x = np.copy(wave)
    y = np.copy(flux)

    z, ind = find_nearest(x, center_wave)
    x = x[ind - delta:ind + delta]
    y = y[ind - delta:ind + delta]

    z2, ind2 = find_nearest(y, np.min(y))

    obs_center_wave = x[ind2]

    delt_lamb = abs(obs_center_wave - center_wave)

    return delt_lamb


# ==============================================================================
def plot_line(wave, flux, center_wave, delta, vel, label, ax,
              calc_central_wave, save_fig=None, fig_name=None,
              gauss_fit=None):

    '''
    Function to plot a line profile.

    :param wave: Observed wavelenght of the line (float, ex: 6562.81 #AA)
    :param flux: Array with the fluxes (numpy array)
    :param center_wave: Lab central wave (float)
    :param delta: range to be considered around the lamb_obs (float)
    :param vel: parameter for the function fit_line (float)
    :param label: label to be plotted in the legend
    :param ax: subfigure name (ax, ay or az)
    :param calc_central_wave: Do calculate the central wave? (boolean)
    :param save_fig: Save the figure? (boolean)
    :param gauss_fit: Would you like to perform a gaussian fit? (boolean)
    :return: figures
    '''

    # Searching the central line
    x = np.copy(wave)
    y = np.copy(flux)

    z, ind = find_nearest(x, center_wave)
    x = x[ind - delta:ind + delta]
    y = y[ind - delta:ind + delta]

    if calc_central_wave is True:
        z2, ind2 = find_nearest(y, np.min(y))
        obs_center_wave = x[ind2]

    ax.axvline(x=center_wave, ls=':', color='gray', alpha=0.5)
    ax.axvline(x=obs_center_wave, ls='--', color='gray', alpha=0.5)
    ax.axhline(y=0, ls='--', color='gray', alpha=0.5)

    ax.plot(wave, flux, label=label)

    if gauss_fit is True:
        fit_line(central_wave=center_wave, delta=delta, vel=vel)

    if save_fig is True:
        plt.minorticks_on()
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_name)
        plt.close()

    return


# ==============================================================================
def fluxXvel(wave, flux, flux_plus_i, central_wave, delta,
             label, ax, fit=None):
    '''
    Function to plot a line profile.

    :param wave: Observed wavelenght of the line (numpy array)
    :param flux: Array with the fluxes (numpy array)
    :param flux_plus_i: Array with the fluxes add by a constant value
     (numpy array)
    :param delta: range to be considered around the lamb_obs (float)
    :param central_wave: Lab central wave (float)
    :param label: label to be plotted in the legend
    :param ax: subfigure name (ax, ay or az)
    :param fit: Do nothing (boolean)
    :return: velocity and associated flux (arrays)
    '''

    vel_min = -1. * delta
    vel_max = +1. * delta

    vel = []
    for i in range(len(wave)):
        delta_lamb = wave[i] - central_wave
        vel.append(light_speed * (delta_lamb / central_wave) * 1.e-3)  # km/s

    ax.plot(vel, flux_plus_i, label=label)
    ax.set_xlim(vel_min, vel_max)

    return vel, flux


# ==============================================================================
def identify_object(file_name):
    '''
    Function to identify the object.

    :param file_name: fits file (string)
    :return: object name (string)
    '''

    if file_name[-6:] != 'README':
        hdulist = pyfits.open(file_name)
        header = hdulist[0].header
        obj = header['OBJECT']

        if obj != 'STD,TELLURIC' and header['HIERARCH ESO PRO CATG'] \
           != 'SKY_TAB_MULT_UVB' and header['HIERARCH ESO PRO CATG'] \
           != 'SKY_TAB_MULT_VIS' and header['HIERARCH ESO PRO CATG'] \
           != 'SKY_TAB_MULT_NIR' and header['HIERARCH ESO PRO CATG'] \
           != 'SCI_SLIT_FLUX_MERGE2D_NIR':

                return obj


# ==============================================================================
def read_header(file_name):
    '''
    Simple function to read the header of a fits file.

    :param file_name: fits file (string)
    :return: header
    '''

    hdulist = pyfits.open(file_name)
    header = hdulist[0].header

    return header


# ==============================================================================
def print_keys(file_name):
    '''
    Simple function to print the keys header of a fits file.

    :param file_name: fits file (string)
    :return: keys
    '''

    hdulist = pyfits.open(file_name)
    header = hdulist[0].header
    keys = header.keys()
    return keys


# ==============================================================================
def remove_negs(num_list):
    '''
    Removes the negative values from a list.

    :param num_list: array of values (array)
    :return: keys
    '''

    r = num_list[:]
    for item in num_list:
        if item < 0:
            r.remove(item)
    return r


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
def create_txt_file(x, y, file_name):
    '''
    Create a txt file.

    :param x: array with n elements (array)
    :param y: array with n elements (array)
    :param file_name: file's name (string)
    :return: txt file
    '''

    with open(file_name, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(x, y))

    return


# ==============================================================================
def smooth_spectrum(wave, flux, doplot=None):
    '''
    Smooth the spectrum by convolving with a (normalized) Hann
    window of 400 points.

    :param wave: Array with the wavelenght (numpy array)
    :param flux: Array with the fluxes (numpy array)
    :param doplot: Would you like to see the plot? (boolean)
    :return: smoothed flux (array)
    '''

    kernel = np.hanning(100)
    kernel = kernel / kernel.sum()
    smoothed = np.convolve(kernel, flux, mode='SAME')

    if doplot is True:
        plt.plot(wave, flux, label='original')
        plt.plot(wave, smoothed, label='smoothed', linewidth=3, color='orange')
        plt.yscale('log')
        plt.tight_layout()
        plt.legend()

    return smoothed


# ==============================================================================
def normalize_spectrum(wave, flux, input_file, output_folder, arm):
    '''
    Function of normalization.

    Source: http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?continuum
    :param wave: Array with the wavelenght (numpy array)
    :param flux: Array with the fluxes (numpy array)
    :param input_file: fits file (string)
    :param output_folder: folder where it will be saved the output (string)
    :return: normalized fits file and normalized flux (output_file.fits)
    '''

    # Rewrite the original fits
    string = str(input_file)
    rule = '(?!.*\/)(.*?)(?=\.fits)'
    match = re.search(rule, string)
    new_list = match.group()

    spt.writeFits(flux, wave, savename='temp/' + new_list + '.fits')
    new_fits = output_folder + new_list
    input_file = new_fits[:] + '.fits'

    output_file = output_folder + new_list + '_norm.fits'

    if arm == 'UVB':
        iraf.noao.onedspec.continuum(input_file, output_file, logscale='no',
                                     function='spline3', order=36,
                                     low_reject=2., interactive=interact,
                                     override='yes', markrej='yes',
                                     bands='*')
    if arm == 'VIS':
        iraf.noao.onedspec.continuum(input_file, output_file, logscale='no',
                                     function='spline3', order=40,
                                     low_reject=2., interactive=interact,
                                     override='yes', markrej='yes', lines='*')
    if arm == 'NIR':
        iraf.noao.onedspec.continuum(input_file, output_file, logscale='no',
                                     function='chebyshev', order=20,
                                     low_reject=2., interactive=interact,
                                     override='yes', markrej='yes', lines='*')

    new_norm_fits = output_folder + new_list + '_norm.fits'
    flux_norm = read_norm_fits(new_norm_fits)

    return new_norm_fits, flux_norm


# ==============================================================================
def read_txt(table_name, ncols):
    '''
    Read a simple txt file.

    :param table_name: name of the table
    :param ncols: number of columns
    :return: x, y (arrays) or x, y, z (arrays)
    '''

    if ncols == 2:
        type = (0, 1)
        a = np.loadtxt(table_name, dtype='float', comments='#', delimiter='\t',
                       skiprows=0, usecols=type, unpack=True, ndmin=0)
        x, y = a[0], a[1]
        return x, y

    if ncols == 3:
        type = (0, 1, 2)
        a = np.loadtxt(table_name, dtype='float', comments='#', delimiter='\t',
                       skiprows=0, usecols=type, unpack=True, ndmin=0)
        x, y, z = a[0], a[1], a[2]
        return x, y, z


# ==============================================================================
def plot_fits_image(folder_fig, fits_file, obj, mjd, scale=None, zoom=None):
    '''
    Read a simple txt file.

    :param folder_fig: folder where it will be saved the figures (string)
    :param fits_file: name of fits file (string)
    :param obj: object name (string)
    :param mjd: MJD (float)
    :param scale: linear, log, and other options (string)
    :param zoom: Plot a zoom of the image? (boolean)
    :return: Image
    '''

    print('fits: %s' % fits_file)
    hdulist = pyfits.open(fits_file)

    # Reading the header
    header = hdulist[0].header
    ra = header['RA']
    dec = header['DEC']

# ------------------------------------------------------------------------------
    # Open figure
    fig = aplpy.FITSFigure(fits_file)

    # show the image with a grey scale:
    fig.show_grayscale()

    # Calculates the median
    median_fig = np.median(fig.image.get_array())
    median_fig = median_fig.data

    fig.show_grayscale(vmin=median_fig, stretch=scale)

# ------------------------------------------------------------------------------
    # Colorbar
    fig.add_colorbar()
    fig.colorbar.show()
    fig.colorbar.set_location('right')
    fig.colorbar.set_width(0.2)  # arbitrary units, default is 0.2
    fig.colorbar.set_pad(0.05)   # arbitrary units, default is 0.05
    fig.colorbar.set_font(size='medium', weight='medium', stretch='normal',
                          family='sans-serif', style='normal',
                          variant='normal')
    fig.colorbar.set_axis_label_text('Counts')
    fig.colorbar.set_axis_label_font(size=12, weight='bold')
    fig.colorbar.set_axis_label_pad(10)

# ------------------------------------------------------------------------------
    # Coordinates
    fig.set_yaxis_coord_type('longitude')

    # Show the image with different color scale
    # fig.show_colorscale(cmap='plasma', stretch='log'), invert=False)
    # optional invert=True
    # fig.show_colorscale(vmin=1.0,vmax=5000,stretch='log',cmap='gist_heat')


# ------------------------------------------------------------------------------
    # Add a scalebar to indicate the size of the image (for 0.01 degrees)
    angle_deg = 0.004  # degrees
    fig.add_scalebar(angle_deg, str(angle_deg) + " degrees", color='black',
                     corner='top right')

    # Change the text of the scalebar to mention that 0.05 angular separation
    # correspond to expressed in pc:

    # SIMBAD query
    # result_table = Simbad.query_object(obj)
    customSimbad = Simbad()
    customSimbad.get_votable_fields()
    customSimbad.add_votable_fields('plx', 'rot')
    customSimbad.get_votable_fields()
    result_table = customSimbad.query_object(obj)
    parallax = result_table['PLX_VALUE']

    distance_pc = 1000. / parallax
    pc_to_au = u.pc.to(u.au)
    distance_au = distance_pc * pc_to_au
    angle_rad = math.radians(angle_deg)
    scale = float(distance_au * angle_rad)

    fig.scalebar.set_label("{:.2f}".format(scale) + " AU")

# ------------------------------------------------------------------------------
    # Change the grid spacing (to be 0.1 degree)
    # fig.ticks.set_xspacing("auto")
    # fig.ticks.set_yspacing("auto")

    # Change the formating of the tick labels
    fig.tick_labels.set_xformat('hh:mm:ss')
    fig.tick_labels.set_yformat('dd:mm:ss')

    # Font size and ticks
    # fig.axis_labels.set_font(size='large')
    # fig.tick_labels.set_font(size='large')

# ------------------------------------------------------------------------------
    # Add a grid over the image
    fig.add_grid()
    fig.grid.set_color('black')
    fig.grid.set_alpha(0.3)
    fig.grid.set_linewidth(0.4)
    fig.grid.x_auto_spacing = True
    fig.grid.y_auto_spacing = True


# ------------------------------------------------------------------------------
    # Add a marker to indicate the position
    fig.show_markers(ra, dec, layer='markers', edgecolor='white',
                     facecolor='none', marker='o', s=10, alpha=0.5)

    # We can plot an array of ra,dec = [...],[...]
    # All arguments of the method scatter() from matplotlib can be used

    # Add a label to indicate the location
    fig.add_label(ra + 0.0055, dec + 0.0055, obj, layer='source',
                  color='black')

# ------------------------------------------------------------------------------

    fig.show_contour(cmap='Blues', smooth=3)
    fig.show_arrows(x=ra, y=dec, dx=0.005, dy=0.005)

# ------------------------------------------------------------------------------
    # Use a present theme for publication
    fig.set_theme('publication')  # Or 'pretty'  - for screen visualisation

    if os.path.isdir(folder_fig) is False:
        os.mkdir(folder_fig)

    # Saving...
    plt.tight_layout()
    fig.save(folder_fig + str(obj) + '_' + str(mjd) + '_img.png')

# ------------------------------------------------------------------------------

    if zoom is True:
        plt.tight_layout()
        fig.recenter(ra, dec, width=0.02, height=0.02)
        fig.save(folder_fig + '/' + str(obj) + '_' + str(mjd) +
                 '_img_zoom.png')

    fig.close()
    return


# ==============================================================================
def main():
    print(num_spa * '=')
    print("\n                                   XSHOOTER \n")

    # Globals
    global line_wave_uvb, line_label_uvb, line_label_uvb_2, line_label_uvb,\
        line_wave_vis, line_label_vis, line_label_vis_2, line_wave_nir,\
        line_label_nir, line_label_nir_2, delta, interact, folder_bcd,\
        folder_fig, results_folders, data, name_ffit

# ------------------------------------------------------------------------------
    # Any thing you must define it is here
    inputs = 'eso_demos'  # 'bruno' or 'eso_demos'
    interact = 'no'  # 'yes' or 'no'
    plot_images = False
    plot_sed_lines = True
    delta = 50.  # Angstrom
    commum_folder = '/home/bruno/Dropbox/1_Tese/5_ESO/'

    os.chdir(commum_folder)

# ------------------------------------------------------------------------------
    # If you'd like to put some new line, this can be done here

    # UVB arm
    line_wave_uvb = [4861., 4341., 4102., 3970., 3889., 3835.]
    line_label_uvb = [r'H$\beta$', r'H$\gamma$', r'H$\delta$', r'H$\epsilon$',
                      r'H$\zeta$', r'H$\eta$']
    line_label_uvb_2 = ['Hbeta', 'Hgamma', 'Hdelta', 'Hepsilon',
                        'Hzeta', 'Heta']

    # VIS arm
    line_wave_vis = [6563., 9531., 6332.05, 6565.5, 6723.5,
                     6621.5, 6678., 7066., 7775.]
    line_label_vis = [r'H$\alpha$', r'S III', r'O I', r'N II', r'S II',
                      r'He II', r'He I', r'He I', r'O']
    line_label_vis_2 = ['Halpha', 'SIII', 'OI', 'NII', 'SII',
                        'HeII', 'HeI', 'HeI_2', 'O']

    # NIR arm
    line_wave_nir = [10050., 16454.687, 21654.,
                     19641., 14300., 12820., 11886., 11287., 10938., 10830.]
    line_label_nir = [r'Pa$\delta$', r'Fe II', r'Br$\gamma$', r'Si $VI$',
                      r'Si $X$', r'Pa$\beta$', r'P II', r'O I', r'Pa$\gamma$',
                      r'He I']
    line_label_nir_2 = ['PaDelta', 'FeII', 'BrGamma', 'SiVI',
                        'SiX', 'PaBeta', 'PII', 'OI', 'PaGamma', 'HeI']


# ------------------------------------------------------------------------------
# In this section, we create automatically the needed folders

    if os.path.isdir(commum_folder + 'results/') is False:
        os.mkdir(commum_folder + 'results/')

    if os.path.isdir(commum_folder + 'temp/') is False:
        os.mkdir(commum_folder + 'temp/')

    if os.path.isdir(commum_folder + 'tables/stars/') is False:
        os.mkdir(commum_folder + 'tables/stars/')

    folder_fig = commum_folder + 'results/'
    folder_table = commum_folder + 'tables/'
    folder_temp = commum_folder + 'temp/'
    folder_bcd = commum_folder + 'tables/stars/'


# ------------------------------------------------------------------------------
# Here we define the input folders (see that there is one for the
# images and another to the data)

    if inputs == 'eso_demos':
        input_data = commum_folder + 'data_demo/'
        input_data_raw = '/home/bruno/Downloads/reduce-eso/demo/reflex_input/'

# ------------------------------------------------------------------------------
    if plot_images is True:

        print(num_spa * '=')
        print('\nCreating stellar images...\n')

        create_list_files(list_name='fits_img', folder=input_data_raw,
                          folder_table=folder_table)

        list_files = read_list_files_all(table_name='fits_img.txt',
                                         folder_table=folder_table)

        list_stars = create_list_stars_img(list_files)

        print('number of stars: %s \n' % len(list_stars))

        create_images(list_files=list_stars, folder_fig=folder_fig)

# ------------------------------------------------------------------------------
    if plot_sed_lines is True:

        print(num_spa * '=')
        print('\nCreating list of files...\n')

        create_list_files(list_name='fits_files', folder=input_data,
                          folder_table=folder_table)

        list_files = read_list_files_all(table_name='fits_files.txt',
                                         folder_table=folder_table)

        list_stars = create_list_stars(list_files)

        if np.size(list_stars) == 1:
            print('number of stars: 1')
            if os.path.exists(folder_fig + str(list_stars)) is True and \
               os.path.exists(folder_fig + str(list_stars) + 'sed/') is True \
               and os.path.exists(folder_fig + str(list_stars) + 'bcd/') \
               is True:
                print('This star has already been calculated ...')
            else:
                print(num_spa * '=')

                data = create_class_object(list_files=list_files,
                                           obj=str(list_stars))

                plot_arm_lines(data=data, folder_fig=folder_fig,
                               folder_temp=folder_temp, delta=delta,
                               star_name=str(list_stars))

                plot_sed(data=data, folder_fig=folder_fig,
                         folder_temp=folder_temp,
                         star_name=str(list_stars))
        else:
            print('number of stars: %s \n' % len(list_stars))
            for i in range(len(list_stars)):
                if os.path.exists(folder_fig + list_stars[i]) is True and \
                   os.path.exists(folder_fig + list_stars[i] + 'sed/') \
                   is True and os.path.exists(folder_fig + list_stars[i] +
                                              'bcd/') is True:
                    print('This star has already been calculated ...')
                else:
                    print(num_spa * '=')
                    print('\n%sth star \n' % (i + 1))

                    data = create_class_object(list_files=list_files,
                                               obj=list_stars[i])

                    plot_arm_lines(data=data, folder_fig=folder_fig,
                                   folder_temp=folder_temp, delta=delta,
                                   star_name=list_stars[i])

                    plot_sed(data=data, folder_fig=folder_fig,
                             folder_temp=folder_temp,
                             star_name=list_stars[i])

    return

# ==============================================================================
if __name__ == '__main__':
    main()
