import pyhdust.phc as phc
import pyfits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PyAstronomy import pyasl
import pyhdust.spectools as spt
import pyhdust as hdt
import os
import re
from numpy import nan as nan
import matplotlib as mpl
from scipy.fftpack import fft, fftfreq
from scipy.signal import fftconvolve
from scipy.special import j1
from numpy import pi, sin, cos, sqrt


mpl.rcParams.update({'font.size': 18})
mpl.rcParams['lines.linewidth'] = 2
font = fm.FontProperties(size=17)
mpl.rc('xtick', labelsize=17)
mpl.rc('ytick', labelsize=17)
fontsize_label = 18  # 'x-large'
num_spa = 87  # 151, 176, 176

__version__ = "0.0.1"
__author__ = "Bruno Mota, Daniel Moser"


# ==============================================================================

class data_object:
    '''
    Class of the stellar objects. Using this class, we can store
    for each star a sort of variables, which can be easily
    accessed.

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

                    boolop, obj, obs_date, mjd, arm, wave, flux,\
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

def vsini_calc_kurucz(wvl, flux, teff, logg, center_wave, flux_err,
                      limbdark, vsini, color, folder_fig, mjd, star_name,
                      lbdc, ltem, llog, lambd, prof, arm):

    '''
    Function that calculate vsini using rotational broadening pyasl.rotBroad
    and the Kurucz Grids of model atmospheres.

    :param limbdark:  between 0 (no) to 1 (maximum)
    :param wvl: observational wavelength (array)
    :param flux: observational flux (array)
    :param flux_err: observational flux error (array)
    :param teff: effective temperature (float)
    :param logg: gravitational acceleration (float)
    :param center_wave: centre wavelength of the line (float)
    :param vsini: array of possible values of vsini (array)
    :param color: array of colors (array)
    :param folder_fig: folder where the figures will be save (string)
    :param mjd: modified Julian date (float)
    :param star_name: star name (string)
    :param lbdc: list of wavelengths from the model (array)
    :param ltem: list of temperatures from the model (array)
    :param llog: list of logg from the model (array)
    :param lambd: wavelenghts from the model (array)
    :param prof: fluxes from the model (array)
    :param arm: xshooter arm (string)
    :return: plots, best vsini, best limbdark, and reduced chi2 values
    '''
    print(num_spa * '-')
    print('\nFinding nearest model...\n')

    lbdc_near, idx_lbcd = find_nearest(array=np.array(lbdc), value=center_wave)
    teff_near, idx_teff = find_nearest(array=np.array(ltem), value=teff)
    logg_near, idx_logg = find_nearest(array=np.array(llog), value=logg)

    # Models data
    idx = np.where((ltem == teff_near) & (llog == logg_near))
    flux_model = prof[idx, idx_lbcd, :].flatten()
    wave_model = lambd[idx_lbcd]
    sigma_model = flux_model * 0.001

    wave_model, flux_model, sigma_model = \
        cut_spec(wave=wave_model, flux=flux_model, err=sigma_model,
                 center_wave=center_wave, delta=50)

# ------------------------------------------------------------------------------
    # Normalization
    print('Selecting best range ...\n')

    idx = np.where((wave_model > center_wave - 50) &
                   (wave_model < center_wave + 50))
    flux_model = spt.linfit(wave_model[idx], flux_model[idx])
    wave_model = wave_model[idx]

    flux_model_list = list(flux_model)
    indx_min = flux_model_list.index(np.min(flux_model))
    delt_2 = abs(wave_model[indx_min] - center_wave)

    if wave_model[indx_min] > center_wave:
        wave_model = wave_model - delt_2
    else:
        wave_model = wave_model + delt_2

# ------------------------------------------------------------------------------
    print('Centering the lines ...\n')

    flux_list = list(flux)
    indx_min = flux_list.index(np.min(flux_list))

    plt.plot(wvl[indx_min], flux[indx_min], 'o', markersize=12)

    wvl_adj = wvl[indx_min - 20:indx_min + 20]
    flux_adj = flux[indx_min - 20:indx_min + 20]
    a, b, c = np.polyfit(x=wvl_adj, y=flux_adj, deg=2)

    fit = a * wvl**2 + b * wvl + c

    fit_list = list(fit)
    ind_adj = fit_list.index(np.min(fit))
    center_adj = wvl[ind_adj]

    delt_2 = center_adj - center_wave

    if center_adj > center_wave:
        wvl = wvl + delt_2
    else:
        wvl = wvl - delt_2

    # plt.plot(wvl, fit, '-.')

# ------------------------------------------------------------------------------
    # Looping
    print('Adjusting to vsinis...\n')
    rflux_arr, chi2red_arr, vsini_arr_2, limbdark_arr = [], [], [], []
    for k in range(len(vsini)):
        # print('Adjust to vsini = %0.2f ...\n' % vsini[k])
        for w in range(len(limbdark)):
            wvl_rflux = np.linspace(np.min(wave_model), np.max(wave_model),
                                    len(flux_model))

            flux_model = np.interp(wvl_rflux, wave_model, flux_model)

            rflux = pyasl.rotBroad(wvl_rflux, flux_model, limbdark[w],
                                   vsini[k], edgeHandling='firstlast')

            # plt.plot(wave_model, flux_model, label=('A={:.2f} al={:.2f}'.
            #    format(A[i], al[j])))

            flux_interp = np.interp(wvl_rflux, wvl, flux)
            flux_err_interp = np.interp(wvl_rflux, wvl, flux_err)

            chi2red = (sum(((flux_interp - rflux) / flux_err_interp)**2)) / \
                      (len(rflux) - 1 - 2)

            rflux_arr.append(rflux)
            vsini_arr_2.append(vsini[k])
            limbdark_arr.append(limbdark[w])
            chi2red_arr.append(chi2red)

    ind_min_adj = chi2red_arr.index(np.min(chi2red_arr))
    best_vsini = vsini_arr_2[ind_min_adj]
    best_limbd = limbdark_arr[ind_min_adj]

# ------------------------------------------------------------------------------
    print('Plotting and saving...')
    plt.vlines(x=center_wave, ymin=0.0, ymax=2.0, linestyles='--')

    plt.plot(wave_model, flux_model, label='model', color='orange')

    plt.plot(wvl_rflux, rflux_arr[ind_min_adj], label=('vsini={:.2f}, \
        limbdark={:.2f}'.format(best_vsini, best_limbd)), color='lightgreen')

    plt.plot(wvl, flux, 'b-', label='original')

    # Plot the results
    plt.title("Rotational broadening")
    plt.xlabel("Wavelength $[\AA]$")
    plt.ylabel("Normalized flux")
    plt.ylim(0.3, 1.2)
    plt.xlim(center_wave - 30, center_wave + 30)
    plt.legend(loc='best', fontsize=12)

    if os.path.isdir(folder_fig + star_name + '/vsini/') is False:
        os.mkdir(folder_fig + star_name + '/vsini/')
    folder_fig_2 = folder_fig + star_name + '/vsini/'

    name_fig = folder_fig_2 + star_name + '_' + str(mjd) + '_' + \
        arm + '_' + 'vsini.png'
    plt.tight_layout()
    plt.savefig(name_fig)
    plt.close()

    return chi2red_arr, vsini_arr_2


# ==============================================================================
def vsini_calc_scipy(wvl, flux, center_wave, flux_err, limbdark,
                     vsini, star_name, folder_fig, mjd, arm,
                     lbdc, ltem, llog, lambd, prof,
                     teff, logg):
    '''
    Function to calculate vsini using Fourier transform (Scipy)

    :param wvl: observed wavelengths (array)
    :param flux: observed flux (array)
    :param center_wave: central line wave (float)
    :param flux_err: flux error (array)
    :param limbdark:  between 0 (no) to 1 (maximum)
    :param vsini: input vsini (float)
    :param star_name: star's name (string)
    :param folder_fig: folder's figures (string)
    :param mjd: modified Julian date (float)
    :param arm: xshooter's arm (string)
    :return: chi2red and best vsini
    '''

# ------------------------------------------------------------------------------
    # Constants
    cc = 2.99792458e+18  # velocity of light in angstrom/s
    vsini_2 = vsini * 1e13  # input vsini in A/s (75 km/s)
    dlam = wvl[1] - wvl[0]  # spectrum resolution (A)
    delta_1 = center_wave - wvl[0]
    delta_2 = wvl[-1] - center_wave
    delta = center_wave * vsini_2 / cc

# ------------------------------------------------------------------------------
    print('Centering the lines ...\n')

    flux_list = list(flux)
    indx_min = flux_list.index(np.min(flux_list))

    plt.plot(wvl[indx_min], flux[indx_min], 'o', markersize=12)

    wvl_adj = wvl[indx_min - 20:indx_min + 20]
    flux_adj = flux[indx_min - 20:indx_min + 20]
    a, b, c = np.polyfit(x=wvl_adj, y=flux_adj, deg=2)

    fit = a * wvl**2 + b * wvl + c

    fit_list = list(fit)
    ind_adj = fit_list.index(np.min(fit))
    center_adj = wvl[ind_adj]

    delt_2 = center_adj - center_wave

    if center_adj > center_wave:
        wvl = wvl + delt_2
    else:
        wvl = wvl - delt_2

# ------------------------------------------------------------------------------
    # The rotational broadening function G
    lambdas = np.linspace(-delta_1, delta_2, 10. / dlam + 1)
    y = 1 - (lambdas / delta)**2  # transformation of wavelengths
    G = (2 * (1 - limbdark) * sqrt(y) + pi * limbdark / 2. * y) / \
        (pi * delta * (1 - limbdark / 3))  # the kernel

    keep = -np.isnan(G)  # returns boolean arr with False where there are nans
    lambdas, G = lambdas[keep], G[keep]  # crop the arrays not to contain nans
    lambdas.min(), lambdas.max()

# ------------------------------------------------------------------------------
    # Synthetic Line
    print(num_spa * '-')
    print('\nFinding nearest model...\n')

    lbdc_near, idx_lbcd = find_nearest(array=np.array(lbdc), value=center_wave)
    teff_near, idx_teff = find_nearest(array=np.array(ltem), value=teff)
    logg_near, idx_logg = find_nearest(array=np.array(llog), value=logg)

# ------------------------------------------------------------------------------
    # Models data
    idx = np.where((ltem == teff_near) & (llog == logg_near))
    flux_model = prof[idx, idx_lbcd, :].flatten()
    wave_model = lambd[idx_lbcd]
    sigma_model = flux_model * 0.001

    wave_model, flux_model, sigma_model = \
        cut_spec(wave=wave_model, flux=flux_model, err=sigma_model,
                 center_wave=center_wave, delta=50)

# ------------------------------------------------------------------------------
    # Normalization
    print('Selecting best range ...\n')

    idx = np.where((wave_model > center_wave - 50) &
                   (wave_model < center_wave + 50))
    flux_model = spt.linfit(wave_model[idx], flux_model[idx])
    wave_model = wave_model[idx]

    flux_model_list = list(flux_model)
    indx_min = flux_model_list.index(np.min(flux_model))
    delt_2 = abs(wave_model[indx_min] - center_wave)

    if wave_model[indx_min] > center_wave:
        wave_model = wave_model - delt_2
    else:
        wave_model = wave_model + delt_2

# ------------------------------------------------------------------------------
    # Spectral Line
    wavelengths = np.copy(wave_model)
    spec_line = np.copy(flux_model)

# ------------------------------------------------------------------------------
    # Convolution
    # spec_conv = fftconvolve(1 - spec_line, G, mode='full')
    spec_conv = fftconvolve(1 - spec_line, G, mode='same')
    N = len(spec_line)  # + len(G) + 1  # when mode='full'
    wavelengths_conv = np.arange(-N / 2, N / 2, 1) * dlam + center_wave
    EW_before = np.trapz(1 - spec_line, x=wavelengths)
    EW_after = np.trapz(spec_conv, x=wavelengths_conv)
    spec_conv = 1 - spec_conv / EW_after * EW_before

# ------------------------------------------------------------------------------
    # The analytical Fourier transform
    x = np.linspace(0, 30, 1000)[1:]  # exclude zero
    g = 2. / (x * (1 - limbdark / 3.)) * ((1 - limbdark) * j1(x) +
                                          limbdark * (sin(x) / x**2 -
                                          cos(x) / x))
    g = g**2  # convert to power
    x /= (2 * pi * delta)

# ------------------------------------------------------------------------------
    # Array ajust
    excl = abs(len(wavelengths_conv) - len(flux))

    # if len(wavelengths_conv) < len(flux):
    #     for i in range(excl):
    #         if i % 2 == 0:
    #             flux = flux[:-1]
    #         else:
    #             flux = flux[1:]
    # elif len(wavelengths_conv) > len(flux):
    #     for i in range(excl):
    #         if i % 2 == 0:
    #             wavelengths_conv = wavelengths_conv[:-1]
    #         else:
    #             wavelengths_conv = wavelengths_conv[1:]
    # else:
    #     pass

    excl = abs(len(spec_conv) - len(wavelengths_conv))

    if len(wavelengths_conv) < len(spec_conv):
        for i in range(excl):
            spec_conv = spec_conv[:-1]
    elif len(wavelengths_conv) > len(spec_conv):
        for i in range(excl):
            wavelengths_conv = wavelengths_conv[:-1]
    else:
        pass

# ------------------------------------------------------------------------------
    # Computing the Fourier transform
    keep = np.abs(center_wave - wavelengths_conv) < 1.2

    spec_to_transform = (1 - flux)[keep]  # we need the continuum at zero
    new_n = 100 * len(spec_to_transform)  # new length for zeropadding ????
    spec_fft = np.abs(fft(spec_to_transform, n=new_n))**2  # power of FFT

    x_fft = fftfreq(len(spec_fft), d=dlam)
    keep = x_fft >= 0  # only positive frequencies
    x_fft, spec_fft = x_fft[keep], spec_fft[keep]

# ------------------------------------------------------------------------------
    # The measured vsini corresponds to the first zero in the Fourier transform
    neg_to_pos = (np.diff(spec_fft[:-1]) <= 0) & (np.diff(spec_fft[1:]) >= 0)
    minima = x_fft[1:-1][neg_to_pos]

# ------------------------------------------------------------------------------
    # The frequency domain can be converted to velocity (km/s) as follows
    q1 = 0.610 + 0.062 * limbdark + 0.027 * limbdark**2 + 0.012 * limbdark**3\
        + 0.004 * limbdark**4
    vsini_zeros = cc / center_wave * q1 / minima / 1e13
    print('vsini:')
    print(vsini_zeros[:10])
    print(minima)

# ------------------------------------------------------------------------------
    # Plot the results
    fig = plt.figure()

    ax = plt.subplot(211)
    ax.plot(x, g / g.max())
    ax.plot(x_fft, spec_fft / spec_fft.max(), '--')
    # ax.set_xlabel(r'$v \sin i$ $\rm [km/s]$')
    # ax.set_ylabel(r'$F_{\lambda}/F_c$')
    ax.set_yscale('log')
    ax.set_xticks(minima[:5] * center_wave / q1 / cc * 1e13,
                  ['{0:.2f}'.format(i) for i in vsini_zeros[:5]])
    # ax.arrow(minima[0], 0, 0.5, 0.5, head_width=0.01,
    #          head_length=0.05, fc='k', ec='k')

    ay = plt.subplot(212)
    ay.plot(wavelengths, spec_line, '--', label='non-rotating model')
    ay.plot(wavelengths_conv, spec_conv, 'o',
            label=('Best vsini: %0.2f km/s' % vsini_zeros[0]))
    ay.plot(wvl, flux, '-', label='observed')
    ay.set_xlabel(r'$\lambda$ $[\AA ]$')
    ay.set_ylabel(r'$F_{\lambda}/F_c$')
    ay.set_xlim(min(wvl), max(wvl))

    if os.path.isdir(folder_fig + star_name + '/vsini/fft/') is False:
        os.mkdir(folder_fig + star_name + '/vsini/fft/')
    folder_fig_2 = folder_fig + star_name + '/vsini/fft/'

    name_fig = folder_fig_2 + star_name + '_' + str(mjd) + '_' + \
        arm + '_' + 'vsini_fft.png'

    fig.subplots_adjust(hspace=0.0001)
    # ax.legend(loc='best', fontsize=6, fancybox=False)
    plt.locator_params(axis='y', nbins=5)
    plt.legend(loc='best', fontsize=6, fancybox=False)
    plt.tight_layout()
    plt.savefig(name_fig)
    plt.close()

    best_vsinis = vsini_zeros[:10]

    return best_vsinis


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
    Read list of files in a table, and returns all
    fits file in an array.

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
def read_fits_star_name(file_name, typ):
    '''
    Read XShooter's fits files.

    :param file_name: name of the file in fit extension (string)
    :param typ: define if it will be used to read a image
    (typ='img')or a bintable (typ='data').
    :return: parameters (object if typ='data',
    file_name if typ='img' )
    '''

    if file_name[-3:] != 'txt' and file_name[-2:] != '.Z' \
       and file_name[-3:] != 'xml' and file_name[-2:] != 'db' \
       and file_name[-3:] != 'log':

        print(file_name)
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
def read_fits_xshooter(file_name, print_obj):
    '''
    Read XShooter's fits files.

    :param file_name: name of the file in fit extension (string)
    :return: parameters
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

    if hasattr(tbdata, 'field') is True and xtension != 'IMAGE' \
       and ('WAVE' in tbdata.names) is True:

        wave = tbdata.field('WAVE')
        flux = tbdata.field('FLUX')
        sigma = tbdata.field('ERR')
        qual = tbdata.field('QUAL')
        snr = tbdata.field('SNR')
        boolop = True

        return boolop, obj, obs_date, mjd, arm,\
            wave, flux, sigma, qual, snr

    else:
        boolop = False

        return boolop


# ==============================================================================
def read_fits(file_name):

    '''
    Read XShooter's fits files.

    :param file_name: name of the file in fit extension (string)
    :return: parameters
    '''

    hdulist = pyfits.open(file_name)
    header = hdulist[0].header
    tbdata = hdulist[0].data
    obj = header['OBJECT']
    mjd = header['MJD-OBS']
    obs_date = header['DATE-OBS']

    flux = np.copy(tbdata)
    wave = np.ones(len(flux)) * header['CRVAL1']
    wave = wave + header['CDELT1'] * np.arange(len(flux))
    sigma = np.array(len(flux) * [0.01])

    return obj, obs_date, mjd, wave, flux, sigma


# ==============================================================================
def cut_spec(wave, flux, err, center_wave, delta):

    '''
    Cut a spectra to a given range.

    :param wave: array with the wavelengths (array)
    :param flux: array with the wavelengths (array)
    :param center_wave: central wavelength (float)
    :param delta: interval to be considered around
        the central wavelength (float)
    :return: cut arrays (x, y, z)
    '''

    # Searching the central line
    x = np.copy(wave)
    y = np.copy(flux)
    w = np.copy(err)

    z, ind = find_nearest(x, center_wave)
    x = x[ind - delta:ind + delta]
    y = y[ind - delta:ind + delta]
    w = w[ind - delta:ind + delta]

    return x, y, w


# ==============================================================================
def find_nearest(array, value):
    '''
    Find the nearest value inside an array.

    :param array: array
    :param value: desired value (float)
    :return: nearest value and its index
    '''

    idx = (np.abs(array - value)).argmin()

    return array[idx], idx


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
        a = np.loadtxt(table_name, dtype='float', comments='#',
                       delimiter='\t', skiprows=0, usecols=type,
                       unpack=True, ndmin=0)
        x, y = a[0], a[1]
        return x, y

    if ncols == 3:
        type = (0, 1, 2)
        a = np.loadtxt(table_name, dtype='float', comments='#',
                       delimiter='\t', skiprows=0, usecols=type,
                       unpack=True, ndmin=0)
        x, y, z = a[0], a[1], a[2]
        return x, y, z


# ==============================================================================
def smooth_spectrum(wave, flux, doplot=None):
    '''
    Smooth the spectrum by convolving with a (normalized)
    Hann window of 400 points.

    :param wave: Array with the wavelenght (numpy array)
    :param flux: Array with the fluxes (numpy array)
    :param doplot: Would you like to see the plot? (boolean)
    :return: smoothed flux (array)
    '''

    kernel = np.hanning(60)
    kernel = kernel / kernel.sum()

    smoothed = np.convolve(kernel, flux, mode='valid')

    if doplot is True:
        plt.plot(wave, flux, label='original')
        plt.plot(wave, smoothed, label='smoothed', linewidth=3,
                 color='orange')
        plt.yscale('log')
        plt.legend()

    return smoothed


# ==============================================================================
def read_params_star(table_name):

    '''
    Read the output of the bcd method (txt file).

    :param table_name: name of the table (string)
    :return: stellar parameters
    '''

    try:
        typ = (0, 1, 2, 3, 4, 5)
        a = np.loadtxt(table_name, dtype='float', comments='#',
                       delimiter='\t', skiprows=0, usecols=typ,
                       unpack=True, ndmin=0)
        D_o, lb0_o, te_ip, lgg_ip, Tp_ip, Ms_ip = a[0], a[1],\
            a[2], a[3], a[4], a[5]

        return D_o, lb0_o, te_ip, lgg_ip, Tp_ip, Ms_ip
    except:

        print('There is no bcd parameters.')
        D_o, lb0_o, te_ip, lgg_ip, Tp_ip, Ms_ip = 0, 0, 0, 0, 0, 0

        return D_o, lb0_o, te_ip, lgg_ip, Tp_ip, Ms_ip


# ==============================================================================
def plot_vsini(data, vsini_arr, limbdark, folder_fig, folder_temp,
               folder_table, star_name, center_wave_uvb,
               center_wave_vis, center_wave_nir, delta,
               plotchi2_map, lbdc, ltem, llog, lambd, prof):

    '''
    Plot the SED for each star inside the data structure.

    :param data: list of XShooter fits files (array)
    :param folder_fig: folder where the figures will be saved (string)
    :param folder_temp: folder of the temporary files (string)
    :param limbdark:  between 0 (no) to 1 (maximum)
    :param center_wave: centre wavelength of the line (float)
    :param vsini: array of possible values of vsini (array)
    :param mjd: modified Julian date (float)
    :param star_name: star name (string)
    :param lbdc: list of wavelengths from the model (array)
    :param ltem: list of temperatures from the model (array)
    :param llog: list of logg from the model (array)
    :param lambd: wavelenghts from the model (array)
    :param prof: fluxes from the model (array)
    :return: plots, best vsini, best limbdark, and reduced chi2 values
    '''

    global conc_wave, conc_flux, conc_sigm, conc_mjd, conc_arm, \
        conc_flux_norm, conc_name_ffit

# ------------------------------------------------------------------------------
    print(num_spa * '=')
    print('\nPlotting Vsini\n')
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
            name_ffit = data[i].name_ffit

            conc_wave.append(wave)
            conc_flux.append(flux)
            conc_sigm.append(sigm)
            conc_mjd.append(mjd)
            conc_arm.append(arm)
            conc_name_ffit.append(name_ffit)

# ------------------------------------------------------------------------------
    if len(conc_wave) != 0 and len(conc_flux) != 0:

        plt.figure()
        if len(conc_arm) != 1:
            cor = phc.gradColor(np.arange(len(conc_arm)), cmapn='inferno')
        else:
            cor = 'black'

        print('Reading stellar parameters...\n')
        data_folder = folder_fig + star_name + '/bcd/'
        table_name = data_folder + star_name + '.txt'

        D_o, lb0_o, teff, logg, Tp_ip, Ms_ip = \
            read_params_star(table_name=table_name)

        logg = np.mean(logg)
        teff = np.mean(teff)

        if logg != nan and teff != nan and logg != 0. and teff != 0.:
            for i in range(len(conc_arm)):
                if conc_arm[i] == 'VIS':
                    center_wave = np.copy(center_wave_vis)
                if conc_arm[i] == 'UVB':
                    center_wave = np.copy(center_wave_uvb)
                if conc_arm[i] == 'NIR':
                    center_wave = np.copy(center_wave_nir)
                wave = np.copy(conc_wave[i])
                flux = np.copy(conc_flux[i])
                sigma = np.copy(conc_sigm[i])
                mjd = np.copy(conc_mjd[i])

                cut_wave, cut_flux, cut_ferr = \
                    cut_spec(wave=wave, flux=flux, err=sigma,
                             center_wave=center_wave, delta=delta)

                # Normalizing
                idx = np.where((cut_wave > center_wave - 50) &
                               (cut_wave < center_wave + 50))
                cut_flux_norm = spt.linfit(cut_wave[idx], cut_flux[idx])
                cut_ferr_norm = spt.linfit(cut_wave[idx], cut_ferr[idx])
                cut_wave = cut_wave[idx]

                cor = phc.gradColor(np.arange(len(vsini_arr)), cmapn='inferno')

                chi2red_arr, vsini_arr_2 = \
                    vsini_calc_kurucz(wvl=cut_wave, flux=cut_flux_norm,
                                      teff=teff, logg=logg,
                                      center_wave=center_wave,
                                      flux_err=cut_ferr_norm,
                                      limbdark=limbdark,
                                      vsini=vsini_arr, color=cor,
                                      folder_fig=folder_fig,
                                      mjd=mjd, star_name=star,
                                      lbdc=lbdc, ltem=ltem,
                                      llog=llog, lambd=lambd,
                                      prof=prof, arm=conc_arm[i])

# ------------------------------------------------------------------------------
# Plotting chi square map

                if plotchi2_map is True:

                    cor = phc.gradColor(np.arange(len(vsini_arr_2)),
                                        cmapn='inferno')
                    plt.figure()
                    for j in range(len(chi2red_arr)):
                        plt.scatter(vsini_arr_2[j], chi2red_arr[j],
                                    color=cor[j])

                    ind_min = chi2red_arr.index(np.min(chi2red_arr))
                    best_vsini = vsini_arr_2[ind_min]
                    plt.plot(best_vsini, chi2red_arr[ind_min], 'o',
                             label=('Best vsini: %0.2f km/s' % best_vsini))
                    plt.legend(loc='best')
                    plt.xlabel(r'v sini [km/s]')
                    plt.ylabel(r'$\chi^2_{red}$')
                    ymin = np.min(chi2red_arr)
                    ymax = np.max(chi2red_arr)
                    plt.ylim(ymin - ymin * 0.2, ymax + ymax * 0.2)

                    plt.minorticks_on()

                    if os.path.isdir(folder_fig + star + '/vsini/') is False:
                        os.mkdir(folder_fig + star + '/vsini/')
                    folder_fig_2 = folder_fig + star + '/vsini/'

                    name_fig = folder_fig_2 + star + '_' + str(mjd) + '_'\
                        + str(conc_arm[i]) + '_' + 'vsini_chi2.png'
                    plt.tight_layout()
                    plt.savefig(name_fig)
                    plt.close()

# ------------------------------------------------------------------------------
            # Calc vsini using the scipy method

                limbdark_fft = 0.6  # input limb darkening
                vsini_fft = 75.  # input vsini in A/s (75 km/s)

                best_vsinis_fft = \
                    vsini_calc_scipy(wvl=cut_wave, flux=cut_flux_norm,
                                     center_wave=center_wave,
                                     flux_err=cut_ferr_norm,
                                     limbdark=limbdark_fft,
                                     vsini=vsini_fft, star_name=star,
                                     folder_fig=folder_fig, mjd=mjd,
                                     arm=conc_arm[i], teff=teff, logg=logg,
                                     lbdc=lbdc, ltem=ltem, llog=llog,
                                     lambd=lambd, prof=prof)

    return


# ==============================================================================
def main():

    # Globals
    global line_wave_uvb, line_label_uvb, line_label_uvb_2, line_label_uvb,\
        line_wave_vis, line_label_vis, line_label_vis_2, line_wave_nir,\
        line_label_nir, line_label_nir_2, delta, interact, folder_bcd,\
        folder_fig, results_folders, data, name_ffit, lbdc, ltem, llog,\
        lambd, prof, vsini_arr, limbdark


# ------------------------------------------------------------------------------
    # Anything you must define is here
    inputs = 'eso'  # eso or 'eso_demos'
    delta = 200
    plotchi2_map = True
    center_wave_vis = 6562.8   # Halph
    center_wave_uvb = 4861.    # Hbeta
    center_wave_nir = 21654.0  # FeII = 16454.687 BrGam = 21654.
    vsini_arr = np.arange(1., 600., 10.)
    limbdark = np.arange(0, 1.1, 0.1)
    commum_folder = '/scratch/home/sysadmin/MEGAsync/XShooter/'

# ------------------------------------------------------------------------------
# In this section, we create automatically the needed folders

    if os.path.isdir(commum_folder + 'results/') is False:
        os.mkdir(commum_folder + 'results/')

    if os.path.isdir(commum_folder + 'routines/temp/') is False:
        os.mkdir(commum_folder + 'routines/temp/')

    if os.path.isdir(commum_folder + 'tables/stars/') is False:
        os.mkdir(commum_folder + 'tables/stars/')

    folder_fig = commum_folder + 'results/'
    folder_table = commum_folder + 'tables/'
    folder_temp = commum_folder + 'routines/temp/'
    folder_bcd = commum_folder + 'tables/stars/'

# ------------------------------------------------------------------------------
# Here we define the input folders (see that there is one for the
# images and another to the data)

    if inputs == 'eso_demos':
        input_data = commum_folder + 'data_demo/'

    if inputs == 'eso':
        input_data = '/run/media/sysadmin/SAMSUNG/reduced/'


# ------------------------------------------------------------------------------

    # Reading Kurucz's models
    xdr_folder = folder_table + 'kur_ap00k0.xdr'

    lbdc, ltem, llog, lambd, prof = hdt.readphotxdr(xdr_folder)


# ------------------------------------------------------------------------------

    print(num_spa * '=')
    print('\nCreating list of files...\n')

    create_list_files(list_name='fits_files', folder=input_data,
                      folder_table=folder_table)

    list_files = read_list_files_all(table_name='fits_files.txt',
                                     folder_table=folder_table)

    list_stars = create_list_stars(list_files)

    for i in range(len(list_stars)):
        print('Star %s of %s \n' % ((i + 1), len(list_stars)))

        data = create_class_object(list_files=list_files, obj=list_stars[i])

        plot_vsini(data=data, vsini_arr=vsini_arr, limbdark=limbdark,
                   folder_fig=folder_fig, folder_temp=folder_temp,
                   folder_table=folder_table, star_name=list_stars[i],
                   center_wave_vis=center_wave_vis,
                   center_wave_uvb=center_wave_uvb,
                   center_wave_nir=center_wave_nir, delta=delta,
                   plotchi2_map=plotchi2_map, lbdc=lbdc, ltem=ltem, llog=llog,
                   lambd=lambd, prof=prof)

# ==============================================================================
if __name__ == '__main__':
    main()
