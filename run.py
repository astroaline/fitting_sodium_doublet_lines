import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from astropy.io import fits
import astropy.units as u
from astropy.constants import c
import importlib

my_path = os.getcwd() + '/'

obj_name = 'PDS_70'
instrument_name = 'HARPS'

# Path of my data
data_path = '/data/'

# Path where I want the fits to be saved
output_path = '/results/'

# This is a list of file names of all PDS 70 HARPS data
file_list = ['HARPS.2018-03-29T06:33:21.929',
             'HARPS.2018-03-29T06:48:53.411',
             'HARPS.2018-03-29T07:04:24.413',
             'HARPS.2018-03-29T07:19:55.404',
             'HARPS.2018-03-29T07:36:39.719',
             'HARPS.2018-03-29T07:52:10.411',
             'HARPS.2018-03-30T05:41:37.996',
             'HARPS.2018-03-30T05:57:09.427',
             'HARPS.2018-03-30T06:12:40.418',
             'HARPS.2018-03-30T08:22:46.842',
             'HARPS.2018-03-30T08:38:17.403',
             'HARPS.2018-03-30T08:53:48.465',
             'HARPS.2018-03-31T03:39:16.713',
             'HARPS.2018-03-31T03:54:47.425',
             'HARPS.2018-03-31T06:35:19.070',
             'HARPS.2018-03-31T06:50:49.413',
             'HARPS.2018-03-31T08:28:24.178',
             'HARPS.2018-03-31T08:43:55.480',
             'HARPS.2018-04-18T05:12:34.801',
             'HARPS.2018-04-19T05:04:01.822',
             'HARPS.2018-04-19T05:19:33.003',
             'HARPS.2018-04-20T05:19:03.239',
             'HARPS.2018-04-20T05:49:35.891',
             'HARPS.2018-04-21T05:51:29.174',
             'HARPS.2018-04-22T05:25:12.656',
             'HARPS.2018-04-23T04:50:54.677',
             'HARPS.2018-05-01T04:29:40.281',
             'HARPS.2018-05-01T05:00:11.202',
             'HARPS.2018-05-06T03:28:07.286',
             'HARPS.2018-05-06T03:58:38.487',
             'HARPS.2018-05-13T05:09:56.832',
             'HARPS.2018-05-13T05:40:27.782',
             'HARPS.2019-02-13T08:32:20.542',
             'HARPS.2019-02-13T09:02:50.943',
             'HARPS.2019-05-01T03:26:54.083',
             'HARPS.2019-05-01T04:07:23.608',
             'HARPS.2020-02-25T06:48:17.236',
             'HARPS.2020-02-25T07:18:48.758',
             'HARPS.2020-02-26T06:38:53.902',
             'HARPS.2020-02-26T07:09:26.084',
             'HARPS.2020-02-27T04:53:10.180',
             'HARPS.2020-02-27T05:23:41.661',
             'HARPS.2020-02-29T05:14:52.221',
             'HARPS.2020-02-29T05:45:23.028',
             'HARPS.2020-03-12T05:12:57.551',
             'HARPS.2020-03-12T06:02:39.146',
             'HARPS.2020-03-13T04:36:00.449',
             'HARPS.2020-03-13T05:06:32.211',
             'HARPS.2020-03-14T06:32:01.044',
             'HARPS.2020-03-14T07:02:32.254',
             'HARPS.2020-03-15T04:16:16.025',
             'HARPS.2020-03-15T04:46:47.164']

# This is a list of dates (i.e. nights), because most nights have more than one exposure
date_list = ['2018-03-29']

# Na I wavelengths in nm in vacuum and air, from https://physics.nist.gov/PhysRefData/ASD/lines_form.html
Na1_vac, Na2_vac = 589.1583264, 589.7558147
Na1_air, Na2_air = 588.995095, 589.592424

# Speed of light
c_light = (c.to('km/s')).value

# System radial velocity in km/s, calculated using PHOENIX (see paper Method Section 1.6)
v_system = 6.01


# Define functions
def find_RV(wave, Na_vac):
    wave = np.array(wave) * u.nm
    RV = wave.to(u.km / u.s, u.doppler_optical(Na_vac * u.nm))
    return RV.value

def find_wave(rv, Na_vac):
    rv = np.array(rv) * u.km / u.s
    wave = rv.to(u.nm, u.doppler_optical(Na_vac * u.nm))
    return wave.value

def norm_flux(wave, flux, fluxerr, chunk_list):
    diff0, diff1 = np.absolute(np.array(wave) - chunk_list[0]), np.absolute(np.array(wave) - chunk_list[-1])
    w_index = [diff0.argmin(), diff1.argmin()]
    w_norm = wave[w_index[0]:w_index[-1]]
    f = flux[w_index[0]:w_index[-1]]
    ferr = fluxerr[w_index[0]:w_index[-1]]

    f_norm, ferr_norm = [], []
    for i, value in enumerate(f):
        value_err = ferr[i]
        f1 = value / (np.median(f))
        f_err = (value_err / (np.median(f)))
        f_norm.append(f1)
        ferr_norm.append(f_err)

    return w_norm, f_norm, ferr_norm

chunk_list = [find_wave(-135, Na1_vac), find_wave(135, Na2_vac)]
chunk_list1 = [find_wave(-135, Na1_vac), find_wave(135, Na1_vac)]
chunk_list2 = [find_wave(-135, Na2_vac), find_wave(135, Na2_vac)]




# Here we make a dictionary of exposure times for each date
time_dict = {}
for date in date_list:
    time_list = []  
    for file in file_list:
        if file[6:16] == date:
            time = file[-12:]
            time_list.append(time)
    time_dict[date] = time_list

wavelength1_date, flux1_date, fluxerr1_date = {}, {}, {}
wavelength2_date, flux2_date, fluxerr2_date = {}, {}, {}

for date in date_list:
    plotname = obj_name + '_' + instrument_name + '_' + date

    wavelength1_time, flux1_time, fluxerr1_time = {}, {}, {}
    wavelength2_time, flux2_time, fluxerr2_time = {}, {}, {}

    for time in time_dict[date]:

        # Read telluric corrected files. Mine are saved in HDF5, but one can modify
        # the following lines to read their own files
        order = 56  # sodium doublet
        h5_path = data_path + 'HARPS.' + date + 'T' + time + '_telluric_corrected.h5'

        with pd.HDFStore(h5_path, 'r') as store:
            df = store[str(order)]
            wavelength = df['wavelength'].values
            flux = df['flux'].values
            fluxerr = df['flux_err'].values
            model = df['model_tel'].values

            # Mask removed lines (interpolated when correcting tellurics)
            removed_df = pd.read_csv('removed_lines.csv')
            file = 'HARPS.' + date + 'T' + time

            if file in removed_df['file'].values:
                removed_df = removed_df.set_index('file')
                wave1_min, wave1_max = removed_df.loc[file, 'wave1_min'], removed_df.loc[file, 'wave1_max']
                wave2_min, wave2_max = removed_df.loc[file, 'wave2_min'], removed_df.loc[file, 'wave2_max']

                mask = ~(((wavelength >= wave1_min) & (wavelength <= wave1_max)) | ((wavelength >= wave2_min) & (wavelength <= wave2_max)))
                wavelength, flux, fluxerr = wavelength[mask], flux[mask], fluxerr[mask]

            mask2 = ~((wavelength >= 590.03) & (wavelength <= 590.15))
            wavelength, flux, fluxerr = wavelength[mask2], flux[mask2], fluxerr[mask2]

        # Here the flux is normalised to the mean value in a wider chunk
        # that includes both doublet lines and some continuum
        wavelength_norm, flux_norm, fluxerr_norm = norm_flux(wavelength, flux, fluxerr, chunk_list)
        wavelength_norm, flux_norm, fluxerr_norm = np.asarray(wavelength_norm), np.asarray(flux_norm), np.asarray(fluxerr_norm)

        # Now split this wide chunk into 2 chunks of spectra
        mask1 = (wavelength_norm >= chunk_list1[0]) & (wavelength_norm <= chunk_list1[1])
        mask2 = (wavelength_norm >= chunk_list2[0]) & (wavelength_norm <= chunk_list2[1])
        wavelength1_norm, flux1_norm, fluxerr1_norm = wavelength_norm[mask1], flux_norm[mask1], fluxerr_norm[mask1]
        wavelength2_norm, flux2_norm, fluxerr2_norm = wavelength_norm[mask2], flux_norm[mask2], fluxerr_norm[mask2]

        wavelength1_time[time], flux1_time[time], fluxerr1_time[time] = wavelength1_norm, flux1_norm, fluxerr1_norm
        wavelength2_time[time], flux2_time[time], fluxerr2_time[time] = wavelength2_norm, flux2_norm, fluxerr2_norm

    wavelength1_date[date], flux1_date[date], fluxerr1_date[date] = wavelength1_time, flux1_time, fluxerr1_time
    wavelength2_date[date], flux2_date[date], fluxerr2_date[date] = wavelength2_time, flux2_time, fluxerr2_time


# Here we will add the data in pairs, to strengthen our signal. This is ok because the
# exocomet lines do not vary significantly between these exposures
wavelength1_pair, wavelength2_pair, flux1_pair, flux2_pair, fluxerr1_pair, fluxerr2_pair = {}, {}, {}, {}, {}, {}

for date in date_list:
    if len(time_dict[date]) <= 2:
        w1_date, w2_date, f1_date, f2_date, ferr1_date, ferr2_date = [], [], [], [], [], []

        for time in time_dict[date]:
            for w1 in wavelength1_date[date][time]:
                w1_date.append(w1)
            for w2 in wavelength2_date[date][time]:
                w2_date.append(w2)
            for f1 in flux1_date[date][time]:
                f1_date.append(f1)
            for f2 in flux2_date[date][time]:
                f2_date.append(f2)
            for ferr1 in fluxerr1_date[date][time]:
                ferr1_date.append(ferr1)
            for ferr2 in fluxerr2_date[date][time]:
                ferr2_date.append(ferr2)

        zipped1 = list(map(list, zip(w1_date, f1_date, ferr1_date)))
        zipped_sorted1 = sorted(zipped1, key=lambda x: x[0])
        zipped2 = list(map(list, zip(w2_date, f2_date, ferr2_date)))
        zipped_sorted2 = sorted(zipped2, key=lambda x: x[0])

        unzipped1 = [list(t) for t in zip(*zipped_sorted1)]
        unzipped2 = [list(t) for t in zip(*zipped_sorted2)]

        wavelength1_pair[date], wavelength2_pair[date] = unzipped1[0], unzipped2[0]
        flux1_pair[date], flux2_pair[date] = unzipped1[1], unzipped2[1]
        fluxerr1_pair[date], fluxerr2_pair[date] = unzipped1[2], unzipped2[2]

    else:
        pairs = [[a, b] for a, b in zip(time_dict[date][::2], time_dict[date][1::2])]
        if len(time_dict[date]) % 2 == 1:
            pairs.append([time_dict[date][-1]])

        for p, pair in enumerate(pairs):
            w1_pair, w2_pair, f1_pair, f2_pair, ferr1_pair, ferr2_pair = [], [], [], [], [], []

            for time in pair:
                for w1 in wavelength1_date[date][time]:
                    w1_pair.append(w1)
                for w2 in wavelength2_date[date][time]:
                    w2_pair.append(w2)
                for f1 in flux1_date[date][time]:
                    f1_pair.append(f1)
                for f2 in flux2_date[date][time]:
                    f2_pair.append(f2)
                for ferr1 in fluxerr1_date[date][time]:
                    ferr1_pair.append(ferr1)
                for ferr2 in fluxerr2_date[date][time]:
                    ferr2_pair.append(ferr2)

            zipped1 = list(map(list, zip(w1_pair, f1_pair, ferr1_pair)))
            zipped_sorted1 = sorted(zipped1, key=lambda x: x[0])
            zipped2 = list(map(list, zip(w2_pair, f2_pair, ferr2_pair)))
            zipped_sorted2 = sorted(zipped2, key=lambda x: x[0])

            unzipped1 = [list(t) for t in zip(*zipped_sorted1)]
            unzipped2 = [list(t) for t in zip(*zipped_sorted2)]

            wavelength1_pair[date + '_' + str(p+1)], wavelength2_pair[date + '_' + str(p+1)] = unzipped1[0], unzipped2[0]
            flux1_pair[date + '_' + str(p+1)], flux2_pair[date + '_' + str(p+1)] = unzipped1[1], unzipped2[1]
            fluxerr1_pair[date + '_' + str(p+1)], fluxerr2_pair[date + '_' + str(p+1)] = unzipped1[2], unzipped2[2]

# We created a new date list because some of the dates have more than 2 exposures.
# Therefore, we separated these nights in separate pairs of exposures
# (e.g. '2018-03-29_1', '2018-03-29_2', and '2018-03-29_3')
new_date_list = wavelength1_pair.keys()

# This line below is just for testing (so Code Ocean only run the fit for
# the first epoch). It may be commented out
new_date_list = ['2018-03-29_1']




# Now we fit the doublet lines in 2 separate chunks
for date in new_date_list:
    from fitting import fit_lines
    from fitting import setup_numpyro

    plotname = obj_name + '_' + instrument_name + '_' + date

    # Import and read parameter files, which contain parameters of the variable absorption lines.
    # These are different for each pair of exposure.
    # Parameters are detailed in fit_lines() in fitting.py
    sys.path.insert(1, output_path)
    import parameters
    importlib.reload(parameters)

    print('NOW FITTING .................... ' + date)

    def my_test(nc, absorption=True, cpu_cores=4):
        setup_numpyro(cpu_cores)

        x1 = wavelength1_pair[date]
        x2 = wavelength2_pair[date]

        y1 = flux1_pair[date]
        y2 = flux2_pair[date]

        yerr1 = np.zeros_like(x1) + fluxerr1_pair[date]
        yerr2 = np.zeros_like(x2) + fluxerr2_pair[date]

        bounds = {}

        # These are the same for all spectra:
        poly = 1
        bounds['dx1'] = [Na2_vac - Na1_vac - 5e-3, Na2_vac - Na1_vac + 5e-3]

        voigt = [1]
        emission = [2]

        m1 = Na1_vac # stellar photospheric absorption
        m2 = Na1_vac # stationary emission line
        m3 = 589.115 # stationary absorption line 1 (blue-most)
        m4 = 589.127 # stationary absorption line 2 (red-most)
        mu1 = [m1 - 0.002, m1 + 0.002] # ~1.0 km/s
        mu2 = [m2 - 0.002, m2 + 0.002]
        mu3 = [m3 - 0.002, m3 + 0.002]
        mu4 = [m4 - 0.002, m4 + 0.002]
        sigma1 = [1, 20]
        sigma2 = [1, 40]
        sigma3 = [0.001, 3]
        sigma4 = [0.001, 3]
        gamma1 = [20, 60]
        gamma2 = [0, 0]
        gamma3 = [0, 0]
        gamma4 = [0, 0]
        A1 = [0.001, 5]
        A2 = [0.001, 5]
        A3 = [0.001, 5]
        A4 = [0.001, 5]
        SR1 = [1, 1]
        SR2 = [1, 1]
        SR3 = [0, 1]
        SR4 = [0, 1]

        osc_ratio = 0.641 / 0.320
        R1 = [osc_ratio, osc_ratio]
        R2 = [osc_ratio, osc_ratio]
        R3 = [osc_ratio, osc_ratio]
        R4 = [osc_ratio, osc_ratio]
        c0 = [1.0, 2.0]
        c1 = [-2e-3, 0]
        c2 = [-1e-6, 0]
        c3 = [-1e-8, 1e8]
        d0 = [1.0, 2.0]
        d1 = [-2e-3, 0]
        d2 = [-1e-6, 0]
        d3 = [-1e-8, 1e8]

        bounds['A1'], bounds['A2'], bounds['A3'], bounds['A4'] = A1, A2, A3, A4
        bounds['mu1'], bounds['mu2'], bounds['mu3'], bounds['mu4'] = mu1, mu2, mu3, mu4
        bounds['sigma1'], bounds['sigma2'], bounds['sigma3'], bounds['sigma4'] = sigma1, sigma2, sigma3, sigma4
        bounds['SR1'], bounds['SR2'], bounds['SR3'], bounds['SR4'] = SR1, SR2, SR3, SR4
        bounds['R1'], bounds['R2'], bounds['R3'], bounds['R4'] = R1, R2, R3, R4
        bounds['c0'], bounds['c1'] = c0, c1
        bounds['d0'], bounds['d1'] = d0, d1
        bounds['gamma1'], bounds['gamma2'], bounds['gamma3'], bounds['gamma4'] = gamma1, gamma2, gamma3, gamma4

        if poly > 1:
            bounds['c2'], bounds['d2'] = c2, d2
            if poly > 2:
                bounds['c3'], bounds['d3'] = c3, d3

        if nc > 4:
            bounds['A5'], bounds['mu5'], bounds['sigma5'], bounds['SR5'], bounds['R5'], bounds['gamma5'] = [0.001, 5], parameters.mu5, parameters.sigma5, [0, 1], [osc_ratio, osc_ratio], [0, 0]
            if nc > 5:
                bounds['A6'], bounds['mu6'], bounds['sigma6'], bounds['SR6'], bounds['R6'], bounds['gamma6'] = [0.001, 5], parameters.mu6, parameters.sigma6, [0, 1], [osc_ratio, osc_ratio], [0, 0]
                if nc > 6:
                    bounds['A7'], bounds['mu7'], bounds['sigma7'], bounds['SR7'], bounds['R7'], bounds['gamma7'] = [0.001, 5], parameters.mu7, parameters.sigma7, [0, 1], [osc_ratio, osc_ratio], [0, 0]
                    if nc > 7:
                        bounds['A8'], bounds['mu8'], bounds['sigma8'], bounds['SR8'], bounds['R8'], bounds['gamma8'] = [0.001, 5], parameters.mu8, parameters.sigma8, [0, 1], [osc_ratio, osc_ratio], [0, 0]

        # Feed these parameters into the fitting function. For the purposes of this example we are
        # assuming the number of cores = 4 and the number or warm-ups and samples = 200. For the
        # fits in the paper, we assumed cpu_cores = 20, nwarmup = 1000, and nsamples = 1000
        fit_lines([x1, x2], [y1, y2], [yerr1, yerr2],
                bounds, cpu_cores=4, oversample=5,
                absorption=True, emission=emission, voigt=voigt, progress_bar=True,
                nwarmup=200, nsamples=200, outdir=output_path, plotname=plotname)

    my_test(nc=4+parameters.file_nc)