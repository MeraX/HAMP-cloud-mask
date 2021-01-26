import xarray
import numpy as np
import collections
import cv2
import datetime
from scipy.ndimage import map_coordinates
import enum
import os.path


# this is a 10x10-degree WGS84 geoid datum, in meters relative to the WGS84 reference ellipsoid. given the maximum slope, you should probably interpolate.
# NIMA suggests a 2x2 interpolation using four neighbors. we'll go cubic spline
wgs84_geoid = np.array([[13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13],  # 90N
        [3, 1, -2, -3, -3, -3, -1, 3, 1, 5, 9, 11, 19, 27, 31, 34, 33, 34, 33, 34, 28, 23, 17, 13, 9, 4, 4, 1, -2, -2, 0, 2, 3, 2, 1, 1],  # 80N
        [2, 2, 1, -1, -3, -7, -14, -24, -27, -25, -19, 3, 24, 37, 47, 60, 61, 58, 51, 43, 29, 20, 12, 5, -2, -10, -14, -12, -10, -14, -12, -6, -2, 3, 6, 4],  # 70N
        [2, 9, 17, 10, 13, 1, -14, -30, -39, -46, -42, -21, 6, 29, 49, 65, 60, 57, 47, 41, 21, 18, 14, 7, -3, -22, -29, -32, -32, -26, -15, -2, 13, 17, 19, 6],  # 60N
        [-8, 8, 8, 1, -11, -19, -16, -18, -22, -35, -40, -26, -12, 24, 45, 63, 62, 59, 47, 48, 42, 28, 12, -10, -19, -33, -43, -42, -43, -29, -2, 17, 23, 22, 6, 2],  # 50N
        [-12, -10, -13, -20, -31, -34, -21, -16, -26, -34, -33, -35, -26, 2, 33, 59, 52, 51, 52, 48, 35, 40, 33, -9, -28, -39, -48, -59, -50, -28, 3, 23, 37, 18, -1, -11],  # 40N
        [-7, -5, -8, -15, -28, -40, -42, -29, -22, -26, -32, -51, -40, -17, 17, 31, 34, 44, 36, 28, 29, 17, 12, -20, -15, -40, -33, -34, -34, -28, 7, 29, 43, 20, 4, -6],  # 30N
        [5, 10, 7, -7, -23, -39, -47, -34, -9, -10, -20, -45, -48, -32, -9, 17, 25, 31, 31, 26, 15, 6, 1, -29, -44, -61, -67, -59, -36, -11, 21, 39, 49, 39, 22, 10],  # 20N
        [13, 12, 11, 2, -11, -28, -38, -29, -10, 3, 1, -11, -41, -42, -16, 3, 17, 33, 22, 23, 2, -3, -7, -36, -59, -90, -95, -63, -24, 12, 53, 60, 58, 46, 36, 26],  # 10N
        [22, 16, 17, 13, 1, -12, -23, -20, -14, -3, 14, 10, -15, -27, -18, 3, 12, 20, 18, 12, -13, -9, -28, -49, -62, -89, -102, -63, -9, 33, 58, 73, 74, 63, 50, 32],  # 0
        [36, 22, 11, 6, -1, -8, -10, -8, -11, -9, 1, 32, 4, -18, -13, -9, 4, 14, 12, 13, -2, -14, -25, -32, -38, -60, -75, -63, -26, 0, 35, 52, 68, 76, 64, 52],  # 10S
        [51, 27, 10, 0, -9, -11, -5, -2, -3, -1, 9, 35, 20, -5, -6, -5, 0, 13, 17, 23, 21, 8, -9, -10, -11, -20, -40, -47, -45, -25, 5, 23, 45, 58, 57, 63],  # 20S
        [46, 22, 5, -2, -8, -13, -10, -7, -4, 1, 9, 32, 16, 4, -8, 4, 12, 15, 22, 27, 34, 29, 14, 15, 15, 7, -9, -25, -37, -39, -23, -14, 15, 33, 34, 45],  # 30S
        [21, 6, 1, -7, -12, -12, -12, -10, -7, -1, 8, 23, 15, -2, -6, 6, 21, 24, 18, 26, 31, 33, 39, 41, 30, 24, 13, -2, -20, -32, -33, -27, -14, -2, 5, 20],  # 40S
        [-15, -18, -18, -16, -17, -15, -10, -10, -8, -2, 6, 14, 13, 3, 3, 10, 20, 27, 25, 26, 34, 39, 45, 45, 38, 39, 28, 13, -1, -15, -22, -22, -18, -15, -14, -10],  # 50S
        [-45, -43, -37, -32, -30, -26, -23, -22, -16, -10, -2, 10, 20, 20, 21, 24, 22, 17, 16, 19, 25, 30, 35, 35, 33, 30, 27, 10, -2, -14, -23, -30, -33, -29, -35, -43],  # 60S
        [-61, -60, -61, -55, -49, -44, -38, -31, -25, -16, -6, 1, 4, 5, 4, 2, 6, 12, 16, 16, 17, 21, 20, 26, 26, 22, 16, 10, -1, -16, -29, -36, -46, -55, -54, -59],  # 70S
        [-53, -54, -55, -52, -48, -42, -38, -38, -29, -26, -26, -24, -23, -21, -19, -16, -12, -8, -4, -1, 1, 4, 4, 6, 5, 4, 2, -6, -15, -24, -33, -40, -48, -50, -53, -52],  # 80S
        [-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30]],  # 90S
    dtype=np.float)


# ok this calculates the geoid offset from the reference ellipsoid
def wgs84_height(lon, lat):
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    xi = (18 + lon / 10.0) % 36
    yi = (9. - lat / 10.0)

    return np.array(map_coordinates(wgs84_geoid, [yi, xi]), dtype=np.float)


@enum.unique
class Radar_flag(enum.IntEnum):
    clear = -1 # clear sky signal
    good = 0 # good signal
    noise = 1
    surface = 2
    sea = 3
    calibration = 4
    side_lobes = 5
    no_measurement = 6
    high_roll = 7

@enum.unique
class Cloud_flag(enum.IntEnum):
    unknown = -1
    clear = 0
    probably = 1
    certain = 2


def calc_radar_mask(data_flag, clutter_threshold=4):
    assert data_flag.dims == ('time', 'height')

    #0: clear, 1: maybe, 2: certain, -1: unknown
    radar_mask = xarray.full_like(data_flag.time, Cloud_flag.clear, dtype=int)
    radar_mask[(data_flag == Radar_flag.good).any('height')] = Cloud_flag.probably

    cloud_mask2d = (data_flag == Radar_flag.good).transpose('time', 'height').values
    cloud_mask8bit = cloud_mask2d.astype(np.uint8)

    number, markers = cv2.connectedComponents(cloud_mask8bit)

    assert np.all((markers==0) == (cloud_mask2d==0)), 'Cloud free should be marked with 0. This seems not to be the case.'

    IDs, count = np.unique(markers, return_counts=True)
    IDs = IDs[1:] # the first ID is the background, not needed here
    count = count[1:] # the first ID is the background, not needed here
    potential_clouds = np.where(markers != 0) # indices of potential cloudy pixels

    #remove clouds that are smaller than n pixel
    invalid_cloud_IDs = set(IDs[count < clutter_threshold]) # set of echos that are too small and that are likely to be clutter
    for x, y in zip(*potential_clouds):
        if markers[x,y] in invalid_cloud_IDs:
            markers[x,y] = 0 # mark as cloud free

    certainly_cloudy = (markers != 0).any(axis=1)
    assert (radar_mask[certainly_cloudy] == Cloud_flag.probably).all()
    radar_mask[certainly_cloudy] = Cloud_flag.certain

    no_echo_signal = (data_flag != Radar_flag.good)
    no_clear_signal = (data_flag != Radar_flag.clear)
    radar_mask[(no_echo_signal & no_clear_signal).all('height')] = Cloud_flag.unknown

    return radar_mask


def calc_radar_top_height(radar_mask, data_flag):
    assert radar_mask.dims == ('time',)
    assert data_flag.dims == ('time', 'height')

    # Reverse height, such that argmax finds the first (probably ) cloudy pixel from the top
    data_flag_r = data_flag.isel(height=np.s_[::-1])
    pc_pixel = (data_flag_r==Radar_flag.good).where(
        (radar_mask == Cloud_flag.probably) | # probably
        (radar_mask == Cloud_flag.certain), # certain pixe
        drop=True
    )
    assert np.all(pc_pixel.height.diff('height') < 0), 'dBZ.height is not strong monotonously decreasing.'
    index_of_cloud_top = pc_pixel.argmax('height')

    cloud_top_height = pc_pixel.height[index_of_cloud_top]

    return cloud_top_height.reindex_like(radar_mask, fill_value=np.nan)


def length_of_True_chunks(mask):
    """Count the length of each chunk of True.

    Counts the length of consecutive "True" values in mask.

    Parameters
    ----------
    mask : {ndarray (dtype=bool)}
        Boolean vector

    Returns
    -------
    ndarray (dtype=int)
        Vector with the length of each chunk of False entries
    """
    mask = np.asarray(mask)
    assert mask.dtype == bool
    assert len(mask.shape) == 1

    cc = np.concatenate((
        [mask[0]], # add start index 0 to np.where if first entry is True
        mask[:-1] != mask[1:], # where does it switch between True and False?
        [True] # add end index. This is only considered by the [::2] slice if it belongs to and False chunk
    ))
    return np.diff(np.where(cc)[0])[::2] # length of phases between changes to False and than to True


def make_HAMP_cloudmask(
    retrieval_name,
    bahamas_name,
    radar_name,
    out_name,
):

    clutter_threshold = 4
    retrieval = xarray.open_dataset(retrieval_name)

    ###
    # HAMP_MWR cloud mask
    #
    #0: clear, 1: maybe, 2: certain, -1: unknown
    radiometer_mask = (retrieval.lwp_hamp > 0.02).astype(int)
    radiometer_mask += (retrieval.lwp_hamp > 0.03).astype(int)
    radiometer_mask[np.isnan(retrieval.lwp_hamp)] = Cloud_flag.unknown

    ###
    # HAMP_RADAR cloud mask
    #
    bahamas = xarray.open_dataset(bahamas_name)

    radar = xarray.open_dataset(radar_name)
    assert np.allclose((radar.time - retrieval.time.values)/np.timedelta64(1, 's'), 0, atol=1)
    radar.assign_coords(time=retrieval.time) # fix issue with time rounding
    assert radar.height.units == 'm'

    if '20200122' in radar_name: # radar was not working
        radar_cloud_top_height = xarray.full_like(radiometer_mask, np.nan, dtype=float)
        radar_mask = xarray.full_like(radiometer_mask, Cloud_flag.unknown, dtype=float)
    else:
        radar = radar.sel(height=np.s_[200:])

        dBZ = radar.dBZ.copy()
        data_flag = radar.data_flag.copy()
        assert (
            (radar.data_flag.long_name == '1: noise; 2: surface; 3: sea; 4: radar calibration; 5: side lobes removed') or
            (radar.data_flag.long_name == '1: noise; 2: surface; 3: sea; 4: radar calibration')
        ), radar.data_flag.long_name

        if radar_name[-23:-10] in ('/radar_201312', '/radar_201608'): # NARVAL
            assert np.isneginf(dBZ).any(), 'neginf should be used to mark "measured, but nothing seen, i.e. signal below noise detection" pixels'
            data_flag.values[np.isneginf(dBZ) & (data_flag==Radar_flag.good)] = Radar_flag.clear
        else: # EUREC4A
            assert not np.isneginf(dBZ).any(), 'neginf should not be used anymore to mark "measured, but nothing seen, i.e. signal below noise detection" pixels'
            radar_raw = xarray.open_dataset(radar_name, mask_and_scale=False)
            assert radar_raw.dBZ.missing_value == -888., '-888. should have been used to mark "measurement, but nothing seen"'
            data_flag.values[(radar_raw.dBZ == radar_raw.dBZ.missing_value) & (data_flag==Radar_flag.good)] = Radar_flag.clear

        assert  Radar_flag.clear in data_flag.values, 'Could not find any clear sky. That is suspicious. Maybe the identification is wrong?'

        # if roll angle > 5, radar gates at altitude >  150 m are affected by sidle lobe echo of the ground
        data_flag.values[(np.abs(bahamas['roll']) > 5)] = Radar_flag.high_roll # high roll angle

        data_flag.values[np.isnan(dBZ) & (data_flag==0)] = Radar_flag.no_measurement
        # There are gaps in the 1 Hz data every about 30 seconds. (Probably they where caused by the radar sampling rate with is slightly > than 1 Hz )
        # Filter for small clouds/clutter pixels, than fix these gaps!
        gap_mask = (data_flag == Radar_flag.no_measurement).all('height') # measurement 'gaps' in 1 Hz time series

        same_as_previous = np.concatenate(([True], gap_mask.values[:-1] == gap_mask[1:]))
        same_as_next = np.concatenate((gap_mask.values[:-1] == gap_mask[1:], [True]))
        gap_mask[same_as_previous | same_as_next] = False # A gap must be only one time step. If if the previous or next were also gaps, remove this gap.
        assert length_of_True_chunks(gap_mask).max() <= 1, 'something went wrong in constraining the gaps'

        dBZ_observed = dBZ.isel(time=~gap_mask)
        data_flag_observed = data_flag.isel(time=~gap_mask)

        radar_mask_obseverd = calc_radar_mask(data_flag_observed, clutter_threshold=clutter_threshold)

        # fill measurement gaps in dBZ and radar_mask
        dBZ_interpolated = dBZ_observed.interp(time=dBZ.time, method='nearest',
            kwargs=dict(fill_value=np.nan))

        data_flag = data_flag_observed.interp(time=dBZ.time, method='nearest',
            kwargs=dict(fill_value=Radar_flag.no_measurement))
        data_flag = data_flag.astype(int)
        assert set(Radar_flag).issuperset(np.unique(data_flag)), 'data_flag contains some non-Radar_flag values.'

        radar_mask = radar_mask_obseverd.interp(time=dBZ.time, method='nearest',
            kwargs=dict(fill_value=Cloud_flag.unknown))
        radar_mask = radar_mask.astype(int)
        assert set(Cloud_flag).issuperset(np.unique(radar_mask)), 'radar_mask contains some non-Cloud_flag values.'

        radar_top = calc_radar_top_height(radar_mask, data_flag)
        radar_cloud_top_height = radar_top - wgs84_height(radar_mask.lon, radar_mask.lat)

    ###
    # export as NC
    #

    radar_mask_ds = xarray.Dataset(
        {'time': bahamas.time}
        # subsequent Arrays are added separately to have better control on the order
    )
    radar_mask_ds.time.encoding['units'] = 'seconds since 2020-01-01 00:00:00'
    radar_mask_ds.time.attrs['long_name'] = 'time'

    radar_mask_ds['lat'] = bahamas.lat
    radar_mask_ds.lat.attrs['units'] = 'degree_north' ;
    radar_mask_ds.lat.attrs['standard_name'] = 'latitude' ;
    radar_mask_ds.lat.attrs['long_name'] = 'latitude' ;
    radar_mask_ds.lat.attrs['description'] = 'Platform latitude coordinate'
    radar_mask_ds['lon'] = bahamas.lon
    radar_mask_ds.lon.attrs['units'] = 'degree_east' ;
    radar_mask_ds.lon.attrs['standard_name'] = 'longitude' ;
    radar_mask_ds.lon.attrs['long_name'] = 'longitude' ;
    radar_mask_ds.lon.attrs['description'] = 'Platform longitude coordinate'

    radar_mask_ds['cloud_top'] = ['time'], radar_cloud_top_height.values, dict(
        long_name='cloud top height above sea level',
        standard_name='height_at_cloud_top',
        units='m',
        description=(
            'For each time step at which a cloud is (probably) detected, this variable reports the '
            + 'height of the upper most range gate having a signal above noise level.'
        ),
    )
    radar_mask_ds['cloud_mask'] = ['time'], radar_mask.values.astype(np.int8), dict(
        long_name='cloud flag',
        flag_values=np.array(
            [Cloud_flag.unknown,
            Cloud_flag.clear, Cloud_flag.probably, Cloud_flag.certain],
            dtype=np.int8
        ),
        flag_meanings='unknown no_cloud_detectable probably_cloudy most_likely_cloudy',
        description=(
            'For this mask the observations of radar reflectivity are used. Radar reflectivity is '
            + 'first filtered for clutter. Then if there is any signal above the noise level at '
            + '200 m above sea level or '
            + 'above, this is considered either cloud or still clutter. If signal originates from '
            + f'an object of at least {clutter_threshold:d} pixels size it is most likely, '
            + 'otherwise probably a cloud signal.'
        ),
    )

    encoding = {k: {'zlib':True, 'fletcher32':True} for k in radar_mask_ds.variables}
    encoding['time']['units'] = 'seconds since 2020-01-01 00:00:00'
    assert np.all(np.isfinite(radar_mask_ds.time))
    encoding['time']['_FillValue'] = None
    assert np.all(np.isfinite(radar_mask_ds.lat))
    encoding['lat']['_FillValue'] = None
    assert np.all(np.isfinite(radar_mask_ds.lon))
    encoding['lon']['_FillValue'] = None

    # Some meta data
    attrs = collections.OrderedDict()
    #attrs['convention'] = 'CF-1.7'
    attrs['title'] = 'HAMP Radar Cloud Mask'
    attrs['version'] = "0.1" ;
    attrs['contact'] = "marek.jacob@uni-koeln.de" ;
    #attrs['comment'] = "" ;
    attrs['platform'] = "HALO" ;
    attrs['campaign'] = "EUREC4A"
    attrs['variable'] = "cloud_flag"
    attrs['instrument'] = 'HAMP Radar'
    attrs['institution'] = 'Institute for Geophysics and Meteorology, University of Cologne'
    attrs['author'] = 'Marek Jacob'
    attrs['history'] = (
        datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S') + ' UTC: created'
    )
    attrs['source'] = (
        'From Cloud radar METEK MIRA35, based on ' +
        ' '.join(os.path.basename(s) for s in (radar_name, bahamas_name))
    )
    attrs['created_on'] = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S') + ' UTC'
    attrs['Conventions'] = 'CF-1.8'
    #attrs['doi'] = ''
    attrs['featureType'] = 'trajectory'
    #attrs['research_flight_date'] = str(radar_mask_ds['time'].min().dt.strftime('%Y-%m-%d').values)
    #attrs['citation'] = 'please contact the authors if you want to use the data for publications'
    #attrs['max_latitude'] = radar_mask_ds['lat'].max().values
    #attrs['min_latitude'] = radar_mask_ds['lat'].min().values
    #attrs['max_longitude'] = radar_mask_ds['lon'].max().values
    #attrs['min_longitude'] = radar_mask_ds['lon'].min().values
    #attrs['start_datetime'] = str(radar_mask_ds['time'].min().dt.strftime('%Y-%m-%dT%H:%M:%S').values)
    #attrs['stop_datetime'] = str(radar_mask_ds['time'].max().dt.strftime('%Y-%m-%dT%H:%M:%S').values)
    radar_mask_ds.attrs.update(attrs)

    radar_mask_ds.to_netcdf(out_name.format(instrument='HAMP-Radar'), format='NETCDF4', encoding=encoding)

    radiometer_mask_ds = xarray.Dataset(
        {'time': radar_mask_ds.time}
    )
    radiometer_mask_ds['lat'] = radar_mask_ds.lat
    radiometer_mask_ds['lon'] = radar_mask_ds.lon

    rmv = radiometer_mask.values
    rmv[radiometer_mask_ds.lat>20] = Cloud_flag.unknown # We don't use the LWP retrieval north of 20 deg N.
    radiometer_mask_ds['cloud_mask'] = ['time'], rmv.astype(np.int8), dict(
        long_name='cloud flag',
        flag_values=np.array(
            [Cloud_flag.unknown, Cloud_flag.clear, Cloud_flag.probably, Cloud_flag.certain],
            dtype=np.int8
        ),
        flag_meanings='unknown no_cloud_detectable probably_cloudy most_likely_cloudy',
        description=(
            'For this mask liquid water path (LWP) retrieval by Jacob et al. (2019, AMT, '
            + 'https://doi.org/10.5194/amt-12-3237-2019) is used but without applying the '
            + 'clear-sky offset adjustment using WALES backscatter lidar data. Thresholds of 20 '
            + 'and 30 g.m^-2 are used to detect clouds at medium and high levels of confidence.'
        ),
    )

    encoding = {k: {'zlib':True, 'fletcher32':True} for k in radiometer_mask_ds.variables}
    encoding['time']['units'] = 'seconds since 2020-01-01 00:00:00'
    encoding['time']['_FillValue'] = None
    encoding['lat']['_FillValue'] = None
    encoding['lon']['_FillValue'] = None
    attrs['instrument'] = 'HAMP Microwave Radiometer'
    attrs['source'] = (
        'From Three RPG HALO MWR modules, based on ' +
        ' '.join(os.path.basename(s) for s in (retrieval_name, bahamas_name))
    )
    attrs['title'] = 'HAMP Microwave Radiometer Cloud Mask'
    attrs['references'] = 'Jacob et al. (2019, https://doi.org/10.5194/amt-12-3237-2019 )'
    radiometer_mask_ds.attrs.update(attrs)

    radiometer_mask_ds.to_netcdf(out_name.format(instrument='HAMP-MWR'), format='NETCDF4', encoding=encoding)

dates = [
    #'20200119', # TODO: when retrieval is redone after solving the time offset in WF.
    '20200122', # no radar
    '20200124',
    '20200126',
    '20200128',
    '20200130',
    '20200131', # TODO: Half of radar is missing
    '20200202',
    '20200205',
    '20200207',
    '20200209',
    '20200211',
    '20200213',
    '20200215', # alto strato flight at flight levels the LWP was not trained for. further, the alto is not really shallow
    '20200218', # ferry home
]
for date in dates:
    retrieval_name=f'/home/mjacob/data/EUREC4A/LWP_IWV/EUREC4A_HAMP-MWR_lwp_iwv_{date}_v0.4.0.1_2021-01-25.nc'
    bahamas_name=f'/data/hamp/flights/EUREC4A/unified/bahamas_{date}_v0.6.nc'
    radar_name=f'/data/hamp/flights/EUREC4A/unified/v0.6.1/radar_{date}_v0.6.nc'
    out_name=f'./out/netcdf/EUREC4A_HALO_{{instrument}}_cloud_mask_{date}_v0.6.1.nc'
    make_HAMP_cloudmask(
        retrieval_name,
        bahamas_name,
        radar_name,
        out_name,
    )
