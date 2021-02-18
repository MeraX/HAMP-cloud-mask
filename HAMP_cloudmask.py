import xarray
import numpy as np
import collections
import cv2
import datetime
import enum
import os.path

from wgs64_geoid import wgs84_height

"""HAMP cloud masks

Derive HAMP cloud mask products from microwave radiometer retrieval
and cloud radar.

Preparation:
  * make direktory
    ./out/quicklooks
  * adjust input paths in the following three variables at the end of the script
    retrieval_name
    bahamas_name
    radar_name
"""
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


def find_sampling_gaps_narval(data_flag):
    """Find measurement gap in NARVAL 1 and NARVAL 2 unified radar data

    There are gaps in the 1 Hz data every about 30 seconds.
    (Probably they where caused by the radar sampling rate with is slightly > than 1 Hz )

    We can easily find them in NARVAL 1 and 2 data, as
    a) there is  no valid measurement (dBZ == NaN) for one time step and
    b) a gap is no longer than one time step
    """
    gap_mask = (data_flag == Radar_flag.no_measurement).all('height')

    same_as_previous = np.concatenate(([True], gap_mask.values[:-1] == gap_mask[1:]))
    same_as_next = np.concatenate((gap_mask.values[:-1] == gap_mask[1:], [True]))
    gap_mask[same_as_previous | same_as_next] = False # A gap must be only one time step. If if the previous or next were also gaps, remove this gap.
    # Check that there is always no more than one consecutive gap time step
    assert length_of_True_chunks(gap_mask).max() <= 1, 'something went wrong in constraining the gaps'

    return gap_mask


def find_sampling_gaps_eurec4a(data_flag):
    """Find measurement gap in EUREC4A unified radar data

    There are gaps in the 1 Hz data every about 30 seconds.
    (Probably they where caused by the radar sampling rate with is slightly > than 1 Hz )

    Gaps are a bit trickier to find in EUREC4A than in NARVAL.
    see find_sampling_gaps_narval()

    A gab in EUREC4A can only be identified in cloudy scenes.
    If a gap is adjacent in time to a cloud object there are NaN values in the range gates of
    the adjacent cloud in the gap time step. The clear sky above an below the cloud level
    shows the common clear sky value (-888). At the same time, the original data_flag shows 0.
    (I.e. the NaNs are not caused by any other filter like "side lobe".)
    """
    gap_mask = (
        (data_flag == Radar_flag.no_measurement) # either NaN
        ^ (data_flag == Radar_flag.clear) # or "clear sky" (xor)
    ).all('height')

    same_as_previous = np.concatenate(([True], gap_mask.values[:-1] == gap_mask[1:]))
    same_as_next = np.concatenate((gap_mask.values[:-1] == gap_mask[1:], [True]))
    gap_mask[same_as_previous | same_as_next] = False # A gap must be only one time step. If if the previous or next were also gaps, remove this gap.
    # Check that there is always no more than one consecutive gap time step
    assert length_of_True_chunks(gap_mask).max() <= 1, 'something went wrong in constraining the gaps'

    return gap_mask


def make_HAMP_cloudmask(
    retrieval_name,
    bahamas_name,
    radar_name,
    out_name,
):

    clutter_threshold = 4

    ###
    # HAMP_MWR cloud mask
    #
    #0: clear, 1: maybe, 2: certain, -1: unknown
    retrieval = xarray.open_dataset(retrieval_name)

    radiometer_mask = (retrieval.lwp_uc > 0.02).astype(int)
    radiometer_mask += (retrieval.lwp_uc > 0.03).astype(int)
    radiometer_mask[np.isnan(retrieval.lwp_uc)] = Cloud_flag.unknown

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

        # if roll angle > 5, radar gates at altitude >  150 m are affected by sidle lobe echo of the ground
        data_flag.values[(np.abs(bahamas['roll']) > 5)] = Radar_flag.high_roll # high roll angle

        if radar_name[-23:-10] in ('/radar_201312', '/radar_201608'): # NARVAL
            assert np.isneginf(dBZ).any(), 'neginf should be used to mark "measured, but nothing seen, i.e. signal below noise detection" pixels'
            data_flag.values[np.isneginf(dBZ) & (data_flag==Radar_flag.good)] = Radar_flag.clear
            data_flag.values[np.isnan(dBZ) & (data_flag==Radar_flag.good)] = Radar_flag.no_measurement

            gap_mask = find_sampling_gaps_narval(data_flag)
        else: # EUREC4A
            assert not np.isneginf(dBZ).any(), 'neginf should not be used anymore to mark "measured, but nothing seen, i.e. signal below noise detection" pixels'
            radar_raw = xarray.open_dataset(radar_name, mask_and_scale=False)
            assert radar_raw.dBZ.missing_value == -888., '-888. should have been used to mark "measurement, but nothing seen"'
            data_flag.values[(radar_raw.dBZ == radar_raw.dBZ.missing_value) & (data_flag==Radar_flag.good)] = Radar_flag.clear
            assert np.isnan(radar_raw.dBZ._FillValue), '_FillValue should be NaN and is assumed to be used for time gaps in clouds, above the aircraft, and when side lobes are removed in curves'
            data_flag.values[np.isnan(radar_raw.dBZ) & (data_flag==Radar_flag.good)] = Radar_flag.no_measurement

            gap_mask = find_sampling_gaps_eurec4a(data_flag)

        assert  Radar_flag.clear in data_flag.values, 'Could not find any clear sky. That is suspicious. Maybe the identification is wrong?'

        # First look at true data that is not during a time gap to filter for small clouds/clutter pixels
        dBZ_observed = dBZ.isel(time=~gap_mask)
        data_flag_observed = data_flag.isel(time=~gap_mask)

        radar_mask_obseverd = calc_radar_mask(data_flag_observed, clutter_threshold=clutter_threshold)

        # fill measurement gaps in data_flag and radar_mask
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

if __name__ == '__main__':
    import glob
    dates = [
        '20200119',
        '20200122', # no radar
        '20200124',
        '20200126',
        '20200128',
        '20200130',
        '20200131',
        '20200202',
        '20200205',
        '20200207',
        '20200209',
        '20200211',
        '20200213',
        '20200215',
        '20200218',
    ]
    for date in dates:
        retrieval_name=glob.glob(f'EUREC4A_HALO_HAMP_lwpiwv_l2_any_v0.8_{date}[0-9][0-9][0-9][0-9][0-9][0-9].nc')[0]
        bahamas_name=f'bahamas_{date}_v0.9.nc'
        radar_name=f'radar_{date}_v0.9.nc'
        out_name=f'./out/netcdf/EUREC4A_HALO_{{instrument}}_cloud_mask_{date}_v0.9.nc'
        make_HAMP_cloudmask(
            retrieval_name,
            bahamas_name,
            radar_name,
            out_name,
        )
