import xarray
import numpy as np
import matplotlib.pyplot as plt
import cv2
import datetime
from scipy.ndimage import map_coordinates
import enum
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


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


def center_to_edge(center):
    center = np.asarray(center)
    edge =  np.empty(center.shape[0]+1, dtype=center.dtype)
    # the strange notation in the following lines is used to make the calculations datetime compatible
    edge[1:-1] = center[1:] + (center[:-1] - center[1:])/2.
    edge[0] = edge[1] + (edge[1] - edge[2])
    edge[-1] = edge[-2] + (edge[-2] - edge[-3])
    return edge

dates = [
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
    #'20200218', # ferry home, no masks
]
for date in dates:

    radar_ds = xarray.open_dataset(f'./out/netcdf/HAMP_RADAR_cloud_mask_{date}.nc')
    radiometer_ds = xarray.open_dataset(f'./out/netcdf/HAMP_MWR_cloud_mask_{date}.nc')

    radar = xarray.open_dataset(f'/data/hamp/flights/EUREC4A/unified/radar_{date}_v0.6.nc')
    assert np.allclose((radar.time - radar_ds.time.values)/np.timedelta64(1, 's'), 0, atol=1)
    radar.assign_coords(time=radar_ds.time) # fix issue with time rounding

    wales = xarray.open_dataset(f'/data/hamp/flights/EUREC4A/{date}/WALES-LIDAR/EUREC4A_HALO_WALES_cloudtop_{date}a_V1.nc')
    wales_flag_hamp = wales.cloud_flag.rolling(time=5).mean().interp_like(radar_ds.time)
    wales_top_hamp = wales.cloud_top.rolling(time=5).mean().interp_like(radar_ds.time)
    wales_ot_hamp = wales.cloud_ot.rolling(time=5).mean().interp_like(radar_ds.time)
    wales_on_hamp = wales.interp_like(radar_ds.time)


    ###
    # plot time series
    #
    fig, (ax, ax2) = plt.subplots(
        nrows=2, figsize=(100, 8), sharex=True,
        gridspec_kw=dict(height_ratios=[1, 2]),
    )
    ax.plot(wales_flag_hamp.time, wales_flag_hamp, '-', label='wales 1 s average')
    ax.plot(radar_ds.time, radar_ds.cloud_flag + 0.1, '.', label='radar')
    ax.plot(radiometer_ds.time, radiometer_ds.cloud_flag, '.', label='radiometer')
    ax.set_yticks(radar_ds.cloud_flag.flag_values)
    ax.set_yticklabels(radar_ds.cloud_flag.flag_meanings.split())
    ax.legend()
    ax.grid()

    dBZ = radar.dBZ.transpose('height', 'time').values
    x = center_to_edge(radar.time)
    y = center_to_edge(radar.height)
    ax2.pcolormesh(x, y, dBZ, vmin=-40, vmax=20, cmap='gray')
    ax2.plot(
        wales_top_hamp.time, wales_top_hamp.where(wales_ot_hamp>1) + wgs84_height(radar_ds.lon, radar_ds.lat),
        linewidth=0.5
    )
    ax2.plot(
        radar_ds.time, radar_ds.cloud_top + wgs84_height(radar_ds.lon, radar_ds.lat),
        linewidth=0.5
    )
    ax2.set_ylabel('Height above WGS84 (m)')
    ax2.set_ylim(0, 4500)
    ax2.grid()
    fig.tight_layout()
    fig.savefig(f'./out/quicklooks/timeseries_{date}.png', dpi=200)
    plt.close(fig)


    ###
    # plot histograms
    #
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
    h, xedges, yedges, QM = axes[0].hist2d(
        radiometer_ds.cloud_flag.values, radar_ds.cloud_flag.values,
        bins=[-1.5, -0.5, 0.5, 1.5, 2.5], vmin=0, vmax=1, density=True,
    )
    # print values into matrix plot
    x = (xedges[:-1] + xedges[1:])/2
    y = (yedges[:-1] + yedges[1:])/2
    for ii in range(len(x)):
        for jj in range(len(y)):
            axes[0].text(x[ii], y[jj], '%.2f' % h[ii, jj],
                color='#ffffff',
                horizontalalignment='center',
                verticalalignment='center',
            )
    axes[0].set_xlabel('radiometer')
    axes[0].set_xticks(radiometer_ds.cloud_flag.flag_values)
    axes[0].set_xticklabels(radiometer_ds.cloud_flag.flag_meanings.split())
    axes[0].set_ylabel('radar')
    axes[0].set_yticks(radar_ds.cloud_flag.flag_values)
    axes[0].set_yticklabels(radar_ds.cloud_flag.flag_meanings.split())

    finite = np.isfinite(radar_ds.cloud_top + wales_top_hamp.values)
    axes[1].hist((radar_ds.cloud_top-wales_top_hamp.values)[finite], bins=100, density=True)

    axins = inset_axes(axes[1], width='50%', height='70%', loc=1)
    axins.hist((radar_ds.cloud_top-wales_top_hamp.values)[finite], bins=np.linspace(-100, 100, 60), density=True)

    axins.set_xlim(-100, 100)
    axins.set_xticks([-100, 0, 100])
    # fix the number of ticks on the inset axes
    axins.yaxis.get_major_locator().set_params(nbins=7)
    axins.axvline(0, color='#cccccc')

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(axes[1], axins, loc1=3, loc2=4, fc="none", ec="#cccccc")

    axes[1].set_xlabel('radar top height - lidar top height')

    fig.tight_layout()
    fig.savefig(f'./out/quicklooks/stats_{date}.png', dpi=200)
    plt.close(fig)

    # fig, ax = plt.subplots()
    # for flag in Cloud_flag:
    #     wales_ot_hamp.where(radar_mask==flag).plot.hist(ax=ax, bins=40, label=flag.name, histtype='step')
    # ax.legend()
    #plt.show()
    break




import sys; sys.exit(66)
###########
fig, axes = plt.subplots(nrows=2, sharex=True)
axes[0].plot(bahamas['roll'])
axes[1].imshow(radar.dBZ.where(radar_mask != -1).values.T, vmin=-30, vmax=20, aspect='auto', origin='lower')
axes[0].grid(); axes[1].grid();