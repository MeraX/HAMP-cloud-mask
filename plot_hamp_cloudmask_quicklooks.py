import xarray
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from wgs64_geoid import wgs84_height

"""
Make quick look plots of the HAMP cloudmasks
Compare HAMP MWR, Radar and WALES

Preparation:
  * make direktory
    ./out/quicklooks
  * adjust paths:
    unified_data_dir
    WALES_data_dir
    in 'open_dataset' for 'radar_cm' and 'radiometer_cm'
"""

unified_data_dir = ""
WALES_data_dir = ""

def center_to_edge(center):
    center = np.asarray(center)
    edge =  np.empty(center.shape[0]+1, dtype=center.dtype)
    # the strange notation in the following lines is used to make the calculations datetime compatible
    edge[1:-1] = center[1:] + (center[:-1] - center[1:])/2.
    edge[0] = edge[1] + (edge[1] - edge[2])
    edge[-1] = edge[-2] + (edge[-2] - edge[-3])
    return edge

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
for date in dates[8:9]:

    ###
    # Open data locally
    #
    cloud_mask_version = '0.9'
    radar_cm = xarray.open_dataset(f'./out/netcdf/EUREC4A_HALO_HAMP-Radar_cloud_mask_{date}_v{cloud_mask_version}.nc')
    radiometer_cm = xarray.open_dataset(f'./out/netcdf/EUREC4A_HALO_HAMP-MWR_cloud_mask_{date}_v{cloud_mask_version}.nc')

    radar = xarray.open_dataset(f'{unified_data_dir}/radar_{date}_v0.9.nc')
    assert np.allclose((radar.time - radar_cm.time.values)/np.timedelta64(1, 's'), 0, atol=1)
    radar.assign_coords(time=radar_cm.time) # fix issue with time rounding

    if date in ('20200119', '20200218'):
        # No WALES files available
        wales_flag_hamp = xarray.full_like(radiometer_cm.cloud_mask, np.nan, dtype=float)
        wales_top_hamp = xarray.full_like(radiometer_cm.cloud_mask, np.nan, dtype=float)
        wales_ot_hamp = xarray.full_like(radiometer_cm.cloud_mask, np.nan, dtype=float)
    else:
        wales = xarray.open_dataset(f'{WALES_data_dir}/EUREC4A_HALO_WALES_cloudtop_{date}a_V1.nc')
        # average WALES on HAMP 1 Hz time grid.
        wales_flag_hamp = wales.cloud_flag.rolling(time=5).mean().interp_like(radar_cm.time)
        wales_top_hamp = wales.cloud_top.rolling(time=5).mean().interp_like(radar_cm.time)
        wales_ot_hamp = wales.cloud_ot.rolling(time=5).mean().interp_like(radar_cm.time)
        wales_on_hamp = wales.interp_like(radar_cm.time)


    ###
    # plot time series
    #
    fig, (ax, ax2) = plt.subplots(
        nrows=2, figsize=(100, 8), sharex=True,
        gridspec_kw=dict(height_ratios=[1, 2]),
    )
    ax.plot(wales_flag_hamp.time, wales_flag_hamp, '-', label='WALES 1 s average')
    ax.plot(radar_cm.time, radar_cm.cloud_mask + 0.1, '.', label='Radar')
    ax.plot(radiometer_cm.time, radiometer_cm.cloud_mask, '.', label='Radiometer')
    ax.set_yticks(radar_cm.cloud_mask.flag_values)
    ax.set_yticklabels(radar_cm.cloud_mask.flag_meanings.split())
    ax.legend()
    ax.grid()

    if date != '20200122': # radar was broken
        dBZ = radar.dBZ.transpose('height', 'time').values
        x = center_to_edge(radar.time)
        y = center_to_edge(radar.height)
        ax2.pcolormesh(x, y, dBZ, vmin=-40, vmax=20, cmap='gray')
    ax2.plot(
        wales_top_hamp.time, wales_top_hamp.where(wales_ot_hamp>1) + wgs84_height(radar_cm.lon, radar_cm.lat),
        '.', markersize=3,
        label='WALES cloud top'
    )
    ax2.plot(
        radar_cm.time, radar_cm.cloud_top + wgs84_height(radar_cm.lon, radar_cm.lat),
        '.', markersize=3,
        label='Radar Cloud top'

    )
    ax2.set_ylabel('Height above WGS84 (m)')
    ax2.set_ylim(0, 4500)
    ax2.grid()
    fig.tight_layout()
    fig.savefig(f'./out/quicklooks/timeseries_{date}_v{cloud_mask_version}.png', dpi=200)
    plt.close(fig)


    ###
    # plot histograms
    #
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
    h, xedges, yedges, QM = axes[0].hist2d(
        radiometer_cm.cloud_mask.values, radar_cm.cloud_mask.values,
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
    axes[0].set_xticks(radiometer_cm.cloud_mask.flag_values)
    axes[0].set_xticklabels(radiometer_cm.cloud_mask.flag_meanings.split())
    axes[0].set_ylabel('radar')
    axes[0].set_yticks(radar_cm.cloud_mask.flag_values)
    axes[0].set_yticklabels(radar_cm.cloud_mask.flag_meanings.split())

    finite = np.isfinite(radar_cm.cloud_top + wales_top_hamp.values)
    axes[1].hist((radar_cm.cloud_top-wales_top_hamp.values)[finite], bins=100, density=True)

    axins = inset_axes(axes[1], width='50%', height='70%', loc=1)
    axins.hist((radar_cm.cloud_top-wales_top_hamp.values)[finite], bins=np.linspace(-100, 100, 60), density=True)

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
    fig.savefig(f'./out/quicklooks/stats_{date}_v{cloud_mask_version}.png', dpi=200)
    plt.close(fig)

