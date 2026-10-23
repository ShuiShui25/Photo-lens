"""Scientific plots shared by notebook and web workflows."""
from .workflow import ROOT, system_record

def plot_regions(name, folder=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch
    from regions import Regions
    import json
    record = system_record(name)
    cfg = json.loads((ROOT / record['model_config']).read_text())
    base = ROOT / '1.lens-modeling' / name
    with fits.open((base / cfg['science_cutout']).resolve()) as h:
        data = np.asarray(h[cfg['science_hdu']].data, float)
        wcs = WCS(h[cfg['science_hdu']].header).celestial
    # WCS tick-label extents can make tight_layout collapse the plotting area
    # for some sky positions. Reserve fixed space for axes and an outer legend.
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection':wcs})
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.14, top=0.84)
    ax.imshow(data, origin='lower', cmap='gray', norm=ImageNormalize(data, interval=PercentileInterval(99.5), stretch=AsinhStretch()))
    folder = folder or ROOT / '2.region-detection/regions' / name
    from matplotlib.lines import Line2D
    handles, outlines, pixel_regions = [], [], []
    selections = [(folder / f'{name}_{kind}_point_fraction040.reg', color, label)
                  for kind, color, label in [('lens','#ff75cc','Lens'),('source','#4ceaff','Source'),('background','#ffd166','Symmetric background')]]
    satellite_path = folder / f'{name}_satellites_point_fraction.reg'
    if satellite_path.is_file():
        selections.append((satellite_path, '#ff675f', 'Satellite'))
    for path, color, label in selections:
        regions = list(Regions.read(path, format='ds9'))
        if not regions:
            continue
        for region in regions:
            pixel_region = region.to_pixel(wcs)
            pixel_regions.append(pixel_region)
            outlines.append(pixel_region.plot(ax=ax, facecolor='none', edgecolor=color, linewidth=2))
        handles.append(Line2D([0],[0],color=color,label=label,lw=2))
    x,y = wcs.world_to_pixel_values(record['ra'], record['dec'])
    half_size = 35.0
    for region in pixel_regions:
        box = region.bounding_box
        half_size = max(half_size, abs(box.ixmin-x)+4, abs(box.ixmax-x)+4,
                        abs(box.iymin-y)+4, abs(box.iymax-y)+4)
    ax.set_xlim(x-half_size,x+half_size);ax.set_ylim(y-half_size,y+half_size)
    for coordinate in ax.coords:
        coordinate.set_ticks(number=4)
        coordinate.set_ticklabel(size=10, exclude_overlapping=True)
    ax.set_xlabel('Right ascension');ax.set_ylabel('Declination')
    fig.suptitle(name, y=0.98)
    legend = fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.55,0.945),
                        ncol=2, facecolor='#172033', labelcolor='white', fontsize=10)
    fig.savefig(folder / f'{name}_regions.png',dpi=170)
    for outline in outlines:
        outline.set_visible(False)
    legend.set_visible(False)
    fig.savefig(folder / f'{name}_regions_no_outlines.png',dpi=170);plt.close(fig)

def plot_photoz(name):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    folder = ROOT / '3.photometric-estimation/results' / name
    z = pd.read_csv(folder / 'photoz_redshift_grid.csv').z.to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    pfig, paxes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    wavelengths={'subaru_g':4770,'subaru_r':6230,'subaru_i':7750,'subaru_z':8900,'subaru_y':9750,
                 'cfht_u':3740,'cfht_g':4870,'cfht_r':6250,'cfht_i':7670,'cfht_z':9150}
    for ax,pax,target,color in zip(axes,paxes,['lens','source'],['#3564b5','#bd6544']):
        pulls = pd.read_csv(folder / f'wNB1/signed_pulls_{target}.csv')
        wave=np.array([int(b.replace('pau_nb',''))*10 if b.startswith('pau_nb') else wavelengths[b] for b in pulls.band])
        order=np.argsort(wave)
        ax.errorbar(wave,pulls.flux,yerr=pulls.flux_error,fmt='o',ms=3.5,color=color,alpha=.8,label=f'{target.title()} measured flux')
        ax.plot(wave[order],pulls.model_flux.to_numpy()[order],color='#202c3e',lw=1.2,label='BCNz best-fit model')
        ax.axhline(0,color='gray',lw=.5);ax.set_ylabel('Flux (ZP = 26)');ax.legend(loc='best',fontsize=9);ax.grid(alpha=.15)
        cat=pd.read_parquet(folder / f'wNB1/pzcat_{target}.pq')
        p=cat[[f'z{i}' for i in range(len(z))]].iloc[0].to_numpy(float)
        p=p/np.trapz(p,z)
        pax.plot(z,p,color=color,label=f'{target.title()} · z = {cat.zb.iloc[0]:.3f}')
        pax.fill_between(z,0,p,color=color,alpha=.12)
        pax.set(ylabel='Probability density p(z)',title=target.title(),xlim=(z[0],z[-1]))
        pax.legend();pax.grid(alpha=.15)
    axes[-1].set_xlabel('Observed wavelength (Å)');fig.suptitle(name);fig.tight_layout();fig.savefig(folder/'flux.png',dpi=160);plt.close(fig)
    paxes[-1].set_xlabel('Redshift z')
    pfig.suptitle(name);pfig.tight_layout(rect=(0, 0, 1, 0.96))
    pfig.savefig(folder/'pz.png',dpi=160)
    # Each panel keeps its own probability scale and matching spec-z reference.
    record = system_record(name)
    for pax, target, color in zip(paxes, ['lens', 'source'], ['#3564b5', '#bd6544']):
        limits = pax.get_ylim()
        spec_z = record.get(f'{target}_redshift')
        if spec_z is not None and np.isfinite(spec_z):
            pax.axvline(spec_z, color=color, linestyle='--', linewidth=1.5,
                        label=f'{target.title()} spec-z = {spec_z:.3f}')
        pax.set_ylim(limits);pax.legend()
    pfig.savefig(folder/'pz_specz.png',dpi=160);plt.close(pfig)
