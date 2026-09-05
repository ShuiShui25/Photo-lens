"""Scientific plots shared by notebook and web workflows."""
from .workflow import ROOT, system_record

def plot_regions(name):
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch
    from regions import Regions
    import json
    cfg = json.loads((ROOT / system_record(name)['model_config']).read_text())
    base = ROOT / '1.lens-modeling' / name
    with fits.open((base / cfg['science_cutout']).resolve()) as h:
        data = np.asarray(h[cfg['science_hdu']].data, float)
        wcs = WCS(h[cfg['science_hdu']].header).celestial
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={'projection':wcs})
    ax.imshow(data, origin='lower', cmap='gray', norm=ImageNormalize(data, interval=PercentileInterval(99.5), stretch=AsinhStretch()))
    folder = ROOT / '2.region-detection/regions' / name
    from matplotlib.lines import Line2D
    handles, outlines = [], []
    for kind, color, label in [('lens','#ff75cc','Lens'),('source','#4ceaff','Source'),('background','#ffd166','Symmetric background')]:
        for region in Regions.read(folder / f'{name}_{kind}_point_fraction040.reg', format='ds9'):
            outlines.append(region.to_pixel(wcs).plot(ax=ax, facecolor='none', edgecolor=color, linewidth=2))
        handles.append(Line2D([0],[0],color=color,label=label,lw=2))
    x,y = wcs.world_to_pixel_values(system_record(name)['ra'], system_record(name)['dec'])
    ax.set_xlim(x-35,x+35);ax.set_ylim(y-35,y+35)
    ax.set_xlabel('Right ascension');ax.set_ylabel('Declination');ax.set_title(name)
    legend = ax.legend(handles=handles,loc='upper right',facecolor='#172033',labelcolor='white')
    fig.tight_layout();fig.savefig(folder / f'{name}_regions.png',dpi=170)
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
    pfig, pax = plt.subplots(figsize=(10, 4.5))
    wavelengths={'subaru_g':4770,'subaru_r':6230,'subaru_i':7750,'subaru_z':8900,'subaru_y':9750,
                 'cfht_u':3740,'cfht_g':4870,'cfht_r':6250,'cfht_i':7670,'cfht_z':9150}
    for ax,target,color in zip(axes,['lens','source'],['#3564b5','#bd6544']):
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
    axes[-1].set_xlabel('Observed wavelength (Å)');fig.suptitle(name);fig.tight_layout();fig.savefig(folder/'flux.png',dpi=160);plt.close(fig)
    pax.set(xlabel='Redshift z',ylabel='Probability density p(z)',title=name,xlim=(z[0],z[-1]))
    pax.legend();pax.grid(alpha=.15);pfig.tight_layout();pfig.savefig(folder/'pz.png',dpi=160)
    # Preserve the same axes when showing the catalogue spectroscopic redshifts.
    limits = pax.get_ylim()
    record = system_record(name)
    for target, color in [('lens', '#3564b5'), ('source', '#bd6544')]:
        spec_z = record.get(f'{target}_redshift')
        if spec_z is not None and np.isfinite(spec_z):
            pax.axvline(spec_z, color=color, linestyle='--', linewidth=1.5,
                        label=f'{target.title()} spec-z = {spec_z:.3f}')
    pax.set_ylim(limits);pax.legend()
    pfig.savefig(folder/'pz_specz.png',dpi=160);plt.close(pfig)
