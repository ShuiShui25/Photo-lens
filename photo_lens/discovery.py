"""Discover named local systems without running analyses or writing files."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def local_path(base, value):
    path = (base / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Input path must be inside the repository: {value}')
    return path


def discover_systems():
    registered = {s['id']: s for s in read_json(ROOT / 'config/systems.json')}
    folders = {p.name: p for p in (ROOT / 'data/bb').iterdir()
               if p.is_dir() and not p.name.startswith('.') and (p / 'science').is_dir()}
    systems = []
    for name in sorted(registered.keys() | folders.keys()):
        folder = ROOT / 'data/bb' / name
        record = dict(registered.get(name, {}))
        try:
            meta = read_json(folder / 'system.json')
            # The directory name is the canonical name, never a numeric catalogue ID.
            record.update({k: v for k, v in meta.items() if k not in ('id', 'name')})
            record['id'] = name
            record['name'] = name
            model_path = ROOT / '1.lens-modeling' / name / 'config.json'
            seed = local_path(folder, meta.get('model_parameters', 'model_config.json'))
            cfg = read_json(model_path if model_path.exists() else seed)
            # Seed paths use the same base as the eventual stage configuration.
            record['model_config'] = model_path.relative_to(ROOT).as_posix()
            record['model_seed'] = seed.relative_to(ROOT).as_posix() if seed.is_file() else None
            for field, key in [('ra', 'lens_ra_deg'), ('dec', 'lens_dec_deg'),
                               ('lens_redshift', 'z_lens'), ('source_redshift', 'z_source')]:
                record.setdefault(field, cfg.get(key))
            record.setdefault('aliases', [])
            record.setdefault('field', '—')
            record.setdefault('einstein_radius', f"{cfg['einstein_radius_mean']} arcsec" if cfg.get('einstein_radius_mean') else '—')
            record.setdefault('image_separation', '—')
            record.setdefault('reference', 'catalogue/Combined_lens_info_V2_filtered_zs_lt2_field.html')
            bb = sorted(p.stem for p in (folder / 'science').glob('*.fits'))
            survey = cfg.get('bb_survey') or ('HSC' if any(b.startswith('HSC-') for b in bb) else 'CFHT')
            record['bb_survey'] = survey
            record.setdefault('imaging', f'PAUS + {survey}')
            nb = set()
            for wave in range(455, 846, 10):
                single = ROOT / 'data/nb/single_epoch' / str(wave) / name
                coadd = ROOT / 'data/nb/coadd' / str(wave) / name
                if any(p for p in single.glob('*.fits') if not p.name.endswith('.weight.fits')) or (coadd / f'coadd_cutout_{name}.fits').is_file():
                    nb.add(wave)
                elif (ROOT / 'data/nb/coadd' / f'NB{wave}' / f'coadd_cutout_{name}.fits').is_file():
                    nb.add(wave)
            record['nb_count'] = len(nb)
            record['bands'] = f'{len(nb)} PAUS narrow bands + ' + ', '.join(bb)
            if (folder / 'mugshot.png').is_file():
                record['mugshot'] = (folder / 'mugshot.png').relative_to(ROOT).as_posix()
            record.setdefault('mugshot', 'web/system-placeholder.svg')
            catalogue = local_path(folder, meta.get('catalogue', 'catalogue.csv'))
            if not catalogue.is_file():
                catalogue = ROOT / 'data/catalogue/targets.csv'
            record['catalogue_path'] = catalogue.relative_to(ROOT).as_posix()
            has_catalogue = False
            if catalogue.is_file():
                with catalogue.open() as stream:
                    has_catalogue = any(row.get('lens_id') == name for row in csv.DictReader(stream))
            centre = local_path(folder, meta.get('nb_centres', 'nb_antisymmetry_centres.csv'))
            record['nb_centre_path'] = centre.relative_to(ROOT).as_posix() if centre.is_file() else None
            issues = []
            if not cfg:
                issues.append('Add model_config.json beside the science directory before fitting.')
            else:
                for key in ('science_cutout', 'psf', 'positions_file'):
                    if not cfg.get(key) or not local_path(model_path.parent, cfg[key]).is_file():
                        issues.append(f'Missing modeling input: {key}')
                noise = cfg.get('variance_cutout') or cfg.get('weight_cutout')
                if not noise or not local_path(model_path.parent, noise).is_file():
                    issues.append('Missing variance or weight image.')
            output = model_path.parent / 'output'
            record['has_model'] = all(any(output.glob(f'stage{i}_*/**/files/tracer.json')) for i in range(1, 5))
            regions = ROOT / '2.region-detection/regions' / name
            has_regions = all((regions / f'{name}_{kind}_point_fraction040.reg').is_file()
                              for kind in ('lens', 'source', 'background'))
            record['can_run'] = {
                'modeling': not issues,
                'regions': (output / f'wcs_products_stage4/{name}_exact_fit_plane_qa.json').is_file(),
                'photoz': has_regions and has_catalogue and bool(nb),
            }
            record['input_issues'] = issues
            record['has_results'] = {
                'modeling': (output / 'stage_previews/stage4_fit.png').is_file(),
                'regions': (regions / f'{name}_regions.png').is_file(),
                'photoz': (ROOT / '3.photometric-estimation/results' / name / 'pz.png').is_file(),
            }
            record['notebooks'] = {stage: (ROOT / f'{i}.{directory}/{name}.ipynb').is_file()
                for i, directory, stage in [(1, 'lens-modeling', 'modeling'),
                    (2, 'region-detection', 'regions'), (3, 'photometric-estimation', 'photoz')]}
        except (OSError, ValueError, TypeError, KeyError) as exc:
            record.update(id=name, name=name, ra=None, dec=None, aliases=[], field='—',
                lens_redshift=None, source_redshift=None, bands='Incomplete local data',
                imaging='—', einstein_radius='—', image_separation='—', nb_count=0,
                reference='catalogue/Combined_lens_info_V2_filtered_zs_lt2_field.html',
                mugshot='web/system-placeholder.svg', has_model=False,
                can_run=dict.fromkeys(('modeling', 'regions', 'photoz'), False),
                has_results={}, notebooks={}, input_issues=[str(exc)])
        systems.append(record)
    return systems


def ensure_model_config(record):
    """Materialize an explicit local seed only when the user starts modeling."""
    path = ROOT / record['model_config']
    if path.is_file():
        return path
    if not record.get('model_seed'):
        raise ValueError('Add data/bb/<name>/model_config.json before starting modeling.')
    cfg = read_json(ROOT / record['model_seed'])
    cfg['system'] = record['id']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + '\n')
    return path
