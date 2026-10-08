"""Strict local inputs for a paired MMST experiment; no test data by default."""
import csv
import json
from pathlib import Path

import numpy as np

from .mmst_bridge import SOURCE, enable_dependencies

WEATHER = ['Avg Temperature (K)', 'Max Temperature (K)', 'Min Temperature (K)',
           'Precipitation (kg m**-2)', 'Relative Humidity (%)', 'Wind Gust (m s**-1)',
           'Wind Speed (m s**-1)', 'Downward Shortwave Radiation Flux (W m**-2)',
           'Vapor Pressure Deficit (kPa)']


def records(config, evaluation=False):
    year = config['evaluation_year'] if evaluation else config['train_year']
    with (SOURCE / f'input/county_info_{year}.csv').open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    answer = []
    for row in sorted(rows, key=lambda r: r['FIPS']):
        fips, state = row['FIPS'].zfill(5), row['State']
        answer.append({'fips': fips, 'year': year, 'state': state,
                       'split': 'historical_test' if evaluation else 'train',
                       'images': [f'Sentinel-2 Imagery/data/{year}/{state}/Agriculture_{fips[:2]}_{state}_{year}-{a}_{year}-{b}.h5'
                                  for a, b in [('04-01', '06-30'), ('07-01', '09-30')]],
                       'short_weather': [f'WRF-HRRR Computed Dataset/data/{year}/{state}/HRRR_{fips[:2]}_{state}_{year}-{m:02d}.csv'
                                         for m in config['months']],
                       'long_weather': [[f'WRF-HRRR Computed Dataset/data/{y}/{state}/HRRR_{fips[:2]}_{state}_{y}-{m:02d}.csv'
                                         for m in range(1, 13)] for y in config['long_term_years']],
                       'labels': f'USDA Crop Dataset/data/Soybean/{year}/USDA_Soybean_County_{year}.csv'})
    if len({r['fips'] for r in answer}) != len(answer):
        raise ValueError('Repeated counties in upstream list')
    return answer


def required_paths(rows):
    return sorted({p for r in rows for p in [*r['images'], *r['short_weather'],
                   *(v for year in r['long_weather'] for v in year), r['labels']]})


def client_partition(rows, per_state):
    """Adjacent FIPS blocks within each state; one state is one logical ring."""
    clients = []
    for state in sorted({r['state'] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r['state'] == state]
        for k, block in enumerate(np.array_split(indices, min(per_state, len(indices)))):
            clients.append({'id': f'{state}_{k}', 'ring': state, 'indices': block.tolist()})
    return clients


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]


def value(s):
    return float(str(s).replace(',', ''))


def sample(root, record, config):
    """One complete county-year: 6 dates x all aligned grids, 28 daily values."""
    enable_dependencies()
    import h5py
    import torch

    root, fips = Path(root), record['fips']
    frames, date_months = [], []
    for path in record['images']:
        with h5py.File(root / path, 'r') as hf:
            group = hf[fips]
            dates = sorted(group.keys())
            for date in dates[::2]:  # same monthly sampling as the upstream loader
                data = np.asarray(group[date]['data'])
                if data.ndim != 4 or data.shape[-1] != 3 or data.dtype != np.uint8:
                    raise ValueError(f'Unexpected author image layout: {fips}/{date}')
                h, w = data.shape[1:3]
                if min(h, w) < 224:
                    raise ValueError('Images must support the original 224 pixel crop')
                frames.append(data[:, (h-224)//2:(h-224)//2+224, (w-224)//2:(w-224)//2+224])
                # Both MM-DD and YYYY-MM-DD occur in public data versions.
                date_months.append(int(date.replace('_', '-').split('-')[-2]))
    if date_months != config['months']:
        raise ValueError(f'Expected six aligned monthly dates; found {date_months}')
    images = np.stack(frames).transpose(0, 1, 4, 2, 3).copy()
    grids = images.shape[1]
    if grids + 1 > config['grid_token_capacity']:
        raise ValueError(f'{fips} requires {grids+1} spatial tokens; explicit protocol amendment required')
    short, grid_keys = [], None
    for month, path in zip(config['months'], record['short_weather']):
        selected = [r for r in read_rows(root/path) if r['FIPS Code'].zfill(5) == fips and r['Daily/Monthly'] == 'Daily'
                    and 1 <= int(value(r['Day'])) <= 28]
        if any(int(value(r['Year'])) != record['year'] or int(value(r['Month'])) != month for r in selected):
            raise ValueError('Short weather timestamp mismatch')
        keys = sorted({int(value(r['Grid Index'])) for r in selected})
        if len(keys) != grids or (grid_keys is not None and keys != grid_keys):
            raise ValueError('Image/weather grid count or temporal alignment mismatch')
        grid_keys = keys
        grid_data = []
        for key in keys:
            day_rows = sorted([r for r in selected if int(value(r['Grid Index'])) == key], key=lambda r: value(r['Day']))
            if [int(value(r['Day'])) for r in day_rows] != list(range(1, 29)):
                raise ValueError('Missing or duplicate daily weather, no silent imputation')
            grid_data.append([[value(r[k]) for k in WEATHER] for r in day_rows])
        short.append(grid_data)
    long = []
    for year, paths in zip(config['long_term_years'], record['long_weather']):
        months = []
        for month, path in enumerate(paths, 1):
            selected = [r for r in read_rows(root/path) if r['FIPS Code'].zfill(5) == fips and r['Daily/Monthly'] == 'Monthly']
            if len(selected) != 1:
                raise ValueError(f'Expected one county monthly record: {fips}/{year}/{month}')
            row = selected[0]
            if int(value(row['Year'])) != year or int(value(row['Month'])) != month:
                raise ValueError('Long weather timestamp mismatch')
            months.append([value(row[k]) for k in WEATHER])
        long.append(months)
    labels = [r for r in read_rows(root/record['labels']) if r['state_ansi'].zfill(2)+r['county_ansi'].zfill(3) == fips]
    if len(labels) != 1:
        raise ValueError(f'Expected one author label row for {fips}')
    target = np.array([value(labels[0][k]) for k in ('PRODUCTION, MEASURED IN BU', 'YIELD, MEASURED IN BU / ACRE')])
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError('Invalid log target')
    xs, xl = np.asarray(short, dtype=np.float32), np.asarray(long, dtype=np.float32)
    if not np.isfinite(xs).all() or not np.isfinite(xl).all():
        raise ValueError('Nonfinite meteorology')
    x = torch.from_numpy(images).float()
    mean = x.new_tensor([.466, .471, .380]).view(1, 1, 3, 1, 1)
    std = x.new_tensor([.195, .194, .192]).view(1, 1, 3, 1, 1)
    return ((x-mean)/std).unsqueeze(0), torch.from_numpy(xs).unsqueeze(0), torch.from_numpy(xl).unsqueeze(0), torch.tensor(np.log(target), dtype=torch.float32).unsqueeze(0)
