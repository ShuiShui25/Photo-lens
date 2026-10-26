# bcnz

The BCNz photo-z code described in [Eriksen 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.484.4200E/abstract). Support for
the Bayesian extension [Alarcon 2020](https://ui.adsabs.harvard.edu/abs/2020arXiv200711132A/abstract) is partly added.


The code performs the following steps:

* Create the model.
* Calibration.
* Run the photo-z.

# Installation
Clone the respository and write

pip install -e .

after entering into the cloned directory.

## Running over a galaxy catalgue.
While intended to be integrated in PAUdm, one can also run the code from the command line. As an 
example, one can run

./run_bcnz.py ./output/v5_941 ./cache/models 941 w3 --ip_dask=tcp://scheduler.example.org:8786

where bcnz_paudm.py is a binary in the bcnz/bin directory. Here the photo-z code will run the code
over MEMBA production 941, which is in the W3 field. Intermediate and final outputs are stored
in the 

./output/v5_941

directory. Further, computing the models are time consuming. If not already existing, they will
be calculated and stored in 

./cache/models.

For running in parallel, the code uses [Dask](https://dask.org/). By specifying Dask, one can
run on an existing Dask cluster. For example, dask-jobqueue supports running Dask on HTCondor,
which is used at the [PIC](www.pic.es) data center.

## Other usages.
The BCNz2 code include different modules which can be used directly. For example,

all_filters = bcnz.model.all_filters()

would give a Pandas dataframe with all the filter transmission curves. Connecting to the 
PAUdm database can be done using:

conn = bcnz.connect_db()

For seeing the emission line ratios used:

bcnz.model.line_ratios()

##BCNz on mock catalogues: 
The bcnz_externalfile.p implements BCNz on galaxy mocks. An example of command line to run on mocks is:

./bcnz_externalfile.py ./catalogues/mock.csv ./output/mock ./cache/models --bbnames=['U','B','V','R','I','ZN']

The first argument is the path to the catalogue. This needs to contain the narrow-band fluxes in PAUS units named as NBXXX (e.g. NB455) and the broad bands. The broad-band naming needs to be specified in the argument --bbnames.

The second argument is the path to the output directory. 
The third argument is the path to the models. 

Running the calbration is an option. By default, it is disabled, but it can be enabled by adding --calib=True in the command line.

Running using dask paralelisation is also possible specifying the ip_dask in the command line. 

# Acknowledgement
This project has received funding from the European Union’s Horizon 2020 research
and innovation programme under the grant agreement No
776247 EWC.


## External datasets and portable builds

Photo-lens uses the bundled templates and its local photometry tables. The
optional PAUdm/Bayevz workflows require external datasets that are not included.
Provide a path explicitly or set the corresponding environment variable:

| API | Explicit input | Environment fallback |
| --- | --- | --- |
| `bcnz.config.test_bayevz`, `pauscosmos_deep` | `data_dir` | `BCNZ_BAYEVZ_DATA_DIR` |
| `bcnz.specz.sdss` | `catalogue_path` | `BCNZ_SDSS_CATALOGUE` |
| `bcnz.specz.sdss_gama` | `catalogue_path` | `BCNZ_SDSS_GAMA_CATALOGUE` |
| `bcnz.specz.vipers` | `catalogue_path` | `BCNZ_VIPERS_CATALOGUE` |
| Bayevz `get_data_hack` | `catalogue_file`, `zero_point_file` | `BCNZ_BAYEVZ_CATALOGUE`, `BCNZ_BAYEVZ_ZEROPOINTS` |

Relative inputs are resolved against the current working directory. Missing
inputs raise a descriptive error instead of looking in a developer's directory.
The Bayevz data directory must contain `filt_70bands_noirach/`, `seds/`,
`ext_laws/` and `calib_filters/`. The MPI script accepts `--model-dir`,
`--output-dir`, `--data-dir` and `--catalogue-file`; these paths replace its
former local defaults. This does not supply the optional external datasets,
PAUdm access, MPI runtime or scientific validation of that workflow.

The HTCondor template requires explicit submit variables, for example:

```bash
condor_submit bcnz/bin/cache_model.sub \
  bcnz_python="$(command -v python)" \
  bcnz_cache_script="$(pwd)/bcnz/bin/cache_model.py" \
  bcnz_model_dir="$(pwd)/cache/models"
```

Cython C sources are generated from the packaged `.pyx` files during builds.
Generated C files containing build-machine paths are not distributed as source;
use the build dependencies in `pyproject.toml` to build the extensions.


## Regression checks

After installing this bundled BCNz in the photometry environment, run from the
Photo-lens repository root:

```bash
python -m unittest discover -s tests -v
```

These checks use temporary catalogue fixtures and mock the optional PAUdm
connection. They cover input configuration and catalogue transformations; they
do not require survey databases or an MPI cluster.
