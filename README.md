# Photo-lens

[English](#english) | [中文](#中文)

## English

Photo-lens estimates multiband photometry and photometric redshifts for strong gravitational lens systems. It combines AutoLens lens modeling, automatic region detection, PAUS narrowband and broadband photometry, and BCNz SED fitting to measure fluxes and redshift probability distributions p(z) for the lens galaxy and background source separately.

The repository includes a complete example for **J022511-045433**: the HSC/PAUS images, PSF, positions, models and fit results, regions, photometry, BCNz model caches, and visualizations needed for the workflow. BCNz source code is bundled and compiled during installation; no separate download is required.

### Installation

On Linux, we recommend separate environments for photometry and AutoLens modeling.

```bash
git clone git@github.com:ShuiShui25/Photo-lens.git
cd Photo-lens

# Photometry, region detection, and the web backend: Python 3.10
conda env create -f environment.yml
conda activate photo-lens
python -m ipykernel install --user --name photo-lens --display-name Photo-lens

# AutoLens modeling: Python 3.11
conda create -n photo-lens-modeling -c conda-forge python=3.11 pip
conda activate photo-lens-modeling
python -m pip install -r requirements-modeling.txt
python 1.lens-modeling/autoarray_jax_crop_patch/apply_autoarray_crop_fix.py
python -m ipykernel install --user --name photo-lens-modeling --display-name Photo-lens-modeling
```

For the photometry environment, `requirements.txt` installs the dependencies and bundled BCNz; `environment.yml` also installs a C compiler and SExtractor. The modeling environment uses `requirements-modeling.txt`. To run a new parameter search on a GPU, install JAX CUDA support compatible with your hardware. The default model reconstruction and visualization run on a CPU. Set `PHOTO_LENS_JAX_PLATFORM=cpu` to run a new search on a CPU as well, although this will be slower.

### Option 1: Notebooks

Each workflow directory contains one notebook without saved execution outputs. Run them in order to follow the steps and display results for J022511-045433:

1. [1.lens-modeling](1.lens-modeling/J022511-045433.ipynb): load the images and configuration, reconstruct the models, and display AutoLens plot panels for all four modeling stages. Select the `Photo-lens-modeling` kernel.
2. [2.region-detection](2.region-detection/J022511-045433.ipynb): run automatic detection and display the lens, source, and point-reflected background regions. Select the `Photo-lens` kernel.
3. [3.photometric-estimation](3.photometric-estimation/J022511-045433.ipynb): run photometry and BCNz fitting, and display fluxes, p(z), and band information. Select the `Photo-lens` kernel.

By default, the modeling notebook re-evaluates the saved maximum-likelihood models. Set `REFIT = True` to run a new parameter search.

### Option 2: HTML Dashboard

After installing the environments, run:

```bash
bash start_dashboard.sh
```

The script locates the `photo-lens` and `photo-lens-modeling` environments (or existing `bcnz` and `autolens-jax` environments), starts the local backend, and opens your browser automatically. No `conda activate` is required. If the dashboard is already running, it opens the existing page and returns to the shell; Ctrl+C in that terminal cannot stop the existing server. Use `bash start_dashboard.sh --stop` to stop it, or `bash start_dashboard.sh --restart` to restart it with the current code. Stopping or restarting also cancels its running analysis tasks. Keep the terminal open while using a newly started backend; press `Ctrl+C` to stop it. Use `bash start_dashboard.sh --port 8766` for a different port, or `--no-browser` to skip opening a browser.

You can also start the backend manually in the photometry environment and specify the modeling interpreter:

```bash
conda activate photo-lens
python serve.py --model-python /path/to/photo-lens-modeling/bin/python
```

Open **http://127.0.0.1:8765**. The homepage lists mugshots of available systems. Open a system's detail page to run each of the three modules and view its status, task output, and updated images.

- **1.lens-modeling**: reconstruct the saved models and display AutoLens plot panels. Choose “Refit all model stages” to fit the models again.
- **2.region-detection**: detect and update the lens, source, and symmetric background regions.
- **3.photometric-estimation**: remeasure fluxes and calculate p(z) for the lens and source.

Run the modules in order, using the products of the preceding stage. The backend executes tasks serially and allows running tasks to be stopped. Opening `index.html` directly lets you browse the bundled results; executing scripts requires the local backend above. Static hosting services such as GitHub Pages cannot launch Python tasks.

To specify the photometry interpreter or SExtractor explicitly, pass `--photo-python /path/to/python` or `--sextractor /path/to/source-extractor`. Use `--port` to change the port.

### Data and Configuration

| Path | Contents |
| --- | --- |
| `index.html` | System gallery and the three analysis modules |
| `catalogue/` | Original reference HTML, catalogue data, and mugshot assets |
| `data/` | Example broadband/narrowband images, PSF, positions, and photometry catalogue |
| `1.lens-modeling/` | AutoLens implementation, example configuration, models, and modeling plots |
| `2.region-detection/` | Detection algorithms, DS9 regions, and region plots |
| `3.photometric-estimation/` | Photometry implementation, native BCNz model caches, fluxes, and p(z) results |
| `config/systems.json` | Metadata for bundled systems; local systems are also discovered from `data/bb/` |
| `config/photometry.json` | Photometry and band settings |
| `vendor/bcnz/` | Bundled BCNz source code and license |

See the [Combined lens catalogue](catalogue/Combined_lens_info_V2_filtered_zs_lt2_field.html) for reference information. The local backend discovers named system directories under `data/bb/` on each system-list request. The dashboard refreshes the list every 10 seconds while visible; you can also click **Refresh systems**. Static HTML previews only show the bundled system list.

For a local system, use its astronomical name consistently:

- `data/bb/<name>/`: `science/`, `psf/`, `variance/` or `weight/`, and `positions_gui.txt`.
- `data/nb/single_epoch/<wavelength>/<name>/`: exposure FITS files, `img_list.txt`, exposure times, and zero-point calibration tables.
- `data/nb/coadd/<wavelength>/<name>/`: `coadd_cutout_<name>.fits` and its `.weight.fits` companion.
- `data/bb/<name>/catalogue.csv`: matching catalogue rows; the existing `lens_id` column stores the system name.
- `data/bb/<name>/model_config.json`: explicit AutoLens settings, including redshifts, image HDUs, and pixel scale. Paths are relative to `1.lens-modeling/<name>/`, where the configuration is copied when modeling starts.
- Optional `data/bb/<name>/system.json`, `mugshot.png`, and `nb_antisymmetry_centres.csv` provide display metadata, a thumbnail, and existing NB centre corrections. `system.json` can specify `field`, `ra`, `dec`, `lens_redshift`, `source_redshift`, and `excluded_bands`.

Discovery does not execute any analysis. New systems without saved models start with **Fit model & plot**; subsequent stages become available when their prerequisites exist. Missing results are shown as not yet run. New systems need their own modeling configuration rather than inheriting the example's scientific parameters.

`.gitignore` keeps additional systems' data and generated products local. Only the bundled **J022511-045433** example is included in Git; code and shared BCNz caches remain tracked.

### License

Photo-lens code is licensed under the [GPL-3.0](LICENSE). Bundled BCNz is separately licensed under [GPL-3.0-or-later](vendor/bcnz/LICENCE.TXT); its license and copyright notices are retained in the source files.

---

## 中文

Photo-lens 用于强引力透镜系统的多波段测光与测光红移估计。它结合 AutoLens 透镜建模、自动选区、PAUS 窄带与宽带测光，以及 BCNz SED 拟合，分别得到透镜星系和背景源的 flux 与红移概率分布 p(z)。

仓库提供 **J022511-045433** 的完整示例：流程所需的 HSC/PAUS 图像、PSF、位置、模型与拟合结果、选区、测光、BCNz 模型缓存及可视化。BCNz 源码随仓库提供，安装时自动编译，无需另行下载。

### 安装

建议在 Linux 上分别建立测光环境和 AutoLens 建模环境。

```bash
git clone git@github.com:ShuiShui25/Photo-lens.git
cd Photo-lens

# 测光、选区和网页后台：Python 3.10
conda env create -f environment.yml
conda activate photo-lens
python -m ipykernel install --user --name photo-lens --display-name Photo-lens

# AutoLens 建模：Python 3.11
conda create -n photo-lens-modeling -c conda-forge python=3.11 pip
conda activate photo-lens-modeling
python -m pip install -r requirements-modeling.txt
python 1.lens-modeling/autoarray_jax_crop_patch/apply_autoarray_crop_fix.py
python -m ipykernel install --user --name photo-lens-modeling --display-name Photo-lens-modeling
```

测光环境中的 `requirements.txt` 安装依赖及仓库内 BCNz；`environment.yml` 还安装 C 编译器和 SExtractor。建模环境使用 `requirements-modeling.txt`。重新进行 GPU 参数搜索时，还需安装与硬件匹配的 JAX CUDA 支持；默认的模型重建和展示可在 CPU 上运行。设置 `PHOTO_LENS_JAX_PLATFORM=cpu` 也可在 CPU 上重新搜索，但速度会较慢。

### 方式一：Notebook

每个流程目录有一个普通 notebook，以 J022511-045433 为例，依次运行即可显示步骤与结果：

1. [1.lens-modeling](1.lens-modeling/J022511-045433.ipynb)：读取图像及配置，重建模型，展示四阶段 AutoLens 建模组图。选择 `Photo-lens-modeling` kernel。
2. [2.region-detection](2.region-detection/J022511-045433.ipynb)：运行自动检测，展示 lens、source 和点反射对称背景选区。选择 `Photo-lens` kernel。
3. [3.photometric-estimation](3.photometric-estimation/J022511-045433.ipynb)：执行测光和 BCNz 拟合，展示 flux、p(z) 及波段信息。选择 `Photo-lens` kernel。

建模 notebook 默认重新评估已保存的最大似然模型；将 `REFIT = True` 可重新执行参数搜索。

### 方式二：总 HTML

安装环境后，运行：

```bash
bash start_dashboard.sh
```

脚本自动查找 `photo-lens` 和 `photo-lens-modeling` 环境（也支持已有的 `bcnz` 和 `autolens-jax`），启动后台并自动打开浏览器，无需先执行 `conda activate`。如果后台已运行，脚本打开现有页面后返回命令行，此时在该终端按 Ctrl+C 无法停止旧后台。使用 `bash start_dashboard.sh --stop` 停止旧后台，或 `bash start_dashboard.sh --restart` 重启并加载最新代码。停止或重启也会取消该后台正在运行的分析任务。新启动后台后请保持终端打开，按 `Ctrl+C` 停止。需要更换端口时运行 `bash start_dashboard.sh --port 8766`；不需要自动打开浏览器时添加 `--no-browser`。

也可以在测光环境中手动启动后台，并指定建模环境的 Python：

```bash
conda activate photo-lens
python serve.py --model-python /path/to/photo-lens-modeling/bin/python
```

打开 **http://127.0.0.1:8765**。首页列出可运行系统的 mugshot；点击进入详细页面后，可以分别运行三个模块，查看运行状态、任务输出和更新后的图像。

- **1.lens-modeling**：重建已保存模型并显示 AutoLens 组图；“Refit all model stages” 可重新拟合。
- **2.region-detection**：检测并更新 lens/source/对称背景选区。
- **3.photometric-estimation**：重新测量 flux，并计算 lens/source 的 p(z)。

三个模块按顺序使用上一步的产物；后台串行执行任务，运行中可停止。直接双击 `index.html` 可以浏览随包结果；执行脚本需要上述本地后台，GitHub Pages 等纯静态网页无法启动 Python。

如需显式指定测光环境或 SExtractor，可传入 `--photo-python /path/to/python`、`--sextractor /path/to/source-extractor`。更改端口使用 `--port`。

### 数据与配置

| 路径 | 内容 |
| --- | --- |
| `index.html` | 系统入口与三个分析模块 |
| `catalogue/` | 原始参考 HTML、目录数据及 mugshot 资源 |
| `data/` | 示例的宽带/窄带图像、PSF、位置和测光目录 |
| `1.lens-modeling/` | AutoLens 实现、示例配置、模型及建模图 |
| `2.region-detection/` | 检测算法、DS9 选区和选区图 |
| `3.photometric-estimation/` | 测光实现、原生 BCNz 模型缓存、flux 和 p(z) 结果 |
| `config/systems.json` | 随包系统的信息；本地系统还会从 `data/bb/` 自动发现 |
| `config/photometry.json` | 测光及波段设置 |
| `vendor/bcnz/` | 随包 BCNz 源码与许可证 |

参考资料见 [Combined lens catalogue](catalogue/Combined_lens_info_V2_filtered_zs_lt2_field.html)。本地后台每次读取系统列表时都会扫描 `data/bb/` 下以系统名称命名的目录。页面可见时每 10 秒更新一次，也可点击 **Refresh systems**；直接打开静态 HTML 时仅显示随包系统列表。

新增系统时，统一使用天体名称组织数据：

- `data/bb/<name>/`：`science/`、`psf/`、`variance/` 或 `weight/`，以及 `positions_gui.txt`。
- `data/nb/single_epoch/<波长>/<name>/`：单曝光 FITS、`img_list.txt`、曝光时间及零点标定表。
- `data/nb/coadd/<波长>/<name>/`：`coadd_cutout_<name>.fits` 及对应 `.weight.fits`。
- `data/bb/<name>/catalogue.csv`：该系统的目录记录；保留已有的 `lens_id` 列名，但其值使用系统名称。
- `data/bb/<name>/model_config.json`：该系统的 AutoLens 参数，包括红移、图像 HDU、像素尺度。路径相对于 `1.lens-modeling/<name>/`；启动建模时才会将配置复制到该位置。
- 可选的 `data/bb/<name>/system.json`、`mugshot.png` 和 `nb_antisymmetry_centres.csv` 分别提供显示信息、缩略图和已有 NB 中心修正。`system.json` 可设置 `field`、`ra`、`dec`、`lens_redshift`、`source_redshift` 和 `excluded_bands`。

自动识别不会启动分析。没有已保存模型的新系统首先显示 **Fit model & plot**；后续模块在前置输入齐全后开放，缺失的结果标为尚未运行。新系统需要自己的建模配置，不会套用示例的科学参数。

`.gitignore` 将后续系统的数据和生成结果保留在本地，仅同步最初的 **J022511-045433** 示例；代码和共享 BCNz 缓存继续由 Git 管理。

### 许可

Photo-lens 代码采用 [GPL-3.0](LICENSE)。随包 BCNz 单独遵循 [GPL-3.0-or-later](vendor/bcnz/LICENCE.TXT)，其许可与版权声明保留在源文件中。
