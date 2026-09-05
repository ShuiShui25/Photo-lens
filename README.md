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

The script locates the `photo-lens` and `photo-lens-modeling` environments (or existing `bcnz` and `autolens-jax` environments), starts the local backend, and opens your browser automatically. No `conda activate` is required. If the dashboard is already running, it opens the existing page. Keep the terminal open while using a newly started backend; press `Ctrl+C` to stop it. Use `bash start_dashboard.sh --port 8766` for a different port, or `--no-browser` to skip opening a browser.

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
| `config/systems.json` | Systems available in the dashboard |
| `config/photometry.json` | Photometry and band settings |
| `vendor/bcnz/` | Bundled BCNz source code and license |

See the [Combined lens catalogue](catalogue/Combined_lens_info_V2_filtered_zs_lt2_field.html) for reference information. The dashboard's system list is configured in `config/systems.json`; adding a system also requires its corresponding data and models.

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

脚本自动查找 `photo-lens` 和 `photo-lens-modeling` 环境（也支持已有的 `bcnz` 和 `autolens-jax`），启动后台并自动打开浏览器，无需先执行 `conda activate`。如果后台已运行，直接打开现有页面。新启动后台后请保持终端打开，按 `Ctrl+C` 停止。需要更换端口时运行 `bash start_dashboard.sh --port 8766`；不需要自动打开浏览器时添加 `--no-browser`。

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
| `config/systems.json` | 总页面中可运行系统的信息 |
| `config/photometry.json` | 测光及波段设置 |
| `vendor/bcnz/` | 随包 BCNz 源码与许可证 |

参考资料见 [Combined lens catalogue](catalogue/Combined_lens_info_V2_filtered_zs_lt2_field.html)。工作流首页的系统列表由 `config/systems.json` 配置；新增系统时需同时准备对应数据和模型。

### 许可

Photo-lens 代码采用 [GPL-3.0](LICENSE)。随包 BCNz 单独遵循 [GPL-3.0-or-later](vendor/bcnz/LICENCE.TXT)，其许可与版权声明保留在源文件中。
