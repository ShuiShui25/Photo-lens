# Photo-lens

Photo-lens 是用于强引力透镜系统的多波段测光和测光红移分析框架。它结合宽带透镜建模、自动选区、PAUS 窄带与 HSC/CFHT 宽带测光，以及 BCNz SED 拟合，分别估计前景透镜星系和背景源的红移概率分布。

本仓库以 **PHOTO-LENS032** 为科学结果基线；保留九个系统的稳定测光、红移结果和选区，并包含本地修改版 BCNz 的完整源码。安装时从 `vendor/bcnz` 编译安装，无需另外克隆 BCNz，也不依赖作者机器上已经安装的包。

## 分析流程

1. **宽带建模**：优先选择具有 science、variance/weight、PSF 和同坐标网格 positions 的 HSC 数据，否则使用完整的 CFHT 数据。模型依次采用 MGE 透镜光 + SIE 质量 + Gaussian 源、power-law 质量、双独立中心源，以及固定 20×20 像素源。像素源使用 `ConstantPlusBrightnessZeroth(signal_scale=0.5)`，信号先裁剪到 `[0,1]`；保留 AutoArray FFT crop 修正。13369 的最终模型停在第三阶段。
2. **自动选区**：在精确 AutoLens `FitImaging` 透镜光扣除图像上检测源。SExtractor 检测阈值为 4σ，滞后生长阈值为检测阈值的 0.75 倍；重叠区域按点反射源光比例 `r_s >= 0.4` 分配，保留卫星/次级源处理。稳定选区位于 `region-detection/stable-regions`。
3. **测光**：将天球选区投影到各波段 WCS，计算分数像素孔径、点反射背景扣除、窄带曝光标定及合并、经验噪声与相关噪声诊断。默认使用 81 子像素，161 子像素检查收敛。NB 反对称中心修正仅在位移低于 0.15 arcsec 时接受。
4. **红移拟合**：lens 和 source 使用共同有效波段，`w_NB=1`。HSC 系统使用 PAUS NB + 官方 HSC-G + HSC r/i/z/y；CFHT 系统使用 PAUS NB + CFHT u/g/r/i/z。仅 3110 排除 NB535，测量值保留，拟合与 signed-pull 表排除该波段。
5. **输出与诊断**：输出长表测光、BCNz 输入和 p(z) parquet、波段选择原因、signed pulls、SED/p(z) 图，以及 lens/source 红移顺序概率。

稳定样本：HSC 为 2、8、3110、3336、3441、13369；CFHT 为 12、12971、13246。机器可读参数见 [stable_config.json](stable_config.json)。

## 安装

已锁定的主环境面向 **Linux x86_64 / Python 3.10**。BCNz 的两个 Cython 扩展会在本机编译，因此需要 C 编译器和 Python 开发头文件。GPU 建模使用独立的可选环境，见下文。

```bash
git clone git@github.com:ShuiShui25/Photo-lens.git
cd Photo-lens
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
MPLBACKEND=Agg python tests/smoke_test.py
```

也可在仓库根目录使用 Conda（包含 C 编译器和 SExtractor）：

```bash
conda env create -f environment.yml
conda activate photo-lens
MPLBACKEND=Agg python tests/smoke_test.py
```

`requirements.txt` 同时安装锁定依赖与本地 BCNz；`requirements-lock.txt` 是可安装依赖列表。`docs/environment-original-bcnz.txt` 是原工作环境的完整包版本清单，用于溯源，包含本框架不需要的其他研究软件，不作为推荐安装入口。无需设置指向作者目录的 `PYTHONPATH`。

## 仓库自带的全链路测试

`tests/smoke_test.py` 校验数据及稳定快照的 SHA-256，然后从真实 12 号系统 FITS 图像开始，执行孔径测光、误差和收敛检查、lens/source BCNz 拟合及诊断图生成，并检查 3110 的 NB535 排除规则。测试在临时目录计算，不覆盖稳定结果；输出保存在 `.validation/smoke/`。

为使 Git 下载和测试可控，测试使用 21/41 子像素作测光和收敛检查；夹具保留真实图像与全部模板组合，但将模型红移网格按步长 20 抽样，并额外保留 z=0.01。**这个测试验证软件运行链路，不用于复现最终科学红移数值。** 它从已经冻结的选区开始，不重新执行 GPU 透镜建模和 SExtractor。测试结果与实际覆盖范围见 [docs/VALIDATION.md](docs/VALIDATION.md)。

也可以自动验证“新建环境 → 从 GitHub 克隆 → 安装 → 测试”：

```bash
PHOTO_LENS_BASE_PYTHON=python3.10 bash scripts/verify_from_git.sh
```

该脚本使用新的临时目录和独立环境，打印实际测试的 Git 提交号，并保留日志所指向的结果目录。

## 使用原始分辨率数据重算

九系统完整原始图像、原生 BCNz 缓存和 AutoLens 搜索输出体积较大，不放入普通 Git 历史。仓库自带上述测试数据，可在下载后独立完成测试。完整科学重算需要另行提供科学数据；程序不会偷偷回退到相邻旧版本目录。

如果已有原 PHOTO-LENS03 资源，可复制到当前仓库：

```bash
python scripts/import_science_assets.py /path/to/PHOTO-LENS03
MPLBACKEND=Agg python scripts/run_photoz.py --system 12
MPLBACKEND=Agg python scripts/run_photoz.py --system 3110
```

导入内容为 `data/`、`photometric-estimation/model/`、`photometric-estimation/hsc-g-response/model-g/`。重新测光使用仓库内的稳定选区和 NB 中心表，输出到 `runs/<system>/`。这些大资源和新结果被 Git 忽略。支持任一已配置系统；HSC-G 与 NB535 规则自动应用。

`lens-modeling/` 和 `region-detection/` 保留上游建模及选区实现。重新运行这些前置阶段还需要完整 PSF/positions、AutoLens/JAX 环境、可用 GPU，以及相应模型输出；它们不是主环境安装测试的一部分。原 GPU 环境版本清单单独保存，不能据此宣称全新 GPU 环境已经验证。

## 目录与输出

| 路径 | 内容 |
| --- | --- |
| `lens-modeling/` | 多阶段透镜建模、配置、AutoArray 修正与选区支持模块 |
| `region-detection/` | 自动选区算法、九系统稳定 DS9 选区 |
| `photometric-estimation/paus_lensing034/` | 可复用测光、误差、BCNz 拟合及诊断实现 |
| `photometric-estimation/results/` | PHOTO-LENS032 九系统只读科学基线 |
| `photometric-estimation/all_systems_photoz_report.html` | 已生成的综合报告，下载后用浏览器打开 |
| `vendor/bcnz/` | 本地修改版 BCNz、许可证、修改说明和上游提交记录 |
| `tests/fixtures/` | 可离线运行的真实系统测试夹具 |
| `scripts/` | 稳定重算和科学资源导入入口 |
| `tools/archive/` | 依赖旧研究目录的历史快照构建脚本，仅作溯源 |

结果目录包括 `common_photoz_bands.csv`、`photoz_band_selection_status.csv`、`photoz_redshift_grid.csv`、`wNB1/galcat_in_{lens,source}.pq`、`wNB1/pzcat_{lens,source}.pq`、signed-pull 表和图。冻结文件摘要见 `docs/stable-results-sha256.json`；原版说明见 `docs/PHOTO-LENS032-original.md`。

## 概率定义

具有物理解释的红移顺序概率为：

\[
P(z_l<z_s)=\int dz_l\,p_l(z_l)\int_{z_l}^{\infty}p_s(z_s)\,dz_s.
\]

该计算使用 lens/source 的独立边缘分布。报告中的离散网格不等概率为 `1 - sum(q_l * q_s)`，其中 q 是归一化概率质量。它依赖网格；连续变量严格不等的概率通常为 1，不能将这个网格统计量替代红移顺序概率。

## BCNz 本地修改与许可

本地版本保留输入 NB flux/error 的原单位，修正 p(z) 主峰 best-run 选择，输出实际参与 chi-square 的 NB/BB 缩放模型，并对 likelihood 指数减去最小 chi-square 以避免下溢。详见 [LOCAL_MODIFICATIONS.md](vendor/bcnz/LOCAL_MODIFICATIONS.md)。

Photo-lens 原有代码许可见 [LICENSE](LICENSE)；BCNz 单独遵循 [GPL-3.0-or-later](vendor/bcnz/LICENCE.TXT)。根目录 MIT 许可不覆盖第三方 BCNz，完整溯源见 [NOTICE.md](NOTICE.md)。
