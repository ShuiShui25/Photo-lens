# BCNz 本地修改记录

本文档记录本项目对上游 BCNz 源码所做的本地修改。今后每次直接修改
`photometry/bcnz/` 下的 BCNz 包源码时，都应在这里新增一条记录，并说明修改原因、
影响范围和验证方式。Notebook 自身的修改不记录在本文档中。

## 2026-06-21：移除读取 NB 测光时的隐式 `×0.625` 缩放

- 修改文件：`bcnz/data/paudm_coadd.py`
- 修改函数：`load_coadd_file()`
- 修改内容：
  - 移除 `coadd['flux'] *= 0.625`。
  - 移除 `coadd['flux_error'] *= 0.625`。
  - 删除包含相同实验性缩放的已注释旧版 `load_coadd_file()`，避免后续检索和维护
    时误认为该缩放仍是受支持的处理步骤。
  - 更新函数文档，明确输入 catalogue 中的 flux 和 flux error 将按原单位保留，
    BCNz 不再在读取阶段执行隐式光度缩放。
- 修改原因：
  - 原代码注释表明 `0.625` 是用于测试影响的临时实验系数，并非已确认的物理或
    标定转换。
  - 该缩放只作用于 PAUS NB，而后续追加的 HSC BB 不受影响，因此会人为改变
    NB/BB 的相对归一化。
  - 这既会影响 SED 图中 NB 与 BB 的相对高度，也可能改变 BCNz 拟合结果。
- 行为变化：
  - 修改前：载入后的 NB flux 和 flux error 均为 CSV 输入值的 `0.625` 倍。
  - 修改后：载入后的 NB flux 和 flux error 与 CSV 输入值一致。
- 注意事项：
  - 旧的 photo-z 缓存或输出可能包含经过 `0.625` 缩放的数据，不能用于验证新行为。
  - 比较修改前后的结果时，应使用新的输出目录或清除对应运行的输入 catalogue、
    中间结果和 photo-z 输出，避免静默复用旧结果。
  - 此修改不会自动改变已经写入磁盘的 parquet/CSV 文件；必须重新执行调用
  `load_coadd_file()` 的流程。

## 2026-06-21：修复 best-run 选择和输出 best model 与实际拟合不一致

- 修改文件：
  - `bcnz/fit/libpzqual.py`
  - `bcnz/fit/photoz.py`
- 修改内容：
  - 将 `get_pzcat()` 中用于定位边缘化 p(z) 主峰的
    `pz.argmin(dim='z')` 修正为 `pz.argmax(dim='z')`。
  - `_core_allz()` 除 chi-square 和模板系数外，同时返回实际用于计算 chi-square
    的完整拟合 flux `Fx`。
  - `minimize_all_z()` 为每个 BCNz run 保存对应的 `Fx`。
  - `get_model()` 直接从 `Fx` 选取主峰 redshift 和最佳 run 的模型，不再从原始模板
    与系数重新构造一个缺少 NB 相对缩放的近似模型。
- 修改原因：
  - 原实现注释要求选择 p(z) 主峰，但 `argmin` 实际选中了概率最低的 redshift，
    导致 `best_run` 可能不属于 `zb` 主峰。
  - BCNz 的 chi-square 计算对 NB 使用额外的相对缩放 `k(z)`，原 `get_model()`
    没有应用该系数；因此输出的 `model_pau_nb*` 与真正进入 chi-square 的模型不一致。
  - 该问题会使诊断 SED 图看起来没有拟合明显的窄带特征，即使内部拟合采用的是
    另一组 flux。
- 行为变化：
  - `best_run` 现在是在边缘化 p(z) 最大值所在 redshift 处选择 chi-square 最小的
    run。
  - `model_*` 和 `modelz0_*` 现在与 BCNz 实际计算 chi-square 时使用的 NB/BB-scaled
    flux 处于相同输入单位，可直接与观测 flux 比较。
- 兼容性说明：
  - `minimize_all_z()` 的内部返回值由两个增加为三个；本仓库内唯一调用方
    `photoz()` 已同步更新。
  - 已保存的 `pzcat_*.pq` 不会自动更新，必须重新运行 photo-z。

## 2026-08-09：以最小 chi-square 平移 p(z) 指数，避免 odds 下溢

- 修改文件：`bcnz/fit/libpzqual.py`
- 修改函数：`get_pzcat()`
- 修改内容：将

  ```python
  pz = np.exp(-0.5 * chi2)
  ```

  改为

  ```python
  chi2_min = chi2.min(dim=['run', 'z'])
  pz = np.exp(-0.5 * (chi2 - chi2_min))
  ```

- 修改原因：当同一对象在全部 run/z 点的 chi-square 都很大时，原指数可全部下溢为
  零；后续 clipped normalization 无法恢复相对似然，进而使 p(z)、odds 和 width 失真。
- 数学性质：对每个 `ref_id` 减去同一个 `chi2_min`，只给所有未归一化 likelihood
  乘同一常数。因此在不下溢时，归一化 p(z)、`zb`、odds、best run 严格不变；最大
  指数项恒为 `exp(0)=1`，从而避免全零下溢。
- 影响范围：所有调用本地 BCNz `get_pzcat()` 的分析均采用该稳定计算；旧 parquet 或
  p(z) 输出不能与新结果混用，必须重新执行。
