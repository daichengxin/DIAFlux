# DIAFluX — DIA 搜索引擎 Rust 重写 · 实施计划

> 目标：将参考实现仓库（Python 版 DIA 搜索引擎）在 `opendia`（crate `diaflux`）中用 Rust 完整重写，功能与结果与参考实现完全一致。
>
> 基准（benchmark 目标）：在 MSV000095138（Astral HeLa，Wallmann et al. Nat. Biotech. 2025）上复现原论文的 >120,000 precursors / ~9,800 protein groups @ 1% FDR。

---

## 1. 目标与范围

### 1.1 关键事实（决定整个计划的形态）

1. **参考实现 的生产搜索后端本身已经是 Rust**：`参考搜索后端`（PyPI 包），暴露 `PeakGroupSelection / PeakGroupScoring / PeakGroupQuantification` 与 `CalibrationEstimator`。Python 侧的 `参考实现/search/{selection,scoring}/*.py` 是「classic/numba 参考后端」，`extraction_backend` 默认已切到 `rust`。
   → 因此 opendia 已完成的 `src/core/{selection,scoring,fdr,lfq}/*` **就是该 Rust 层的等价实现**，搜索数值核心基本已经到位。
2. **用户实际 benchmark 的 critical path 不需要 peptdeep**：`library_prediction.enabled=false`、`transfer_step_enabled=false` 是默认值；HeLa 预测库是外部用 alphaPeptDeep 预生成、以 parquet 喂入的。因此「加载预测库 → 搜索 → 标定/优化循环 → FDR → 输出 → directLFQ」这条默认路径**不依赖任何深度学习**。peptdeep 预测与迁移学习是可选的、可最后做的功能。
3. **三个外部生态依赖是唯一真正的硬骨头**（见 §4）：
   - **peptdeep / torch**（库预测 + 迁移学习）—— 仅可选路径需要；**经 Python 子进程 sidecar 直接调用，不重写**（见 §4.1）；
   - **directlfq / quantselect**（LFQ）—— directlfq 已在 opendia 完成，quantselect 可选；
   - **alpharaw**（Thermo/Sciex 原始文件读取）—— Bruker `.d` 与 mzML 已在 Rust 完成；Thermo `.raw` 已走 ThermoRawFileParser→mzML，无需移植。

### 1.2 「功能完全一致」的验收口径

- **确定性数值代码**（selection / scoring / FDR / directLFQ / 标定 LOESS）：目标**逐位一致**（已通过 fft 移位 bug 修复等逐项对齐验证）。
- **torch/peptdeep 相关**（预测、迁移学习）：走 Python sidecar，与 参考实现 共享同一 peptdeep/torch 实现，天然一致。
- **输出物**：`precursors`、`protein_groups`、`stat`、`internal`、`frag`、`psm` 等表结构与列名与 参考实现 完全一致。

---

## 2. 现状基线

| 参考实现 模块 | 状态 | opendia 现有实现 | 保真度说明 |
|---|---|---|---|
| search/selection | ✅ DONE | [selection/](src/core/selection/) | 完整移植（FFT 卷积、kernel、find_peaks、46 特征中的 selection 侧）；需清理调试代码 |
| search/scoring（46 特征） | ✅ DONE | [scoring/](src/core/scoring/) | 完整移植；**缺口：selection score 未传入**（`score.push(0.0)`） |
| fdr（FeedForwardNN + q 值 + keep_best） | ✅ DONE | [fdr/](src/core/fdr/) | FFNN（BatchNorm+MLP+Softmax）、`get_q_values`、`keep_best` 已移植 |
| directLFQ 定量 | ✅ DONE | [lfq/directlfq.rs](src/core/lfq/directlfq.rs) | normalization + protein 强度估计完整 |
| raw_data（Bruker `.d` + mzML） | ✅ DONE | [io/bruker.rs](src/io/bruker.rs)、[io/mzml.rs](src/io/mzml.rs) | 可读；mzML 合成 2-bin mobility；Bruker m/z 标定为近似（非 timsdata.so） |
| libtransform（flatten 子集） | 🟡 PARTIAL | [libtransform.rs](src/core/libtransform.rs) | 仅 fragment cardinality/position/isotope/loss_type/mz_library |
| 库加载 + decoy | 🟡 PARTIAL | [io/library.rs](src/io/library.rs)、[decoy.rs](src/decoy.rs) | parquet/TSV（DIA-NN 列）；**modifications 被丢弃**；decoy 为 DIA-NN 突变法 |
| orchestration | 🟡 PARTIAL | [orchestration.rs](src/core/orchestration.rs) | 固定 4 步 pipeline（transform→search→quantify），**无标定/优化循环** |
| 输出 | 🟡 PARTIAL | main.rs `write_core_output` | 仅 CSV；无 parquet、无 stat/internal/protein_groups |
| **calibration** | ❌ MISSING | — | LOESS 标定（在 参考实现 中已是 Rust `CalibrationEstimator`，需自实现） |
| **workflow（managers/optimizers/peptidecentric）** | ❌ MISSING | — | 编排层整体缺失 |
| **transferlearning** | ❌ MISSING | — | peptdeep 微调 |
| **outputtransform**（df_builders/grouping/outputaccumulator/protein_fdr/search_plan_output） | ❌ MISSING | — | 输出聚合层 |
| **fragcomp** | ❌ MISSING | — | 片段竞争 |
| **search_plan / search_step** | ❌ MISSING | — | 多步编排 |
| **multiplexing / requantification** | ❌ MISSING | — | TMT 等 |
| reporting / validation / constants / exceptions | 🟡 PARTIAL | [log.rs](src/log.rs) | tracing 已接入；其余待补 |

> 注意：opendia 顶层还有一套 **legacy DIA-NN 风格路径**（`src/fdr.rs`、`cross.rs`、`grouping_model.rs`、`types_diann.rs`、`open_search/`、顶层 `scoring/`、`xic/`、`ms/`、`models/`），与 参考实现 无关。计划见 §4.5 的处理建议。

---

## 3. 目标架构（Rust 模块树）

镜像 参考实现 的 Python 包结构，全部挂在 `src/core/` 下：

```
src/
  main.rs                     # clap CLI（对应 cli.py）
  cli.rs                      # 参数→Config 合并（对应 cli._get_config_from_args）
  log.rs                      # tracing（对应 reporting/logging.py）→ 扩展
  core/
    mod.rs
    error.rs                  # 对应 exceptions.py（thiserror）
    constants.rs              # 对应 constants/keys.py + settings.py（CalibCols、SearchStepFiles…）
    config.rs                 # 对应 workflow/config.py（default.yaml → user → cli → multistep 合并）
    default.yaml              # 嵌入二进制，对应 constants/default.yaml
    validation.rs             # 对应 validation/*.py（schema 结构体/dtype）
    speclib.rs                # SpecLibFlat 等价物：precursor_df / fragment_df（中心交接对象）→ 扩展
    utils.rs                  # 对应 utils.py（expand_path / isotope 列）
    xxhash.rs                 # 已有（classifier store key）
    isotopes.rs               # 已有（同位素包络）
    data.rs                   # DiaData（已有，需补 AccumulationTime 等）
    dia_cycle.rs              # 对应 raw_data/dia_cycle.py（cycle 检测）
    search_plan.rs            # 对应 search_plan.py
    search_step.rs            # 对应 search_step.py
    selection/                # ✅ 已有（清理调试代码）
    scoring/                  # ✅ 已有（补 selection score 透传）
    fdr/                      # ✅ 已有（fdr.rs 补 perform_fdr 编排；classifier.rs 已有）
    calibration/
      mod.rs
      estimator.rs            # LOESS 标定（自实现，对应 CalibrationEstimator 语义）
    fragcomp/
      mod.rs
      fragcomp.rs             # 片段竞争（numba → rayon）
      utils.rs                # candidate_hash（已有部分逻辑）
    libtransform/
      mod.rs                  # 已有 transform_library → 扩展
      loader.rs               # 对应 loader.py
      fasta_digest.rs         # 对应 fasta_digest.py
      harmonize.rs            # 对应 harmonize.py（PrecursorInitializer/AnnotateFasta/IsotopeGenerator/RTNormalization）
      flatten.rs              # 对应 flatten.py
      decoy.rs                # 对应 decoy.py（alphabase decoy → 自实现）
      multiplex.rs            # 对应 multiplex.py
      mbr.rs                  # 对应 mbr.py（IndexBuilder 两级 hash）
      prediction.rs           # 对应 prediction.py（peptdeep 子进程 sidecar 驱动）
    transferlearning/
      mod.rs
      train.rs                # 对应 train.py（finetune 子进程 sidecar 驱动；评估/metrics 留在 Python）
    outputtransform/
      mod.rs
      df_builders.rs          # stat/internal/transfer 表
      grouping.rs             # 最大简约分组（set cover）
      outputaccumulator.rs    # 跨文件累积
      protein_fdr.rs          # 蛋白级 FDR（小 MLP）
      search_plan_output.rs   # 总聚合编排
      utils.rs                # read/write_df、列名映射
      quantification/
        mod.rs
        fragment_accumulator.rs
        quant_builder.rs      # 对接 directlfq（已有）/ quantselect
        quant_output_builder.rs
    lfq/                      # ✅ 已有（directlfq）
    workflow/
      mod.rs
      base.rs                 # WorkflowBase
      config.rs               # Config（或并入 config.rs）
      manager.rs              # BaseManager（pickle → serde 版本化）
      managers/
        mod.rs
        calibration.rs        # CalibrationManager
        fdr.rs                # FDRManager（classifier store + 版本）
        optimization.rs       # OptimizationManager
        raw_file.rs           # RawFileManager（按扩展名分派读取器）
        timing.rs             # TimingManager
      optimizers/
        mod.rs
        base.rs
        automatic.rs          # Automatic{RT,MS2,MS1,Mobility}Optimizer
        targeted.rs           # Targeted{…}
        optimization_lock.rs  # 批计划 + 收敛锁
      peptidecentric/
        mod.rs
        peptidecentric.rs     # PeptideCentricWorkflow
        optimization_handler.rs
        extraction_handler.rs # Classic/Ng 后端（Ng 即现有搜索路径）
        recalibration_handler.rs
        multiplexing_requantification_handler.rs
        transfer_library_requantification_handler.rs
        column_name_handler.rs
        library_init.rs
        ng_mapper.rs          # pandas/SpecLibFlat ↔ 搜索核心的转换 shim
        utils.rs
```

---

## 4. 关键技术决策（含推荐）

### 4.1 peptdeep / torch（库预测 + 迁移学习）—— Python 子进程 sidecar，不重写

现状：`libtransform/prediction.py` 与 `transferlearning/train.py` 依赖 `peptdeep`（torch），模型是 `.pth` 文件（`ms2.pth`、`rt.pth`、`ccs.pth`、`charge.pth`）。

**决策：不重写，走 Python 子进程 sidecar。** Rust 二进制用 `std::process::Command` 调起 Python 脚本（`python -m peptdeep predict ...` 或封装脚本），库进出都走 parquet。这与项目已有做法（ThermoRawFileParser 转 mzML、alphaPeptDeep 预生成 HeLa 库）同构。

- **推理（prediction）**：Rust 侧只做「序列化库 → 调子进程 → 回读 parquet」的驱动。
- **微调（transfer learning）**：Rust 侧只做「调起 finetune 子进程 → 监控输出 → 回读 `.pth`」的驱动；训练/评估/早停全部留在 Python（peptdeep 自带）。
- 由此可**删除**为 candle 引入的 `candle-core/candle-nn` 与 `zip` 依赖（除非 `grouping_model.rs` 仍需）。
- 代价：运行可选路径时需要 Python + peptdeep + torch 环境；默认路径（预测库输入）完全不触发，故不影响 M0–M6。

### 4.2 directlfq / quantselect

- directlfq：**已在 opendia 完成**（`lfq/directlfq.rs`）。
- quantselect（`normalization_method=quantselect` 的另一选项）：独立算法包。**推荐**：优先保证 directlfq 路径（默认），quantselect 作为可选后置项或跳过。

### 4.3 alpharaw（Thermo / Sciex 原始文件）

- Bruker `.d` + mzML 已 Rust 实现。Thermo `.raw` 在用户 benchmark 中已用 ThermoRawFileParser→mzML。
- **推荐**：不移植 alpharaw 原生读取器；把 `.raw` 依赖收敛到 mzML 转换。Bruker m/z 标定的近似问题（非 timsdata.so）列为待验证项（P0）。

### 4.4 序列化 / 并行 / 可复现

- **pickle → serde**：`BaseManager`（Calibration/FDR/Optimization/Timing）当前靠 pickle 持久化并带版本守卫。Rust 用 `serde + bincode/postcard` + `version` 字段复刻，保证跨运行 `reuse_calibration` / `reuse_quant` 行为一致。
- **numba `@njit`/`pjit` → ndarray + rayon**：`fragcomp`、`outputaccumulator`、`quant_builder._ion_hash`、`dia_cycle` 直接 Rayon 并行 + `xxhash` 已具备。
- **随机性**：`random_state`（seed 772 的 OptimizationLock 洗牌、FDR 训练 seed）需逐处复刻，保证可复现。
- **torch 线程**：`torch.set_num_threads` 等在 Rust 中由 rayon/ndarray 的线程数配置取代。

### 4.5 legacy 代码处理

- `open_search/` 是 diaflux 特有功能（非参考实现功能），用户历史诉求「保留 open-search 模型」→ **保留但独立 feature flag**，不纳入本平价范围。
- 顶层 `fdr.rs`、`cross.rs`、`grouping_model.rs`、`types_diann.rs`、顶层 `scoring/`、`xic/`、`ms/`、`models/` 为 legacy DIA-NN 风格 → **删除或移出 DIA 搜索路径**（用户历史诉求「删除死代码」）。

---

## 5. 分阶段实施计划

> 依赖关系：Phase 0 是地基（清理 + 补功能缺口）；Phase 1 提供类型/config 基础；Phase 2–4 是搜索前/后的数值链路；Phase 5–6 是编排核心；Phase 7–9 是输出与聚合；Phase 10 是可选 DL；Phase 11 是 CLI 与验收。

### Phase 0 — 稳定并补全现有核心（地基）✅ 已完成

**目标**：把已移植的搜索核心里所有已知功能缺口与调试残留清掉，让 selection→scoring→FDR→directLFQ 数值链路达到 参考实现 等价。

1. ✅ **清理调试代码**（全部 revert）：
   - [selection.rs](src/core/selection/selection.rs)：`precursor_idx==2740/73960`、m/z 硬编码过滤、`[score-dump]`/`[sel-prec]`、硬编码 `frame/187.4`。
   - [candidate.rs](src/core/scoring/candidate.rs)：`[prof-dump]`、`[decoy-dump]`、`[scoring-dense-debug]`、`G_*` 计数器。
   - [orchestration.rs](src/core/orchestration.rs)：`DIAFORGE_SELECTION_ONLY`、`DIAFORGE_DUMP_FEATURES`、`[fdr-debug]`。
   - [decoy.rs](src/decoy.rs)：`DECOY-MARKER-V2-7f3a9c`、`DECOY-DBG`。
   - [data.rs](src/core/data.rs)：`[frames-debug]`/`[mzml-model]`/`[dense-debug]`。
2. ✅ **修复 selection score 透传**：`ScoreCandidate` 增加 `score: f32`，`build_score_candidates` 从 `Candidate.score`（`peak_score[rank]`）填充，`collect()` 改为 `score.push(sc.score)`。
3. ✅ **保留 modifications**：`Precursor` 增加 `mods`/`mod_sites` 字段；`io/library.rs` 新增 `parse_modified_sequence`（解析 `AA(UniMod:NN)` → `ModName@AA` + 1-indexed 位点，N-term 位点=0，含 UniMod 名称映射表），`speclib.rs` 透传到 `precursor_df`。5 个单元测试覆盖。
4. ✅ **输出 parquet**：`write_core_output` 由 CSV 改为 parquet（PSM 表 + 3 级定量矩阵，对齐 `search_output.file_format: "parquet"`）。
5. ✅ 修 main.rs 的 skip-if-exists 路径不一致（`core_results_{run}.parquet` 现与 parquet 输出一致）。

**退出准则**：`cargo build` 干净；`cargo test --lib` 76 passed（含新增 5 个修饰解析测试）。selection/scoring/FDR/directLFQ 在 MSV000095138 上跑通且 PSM 数不再被上述缺口压住。

### Phase 1 — 基础类型 / 配置 / IO

**目标**：建立 参考实现 层其余部分依赖的公共地基。

| Python | Rust | 说明 |
|---|---|---|
| constants/keys.py, settings.py | constants.rs | `CalibCols`、`SearchStepFiles`、`InferenceStrategy`、`NormalizationMethods`、`NUM_FEATURES=46` 等常量表 |
| exceptions.py | error.rs | thiserror 定义的 `CustomError` 家族（含 error_code） |
| validation/schemas.py | validation.rs | precursor/fragment/candidates/features 的精确 dtype 结构体（是 Rust 数据模型的 schema 规范） |
| workflow/config.py | config.rs | 层级合并（default→default_python→user→cli→multistep），list 覆盖 / dict 递归合并 / 类型校验，与 config.py `_update` 一致 |
| constants/default.yaml | default.yaml（embed） | 完整配置面（§1 已梳理全部段） |
| reporting/logging.py | log.rs（扩展） | progress 级别、JSONL/Log 后端、`print_environment` |
| utils.py | utils.rs | `expand_path`、同位素列名 |

**退出准则**：`Config` 能解析并校验 参考实现 完整 `default.yaml`；`serde` 结构体与 validation schema dtype 一致。

### Phase 2 — 原始数据 + 库加载 + libtransform（搜索上游）

**目标**：让库从「DIA-NN parquet」扩展到 参考实现 的完整 `SpecLibFlat` 语义。

| Python | Rust | 说明 |
|---|---|---|
| raw_data/dia_cycle.py | dia_cycle.rs | cycle 长度（自相关）/cycle 起点/合法性（`_get_cycle_length`、`_get_cycle_start`、`_build_cycle_array`） |
| raw_data/bruker.py（transpose） | io/bruker.rs（补全） | AccumulationTime 读取（P0）、m/z 标定精确化 |
| libtransform/loader.py | libtransform/loader.rs | hdf5/csv/parquet 分派加载 |
| libtransform/harmonize.py | harmonize.rs | PrecursorInitializer / AnnotateFasta / IsotopeGenerator / RTNormalization |
| libtransform/flatten.py | flatten.rs | FlattenLibrary / InitFlatColumns / LogFlatLibraryStats（flat 布局） |
| libtransform/decoy.py | decoy.rs（libtransform 版） | 与现有 DIA-NN 突变法对齐；若目标库由 alphabase decoy 生成则复刻其语义 |
| libtransform/fasta_digest.py | fasta_digest.rs | 酶切 + 修饰（alphabase 的 digest 语义自实现） |
| libtransform/multiplex.py | multiplex.rs | 每通道修饰字符串替换 |
| libtransform/mbr.py | mbr.rs | `IndexBuilder` 两级 hash 查找 + `MbrLibraryBuilder` |
| speclib.rs | speclib.rs（扩展） | 完整 `precursor_df`/`fragment_df`（含 proteins/genes/sequence/mods/mod_sites/flat_frag_start_stop_idx） |

**退出准则**：给定 参考实现 输出的库（hdf5 或转 parquet）能加载出与 参考实现 等价的 `SpecLibFlat`；decoy/flatten/harmonize 单元测试与 alphabase 输出对齐。

### Phase 3 — 标定（Calibration）

**目标**：实现 LOESS 标定 + CalibrationManager（这是默认路径的硬需求，也是 参考实现 中唯一的「缺失 Rust 数值」）。

| Python | Rust | 说明 |
|---|---|---|
| calibration/estimator.py | calibration/estimator.rs | 自实现 `CalibrationEstimator`：`fit`/`predict`/`calc_deviation`/`ci`/`metrics`，`n_kernels`、`transform_deviation`（ppm/percent/absolute）与 参考实现 的 Rust 语义一致 |
| workflow/managers/calibration_manager.py | workflow/managers/calibration.rs | `estimator_groups`（fragment/precursor × mz/rt/mobility），`fit`/`predict`（写 `*_calibrated` 列）/`get_stats` |
| calibration/plot.py | SKIP | matplotlib 仅绘图 |

**退出准则**：给定 `(library, observed)` 列对，`predict` 输出与 参考实现 `CalibrationEstimator` 在相同数据上 ±0.1ppm/±0.1% 内一致。

### Phase 4 — FDR 补全

**目标**：FDR 编排完整（分类器已移植）。

| Python | Rust | 说明 |
|---|---|---|
| fdr/fdr.py | fdr/fdr.rs（扩展） | `perform_fdr`（precursor / precursor_channel_wise / channel 三种策略）、`keep_best`、`_fdr_to_q_values` |
| fdr/classifiers.py | ✅ classifier.rs | 已移植（FFNN） |
| fdr/utils.py | fdr/utils.rs | `train_test_split_`、线程管理 |
| fdr/_fdrx/* | SKIP | 未启用、纯 sklearn |

**退出准则**：`perform_fdr(decoy_strategy=precursor)` 输出 qval/proba 与 参考实现 一致。

### Phase 5 — 工作流编排（managers + optimizers + 优化循环）

**目标**：实现 参考实现 的核心编排：标定/优化循环（这是默认路径里最复杂的纯逻辑，也是 paper 能到 12 万 PSM 的关键）。

| Python | Rust | 说明 |
|---|---|---|
| workflow/managers/{base,optimization,timing,raw_file}.rs | 同 | `BaseManager`（serde 版本化）；`OptimizationManager`（ms1/ms2/rt/mobility_error、num_candidates、score_cutoff、fwhm_rt/mobility）；`TimingManager`；`RawFileManager`（按扩展名分派） |
| workflow/optimizers/base.py | optimizers/base.rs | 抽象 `step/skip/plot/_update_workflow/_update_history` |
| workflow/optimizers/automatic.py | optimizers/automatic.rs | `history_df` 收敛启发式（特征平台期）、`_propose_new_parameter`、4 个具体优化器（RT/MS2/MS1/Mobility） |
| workflow/optimizers/targeted.py | optimizers/targeted.rs | targeted 更新规则 + 收敛条件 |
| workflow/optimizers/optimization_lock.py | optimizers/optimization_lock.rs | seed 772 洗牌、指数批计划、`update_with_extraction/fdr/calibration` |
| workflow/peptidecentric/column_name_handler.py | peptidecentric/column_name_handler.rs | `*_calibrated` vs `*_library` 的动态列选择 |
| workflow/peptidecentric/recalibration_handler.py | recalibration_handler.rs | 重标定 + `num_candidates`/`score_cutoff`/`fwhm_*` 更新 |
| workflow/peptidecentric/optimization_handler.py | optimization_handler.rs | 主循环：`_process_batch`→`_filter_dfs`→`recalibrate`→`_step_all_optimizers`→收敛 |

**退出准则**：优化循环在 MSV000095138 上把 ms1/ms2/rt/mobility_error 收敛到与 参考实现 相同量级；`optimization_lock_target`、`max_steps`/`min_steps`/`max_skips` 语义一致。

### Phase 6 — 提取处理器 + 再定量处理器

**目标**：把现有搜索路径包装成 参考实现 的 `ExtractionHandler`，并补齐 requantification。

| Python | Rust | 说明 |
|---|---|---|
| workflow/peptidecentric/extraction_handler.py | extraction_handler.rs | `NgExtractionHandler`（= 现有 rust 后端，默认）；`ClassicExtractionHandler`（= numba 参考，用于交叉验证可选）；`create_handler` 分派 |
| workflow/peptidecentric/ng/ng_mapper.py | ng_mapper.rs | pandas/SpecLibFlat ↔ 搜索核心 shim（`speclib_to_ng`、`candidates_to_ng`、`parse_candidates`、`to_features_df`、`parse_quantification`、`get_feature_names`） |
| workflow/peptidecentric/peptidecentric.py | peptidecentric.rs | `PeptideCentricWorkflow`：`load`/`search_parameter_optimization`/`extraction`/`requantify`/`requantify_fragments` |
| workflow/peptidecentric/library_init.py | library_init.rs | RT 归一化、按可观测 m/z 过滤、channel filter |
| multiplexing_requantification_handler.py | multiplexing_requantification_handler.rs | `score_grouped=True` 重打分 + `decoy_strategy=channel` |
| transfer_library_requantification_handler.py | transfer_library_requantification_handler.rs | `top_k_fragments=9999` 全片段定量 → `frag.transfer.parquet` |

**退出准则**：per-file 产出 `psm.parquet`/`frag.parquet`/`frag.transfer.parquet` 与 参考实现 列名与语义一致。

### Phase 7 — 片段竞争（fragcomp）

| Python | Rust | 说明 |
|---|---|---|
| fragcomp/fragcomp.py | fragcomp/fragcomp.rs | `_get_fragment_overlap` + `_compete_for_fragments`（O(n²) per window，`pjit`→rayon）；`_add_window_idx` |
| fragcomp/utils.py | fragcomp/utils.rs | `add_frag_start_stop_idx`、`candidate_hash`（64-bit = precursor_idx 低 32 + rank<<32） |

**退出准则**：对同一 DIA 窗口内重叠片段 ≥3 的 PSM 对，低 proba 者被正确 invalidate。

### Phase 8 — 搜索计划（SearchPlan / SearchStep）

| Python | Rust | 说明 |
|---|---|---|
| search_plan.py | search_plan.rs | `SearchPlan`：transfer/library/mbr 三步编排、`_get_optimized_values_config`（从 stat.tsv 取中位数 ms1/ms2） |
| search_step.py | search_step.rs | `SearchStep`：config 初始化、`load_library`、逐文件 `_process_raw_file`、`SearchPlanOutput.build` |

**退出准则**：单步（library）与多步（transfer→library→MBR）路径都能跑通，输出目录结构一致。

### Phase 9 — 输出变换与定量（outputtransform + LFQ）

| Python | Rust | 说明 |
|---|---|---|
| outputtransform/utils.py | outputtransform/utils.rs | read/write_df（arrow）、`INTERNAL_TO_OUTPUT_MAPPING`、`merge_quant_levels_to_psm`、`apply_protein_inference` |
| outputtransform/df_builders.py | df_builders.rs | run_stat / run_internal / transfer_library_stat |
| outputtransform/grouping.py | grouping.rs | `_group_and_parsimony`（贪心 set cover）+ `perform_grouping` |
| outputtransform/protein_fdr.py | protein_fdr.rs | 每蛋白组 7 特征 + 小 MLP + q 值（`StandardScaler`+`MLPClassifier` 用 candle/ndarray 复刻） |
| outputtransform/outputaccumulator.py | outputaccumulator.rs | `AccumulationBroadcaster`（rayon）、`TransferLearningAccumulator`、RT 归一化 |
| outputtransform/quantification/fragment_accumulator.py | fragment_accumulator.rs | 跨文件夹片段累积 |
| outputtransform/quantification/quant_builder.py | quant_builder.rs | `_ion_hash`、`prepare_df`、`direct_lfq`（对接已有 directlfq.rs）/ `quantselect_lfq`（可选） |
| outputtransform/quantification/quant_output_builder.py | quant_output_builder.rs | precursor/peptide/protein 三级 LFQ |
| outputtransform/search_plan_output.py | search_plan_output.rs | `build`：precursor 表→stat/internal→LFQ→MBR 库→transfer 库 |

**退出准则**：`precursors`、`protein_groups`、`stat`、`internal` 与 参考实现 在 MSV000095138 上列名与数值（1% FDR 下数量级）一致。

### Phase 10 — 迁移学习与库预测（可选 DL 路径，peptdeep 子进程 sidecar）

不重写 peptdeep：Rust 侧只做驱动与数据编排，实际预测/微调由 Python 子进程完成。

| Python | Rust | 说明 |
|---|---|---|
| libtransform/prediction.py | libtransform/prediction.rs | 驱动：序列化库（parquet）→ `python -m peptdeep predict`（或封装脚本）→ 回读预测结果（rt/ms2/mobility/charge） |
| transferlearning/train.py | transferlearning/train.rs | 驱动：调起 finetune 子进程（finetune_rt/charge/ccs/ms2，scheduler/early stopping 全在 Python）→ 监控输出 → 回读 `.pth` |
| transferlearning/metrics.py | —（留在 Python） | 评估/指标由 peptdeep 训练过程自身完成，不移植 |

**前置**：封装一个可命令行调用的 `diaflux_peptdeep.py`（`predict` + `finetune` 两个子命令），约定 parquet 进出、退出码与日志格式。
**退出准则**：`library_prediction.enabled=true`（FASTA 输入）与 `transfer_step_enabled=true` 能产出与直接跑 peptdeep 等价的预测库/微调模型。

### Phase 11 — CLI 平价 + 端到端验收

1. **CLI**（cli.rs/main.rs）：支持 `--config`、`--library`、`--fasta`、`--output`、`--quant_directory`、`--check` 等，与 `参考实现` CLI 参数面一致。
2. **金标对照**：对 MSV000095138（sample1 五重复）与 PXD007683 分别跑 参考实现 与 diaflux，比对：
   - precursors / peptides / protein groups 数量 @ 1% FDR（目标 >120k / ~9.8k）；
   - 优化后 ms1/ms2/rt/mobility_error；
   - `precursors`/`protein_groups` 表集合重叠率。
3. **单元测试**：镜像 参考实现 `tests/unit_tests` 结构，逐模块 `cargo test`。

**退出准则**：diaflux 在 MSV000095138 复现原论文指标（>120,000 precursors、~9,800 protein groups @ 1% FDR），并输出结构一致的报告文件。

---

## 6. 验证与基准策略

- **逐模块单测**：每个移植模块写对应 `#[cfg(test)]`，输入用 参考实现 的测试夹具/小数据，断言与 Python 参考输出一致。
- **金标文件（golden files）**：对 selection 的 `log_frag+log_prec` 分数、scoring 的 46 特征矩阵、FDR 的 proba/qval、directLFQ 的蛋白强度，各存一份参考输出，CI 里做容差比对（确定性别 1e-6，torch 相关放宽）。
- **端到端基准**：
  1. PXD007683（小数据集，快）做开发期回归；
  2. MSV000095138（sample1 五重复）做最终验收，对照论文指标。
- **可复现**：固定 `random_state`，双端跑 2 次确认确定性与可复现性一致。

---

## 7. 风险与未决项

1. **peptdeep/torch 迁移学习**（Phase 10）已定为 sidecar 方案，风险由「candle 微调工作量大」转为「依赖运行时 Python+torch 环境、子进程健壮性（退出码/日志/数据约定）」。可**将 Phase 10 拆为可选里程碑**，不影响 Phase 0–9 达成「预测库输入 + 常规搜索」的完整平价。
2. **quantselect**：`normalization_method=quantselect` 需另实现或跳过（默认 directlfq 已覆盖）。
3. **Bruker m/z 标定近似**：非 timsdata.so 的 metadata 二次曲线可能有误差，需 P0 验证（对用户 Bruker 数据）。
4. **mzML 合成 2-bin mobility**：对无 IMS 的 mzML 是近似，需确认与 参考实现 对同一 mzML 的处理一致（参考实现 走 alpharaw，可能读数不同）。
5. **hdf5 库格式**：参考实现 `SpecLibFlat` 是 alphabase hdf5；diaflux 目前读 parquet。需决定「用 parquet 作为统一中间格式」还是「自实现 hdf5 读取」——**推荐统一到 parquet**（已有 pyarrow 生态，且用户预测库已是 parquet）。
6. **数值确定性**：selection/scoring/FDR/directLFQ/LOESS 力求逐位一致；torch 部分放宽到统计一致。

---

## 8. 里程碑（建议顺序与依赖）

| 里程碑 | 阶段 | 可交付 | 依赖 |
|---|---|---|---|
| M0 地基稳定 | 0 | 搜索核心清理 + score 透传 + 修饰保留 + parquet 输出 | — |
| M1 基础与配置 | 1 | Config/常量/错误/schema/日志 | M0 |
| M2 库与数据上游 | 2 | dia_cycle + libtransform 全套 + 完整 SpecLibFlat | M1 |
| M3 标定 + FDR | 3,4 | LOESS + perform_fdr | M1, M2 |
| M4 编排核心 | 5,6 | 优化循环 + 提取/再定量处理器（**默认路径闭环**） | M3 |
| M5 搜索计划 + 输出 | 8,9,7 | SearchPlan/SearchStep + 输出聚合 + fragcomp | M4 |
| M6 CLI + 验收 | 11 | 端到端 MSV000095138 达标 | M5 |
| M7 可选 DL（sidecar） | 10 | 库预测 + 迁移学习（peptdeep 子进程） | M5（独立） |

> 关键里程碑是 **M4**（默认路径闭环）与 **M6**（复现论文指标）。M7（DL）可与 M4–M6 并行、独立交付。
