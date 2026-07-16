# TTMD6 位移幅值波形分析：单一复现链

本目录是本研究唯一的代码入口。它从研究者自行取得的 `TTMD6.rar` 开始，完成档案身份校验、解压、坐标质量控制、波形派生、固定六标签有限集合分解、6 类动作特异节点级约束 REML、5000 次运动员整簇 Bootstrap、排除名义长度超过 200 帧的不平衡 REML、1000 次单元内平衡重抽样、结构性缺失与高跳变敏感性、固定关节点组成敏感性、档案顺序相关诊断、AR(1)工作情景、峰值配准、逐一剔除运动员、Bootstrap区间诊断、逐节点边际区间和全部投稿图重建。

该链条不读取既往结果、旧 README、旧表格或 `QC_WARNING.txt`。所有输出都由当前目录 `scripts/` 中的定版脚本重新计算；结束前还会执行数值回归检查，并主动拒绝任何混入的旧 `QC_WARNING.txt`。

## 永久归档与引用

论文分析对应的冻结版本为 GitHub `v1.0.0`，标签提交为 `f5c0562d8b0abfe79cbd20971efc6dc2ea6fd022`。精确复现时应引用 Zenodo 版本 DOI [10.5281/zenodo.21382967](https://doi.org/10.5281/zenodo.21382967)；跨版本引用可使用长期 DOI [10.5281/zenodo.21382966](https://doi.org/10.5281/zenodo.21382966)，该 DOI 始终解析到最新版本。Zenodo 归档包大小为 `9201472` bytes，MD5 为 `81e2dbd99d85ce45d18dbb8d60aa6438`，SHA-256 为 `6da90f49aaebe52662f94f14b55c7e2b5126f125f3ab3b6fd25beb53c96c2230`，ZIP 完整性检查通过。该归档不含第三方 TTMD6 原始坐标压缩包。

## 1. 输入身份

仅接受与本研究使用的 Scientific Reports 2024 官方补充档案逐字节一致的文件：

- 文件名：`TTMD6.rar`
- 字节数：`341074031`
- MD5：`1c9ce9cbf79dd35dd22f16a7199e2a8c`
- SHA-256：`93d1b52a470f14b9dc0ba0600959bff921be891a3da1b71e609bd328224b354d`

官方获取地址为原论文关联的 [Springer Nature Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-024-54150-5/MediaObjects/41598_2024_54150_MOESM1_ESM.rar)，对应的 [Scientific Reports 原论文](https://doi.org/10.1038/s41598-024-54150-5)提供数据来源与上下文。2026年7月15日已从该官方地址完整下载文件，并与本地分析输入核对：字节数、MD5、SHA-256均一致，逐字节比较无差异。因此，Springer Nature 官方补充文件就是本研究实际分析的精确档案；官方下载名虽为 `41598_2024_54150_MOESM1_ESM.rar`，其内容与本地 `TTMD6.rar` 完全相同。

代码包不再分发第三方坐标档案。复现者须依法从上述原始发布位置取得该档案，并通过 `--rar` 指定其路径。后续 Figshare 记录的登记字节数不同，不能在未通过上述精确身份核验时替代官方补充档案。哈希或字节数不符时，程序会在任何分析开始前停止。

## 2. 环境

要求：

- 精确复现使用 Python 3.12；Python 3.10 及以上仅用于范围依赖下的兼容性测试；
- `bsdtar`，且支持 RAR5；
- NumPy 2.x、pandas 2/3、SciPy 1.x、Matplotlib 3.x、Pillow 10–12；
- Arial、Helvetica、Liberation Sans 或 DejaVu Sans 中至少一种；公开仓库统一生成英文图件。

建议在全新虚拟环境中安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

精确复现归档的论文结果时，应使用 Python 3.12 和 `requirements-lock.txt`；已记录的冷复现环境为 Python 3.12.13。`requirements.txt` 给出适用于 Python 3.10 及以上版本的较宽依赖范围，供开发和兼容性测试使用，不能替代归档软件环境的精确说明。

可在运行前核验本发布链代码未被改动：

```bash
shasum -a 256 -c SHA256SUMS
```

在 Debian/Ubuntu 上可另外安装 `libarchive-tools`；macOS 系统自带可用的 `bsdtar`。

## 3. 一键完整复现

从本目录运行：

```bash
python reproduce_all.py \
  --rar /绝对路径/TTMD6.rar \
  --out /绝对路径/ttmd6_reproduced
```

完整分析默认固定为：

- 随机种子 `20260712`；
- 运动员整簇 Bootstrap `5000` 次；
- 单元内无放回平衡重抽样 `1000` 次；
- 归一化时间节点 `200` 个。

默认把坐标解压到系统临时目录，分析完成或报错后清理，不会在发布结果中复制第三方原始数据。若确需保留解压文件，可显式加 `--keep-extracted`。输出目录必须为空；中断后若要从头覆盖同一目录，可加 `--resume`，但程序仍会逐步重跑并在末尾检查是否混入旧文件。

只做输入、依赖、脚本与命令图检查，不执行分析：

```bash
python reproduce_all.py \
  --rar /绝对路径/TTMD6.rar \
  --out /绝对路径/ttmd6_reproduced \
  --plan-only
```

参数 `--n-bootstrap` 和 `--n-balanced-resamples` 仅用于代码开发时的快速冒烟测试。稿件结果必须使用默认的 5000 和 1000，改变这两个值后不应把输出当作定版结果。

## 4. 分析顺序与边界

1. 校验 RAR 字节数、MD5 与 SHA-256，并核对 12000 对球拍—人体 CSV。
2. 审计全部 40 个代码；主推断队列只含代码 1–30 的 9000 对文件，代码 31–40 的 3000 对文件隔离但不静默混入。
3. 确定性折叠 50 个球拍文件与 50 个人体文件的完全相同 200 行重复块；按文件名长度字段去除可确认的尾部零填充。名义长度超过 200 帧表示档案存储边界，不能称为“真实截断帧”。
4. 球拍表征为相邻帧三维位移幅值；人体表征为相邻帧两端均非 `[0,0,0]` 的可用关节点位移幅值均值。主分析不填补坐标、不温莎化、不删除整条试次。
5. 跨动作模型把六个英文标签视为固定有限集合：动作项报告为“噪声校正的有限集合动作对比离差” `D_A`，不解释为从随机动作总体抽样得到的动作方差；6 个动作特异模型承担试次数设计推断。
6. 节点级非负约束 REML 后，以梯形积分得到 `L²` 迹型相对信度 `R_L2,a(n)=B_a/[B_a+W_a/n]`。达到阈值的试次数由积分分量计算，而不是对逐节点显著性结果计数。概化理论只提供重复测量误差按 `W/n` 缩减的设计逻辑，本指标不称标准G系数。
7. 不确定性以运动员为整簇进行 5000 次 Bootstrap；不会把 9000 条试次误当成 9000 个跨运动员推断的独立单位。
8. 敏感性分析包括：名义长度 `>200` 帧排除、不平衡 REML、单元内平衡重抽样、11 点 Hampel 型局部高值处理、含 8 个始终可见关节点的固定组成，以及排除剩余结构性缺失试次的 13 关节点固定组成。这里的局部高值规则使用研究自定义的一侧阈值与插值步骤，不是标准 Hampel 滤波器。
9. 档案数字顺序只作为顺序代理，不能等同经核验的采集时间。代码同时报告档案顺序 `L²` 迹相关、`phi=0/0.10/0.20/0.30` 的 AR(1)工作情景、同一球拍峰值变换作用于成对球拍—人体波形的配准敏感性、逐一剔除运动员、质量规则影响、百分位/基本/BCa区间与分位点Monte Carlo稳定性。

这里的“速度”文件名沿用早期缓存命名，实际量是未除以帧间隔的坐标位移幅值。结果不能解释为物理速度、击球效果、分类性能或跨批次追踪有效性。

## 5. 关键输出

运行成功后，`OUT` 中至少包括：

- `REPRODUCTION_STATUS.json`：输入身份、环境、随机设置和全部硬性回归检查；只有 `status: PASS` 才视为成功复现；
- `REPRODUCTION_COMMANDS.log`：按 UTC 记录实际执行的每一条命令与退出状态；
- `FILE_INVENTORY.csv`：除可选解压坐标外全部输出的字节数与 SHA-256；
- `work/preparation/`：12000 对全档案清单、9000 对主队列清单、完全重复对审计与派生缓存；
- `work/coordinate_qc/`：结构性零三元组、有效坐标跳变和逐试次质量审计；
- `work/structural_arrays/`：结构性缺失处理后的主分析波形和审计；
- `work/reanalysis_structural/results/`：12 个动作×表征模型的点估计、D-study、5000 次整簇 Bootstrap、排除 `>200` 帧后的不平衡 REML、1000 次平衡重抽样、REML 自检、试次级审计和当前 `QC_NOTE.txt`；
- `work/reanalysis_structural/global/`：跨动作方差分解及整簇 Bootstrap；
- `work/reanalysis_structural/global/tables/Table_S_GlobalPointwiseBootstrapBands.csv`：固定六标签分解的5000次运动员整簇逐节点边际区间；这些不是同时置信带，不能用于局部显著性判定；
- `work/reanalysis_structural/assumption_influence/`：档案顺序审计与相关、AR(1)情景、峰值配准、逐一剔除运动员、运动员贡献、BD质量标记分布、Bootstrap诊断和成对动作描述性差值；
- `work/reanalysis_hampel/`：孤立高跳变处理敏感性；
- `work/reanalysis_fixed8/`：8 个始终可见关节点的固定组成敏感性；
- `work/reanalysis_structural/Table_S_CompleteCaseZeroMarkerSensitivity.csv`：整试次完全案例敏感性；
- `work/reanalysis_structural/Table_S_Fixed13JointCompleteCaseSensitivity.csv`：13 关节点固定组成完全案例敏感性；
- `work/reanalysis_structural/Table_S_ContinuousThresholdBootstrap5000_*.csv`：连续和向上取整试次数阈值；
- `figures_final/`：4张主图与3张补充图的 PNG、600 dpi LZW TIFF、PDF、可编辑文本SVG、每图源数据CSV和自动QA清单。

公开版本已统一机器字段：积分指标使用 `R_L2_m*`，阈值使用 `required_n_R_L2_*`，逐节点展示量使用 `pointwise_relative_reliability_m*`。概化理论仅提供重复测量误差按 `W/n` 缩减的设计逻辑；本指标不称为标准G系数。

## 6. 自动成功标准

入口程序在结束前强制核对：

- 12000 对全档案、9000 对主队列、3000 对隔离队列；
- 主队列中 1180 条名义长度 `>200` 帧；
- 50 个球拍与 50 个人体 400 行完全重复块；
- 主队列内部无完全重复试次对；
- 人体 190 条结构性零三元组受影响试次，球拍为 0 条；
- 主分析、Hampel 型局部高值敏感性和固定 8 关节点敏感性的 12 组 `R_L2≥0.90` 试次数回归值；
- 三套动作特异模型全部 REML 自检通过；
- 当前 `QC_NOTE.txt` 存在、任何旧 `QC_WARNING.txt` 不存在；
- 档案顺序、AR(1)、配准、LOPO、BD质量规则、Bootstrap诊断和逐节点边际区间表均生成；
- 4张主图与3张补充图均生成，尺寸、分辨率、TIFF压缩、SVG文字可编辑性和文字边界QA全部通过。

任一条件不满足，程序以非零状态退出，不写出 `PASS`。

## 7. 脚本对应关系

- `scripts/01_prepare_ttmd6_waveforms.py`：档案审计、成对清单与基础波形缓存；
- `scripts/02_global_context_and_bootstrap.py`：跨动作方差分解与整簇 Bootstrap；
- `scripts/03_audit_coordinate_missingness.py`：结构性零与坐标跳变只读审计；
- `scripts/04_prepare_structural_qc_arrays.py`：主分析、Hampel 型局部高值规则、固定关节点波形派生；
- `scripts/run_action_specific_reml.py`：6 动作×2 表征节点级约束 REML、D-study 与敏感性重抽样；
- `scripts/07_complete_case_zero_marker_sensitivity.py`：人体整试次完全案例敏感性；
- `scripts/08_continuous_threshold_bootstrap.py`：连续阈值与整数试次数 Bootstrap 汇总；
- `scripts/09_assumption_influence_sensitivity.py`：档案顺序/AR(1)、峰值配准、LOPO、质量规则影响与Bootstrap区间诊断；
- `scripts/10_global_pointwise_bootstrap.py`：固定六标签分解的运动员整簇逐节点边际区间；
- `scripts/06_make_publication_figures.py`：4张主图、3张补充图、逐图源数据与图形QA。

本发布链的目标是让所有定量陈述可追溯、可重跑、可失败；它不把单次成功运行等同于对数据采集质量或外部效度的额外保证。
