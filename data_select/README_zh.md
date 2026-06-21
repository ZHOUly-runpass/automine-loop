# AutoMine-Loop：自动驾驶无标签数据智能筛选与数据闭环挖掘工具

## 1. 项目概述

AutoMine-Loop 是一套面向**自动驾驶海量无标签图像数据**的智能筛选与数据闭环挖掘框架。其核心目标是：从数以万计的自动驾驶路采帧中，用最小的人工标注成本，挑选出**最有价值**的样本送给标注团队，从而高效提升下游感知模型的性能。

整体算法流水线分为9个阶段：

构建帧清单 → 
隐藏真实标签 → 
计算图像质量特征 → 
提取语义嵌入向量 → 
构建向量检索引擎 → 
聚类去重 → 
训练轻量场景分类器 → 
主动学习式样本筛选 → 
释放标注结果并迭代

---

## 2. 代码文件结构树（仅代码文件，不含 `data/` 数据目录）

```
dat
a_select/
├── pyproject.toml                              # 项目元数据、依赖声明、CLI入口
├── configs/
│   ├── dataset.yaml                            # 数据集划分策略、路径配置7 
│   ├── embedding.yaml                          # 嵌入提取模型、检索索引参数
│   └── active_learning.yaml                    # 主动学习轮次、采样权重、分类器标签
├── src/
│   └── automine_loop/
│       ├── __init__.py                         # 包版本声明
│       ├── cli.py                              # 命令行入口（demo / search 子命令）
│       ├── io.py                               # 通用I/O：读写Parquet/CSV/YAML
│       ├── dataset/
│       │   ├── __init__.py
│       │   ├── demo_data.py                    # 生成合成演示数据集（含背景色彩与标注逻辑）
│       │   └── nuscenes_adapter.py             # nuScenes真实数据集适配器（构建清单+Oracle标签）
│       ├── embedding/
│       │   ├── __init__.py
│       │   ├── encoders.py                     # 颜色直方图编码器（局部确定性基础编码器）
│       │   ├── clip_encoder.py                 # CLIP兼容性占位编码器（继承基础编码器）
│       │   └── dinov2_encoder.py               # DINOv2兼容性占位编码器（继承基础编码器）
│       ├── quality/
│       │   ├── __init__.py
│       │   ├── features.py                     # 图像质量特征计算（亮度、模糊度、平均哈希、损坏检测）
│       │   └── rules.py                        # 基于规则的质量评分（过亮/过暗+清晰度+可用性加权）
│       ├── retrieval/
│       │   ├── __init__.py
│       │   └── index.py                        # 平面内积向量索引（L2归一化+Top-K相似检索）
│       ├── clustering/
│       │   ├── __init__.py
│       │   └── cluster.py                      # PCA降维+K-Means聚类（HDBSCAN可选）；输出簇ID+稀有度
│       ├── classifier/
│       │   ├── __init__.py
│       │   ├── linear_probe.py                 # 多标签逻辑回归线性探针（训练+预测+不确定性计算）
│       │   └── mlp.py                          # MLP占位入口（当前回退到线性探针）
│       ├── active_learning/
│       │   ├── __init__.py
│       │   └── sampler.py                      # 主动采样引擎（随机/不确定性/核心集/规则/组合策略）
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py                      # 多标签评估指标（macro-F1/Precision/Recall、Precision@K）
│       └── pipeline/
│           ├── __init__.py
│           └── run_demo.py                     # 完整演示流水线（串联所有模块，3轮主动学习迭代）
├── app/
│   └── streamlit_app.py                        # Streamlit可视化界面（概览/图像检索/聚类浏览/采样队列）
├── tests/
│   └── test_smoke.py                           # 冒烟测试（端到端验证流水线可运行）
└── reports/
    ├── demo_report.md                          # 演示实验报告（含3轮评估指标）
    └── project_plan.md                         # MVP实施计划与后续扩展方向
```

---

## 3. 各模块算法功能详解

### 3.1 项目入口与配置层

#### `pyproject.toml`
- **核心依赖**：`numpy`、`pandas`、`scikit-learn`、`Pillow`、`matplotlib`、`streamlit`、`PyYAML`。
- **可选依赖（full模式）**：`torch`/`torchvision`/`transformers`（深度学习编码器）、`faiss-cpu`（大规模向量检索）、`hdbscan`（密度聚类）、`nuscenes-devkit`（nuScenes解析）、`pyarrow`（Parquet高效读写）、`pyspark`（分布式处理）。
- **CLI入口**：``automine`` 命令映射到 `automine_loop.cli:main`。

#### `configs/dataset.yaml`
定义数据集划分策略：
- **按场景（`split_by: scene_token`）切分**，避免同一场景的帧散落在训练/测试集中（防止时序泄漏）。
- **划分比例**：初始标注池5%、无标签池75%、验证集10%、测试集10%。
- 指定 manifest、oracle_labels、quality_features 的输出路径。

#### `configs/embedding.yaml`
- **嵌入模型**：默认 `color_histogram`（96维颜色直方图，无外部依赖即可运行）。
- **归一化**：`normalize: true`，确保向量检索引擎使用内积/余弦相似度。
- **检索**：`flat_ip`（平面内积），`top_k=20`。

#### `configs/active_learning.yaml`
- **主动学习轮次**：5轮。
- **每轮查询比例**：2%（最小8个样本）。
- **组合策略权重**：不确定性0.35 + 多样性0.25 + 稀有度0.20 + badcase偏好0.15，冗余惩罚0.15。
- **分类器标签**：`is_vru`（弱势道路使用者）、`is_large_vehicle`（大型车辆）、`is_dense`（密集场景）、`is_occlusion`（遮挡）、`rare_class`（稀有类别）。

---

### 3.2 通用基础设施

#### `src/automine_loop/io.py`
提供项目通用的I/O工具：
- `save_table()`：将DataFrame写入Parquet，若`pyarrow`不可用则自动回退为CSV。
- `read_table()`：优先读取Parquet，不存在则尝试CSV，保证在不安装可选依赖时仍可运行。
- `load_yaml()`：加载YAML配置文件。
- `ensure_parent()`：自动创建目标目录的父目录。

---

### 3.3 数据层（`dataset/`）
生成自动驾驶场景帧的清单（manifest）和隐藏的真实标签（oracle_labels）。

#### `src/automine_loop/dataset/demo_data.py` — 合成演示数据集生成器
由于真实数据集（如nuScenes）需要额外授权下载，本项目内置一套合成数据生成器，使MVP可在无外部数据的环境下运行。

**算法逻辑**：
1. 生成12个场景（scene），每个场景12帧，共144帧图像。
2. 每个场景赋予一个 **"主题"（theme，0~4循环）**，决定图像的背景颜色倾向和标注规律：
   - **theme=0（绿色调）**：VRU场景（行人和骑行者的椭圆标注）
   - **theme=1（红色调）**：大型车辆场景（大矩形标注）
   - **theme=2（蓝色调）**：遮挡场景（黑色矩形模拟遮挡）
   - **theme=3（随机色）**：密集场景（8个目标 vs 默认3个）
   - **theme=4（随机）**：稀有类别场景（菱形标注，仅3帧命中）
3. 每帧使用PIL绘制彩色背景+几何图形模拟目标，保存为JPEG图像。
4. 按场景级别划分split（前5%场景为labeled、5%~80%为unlabeled、80%~90%为val、90%~100%为test），**关键设计：同一场景的所有帧划分到同一split，杜绝时序泄漏**。
5. 输出manifest（记录帧ID、场景ID、图像路径、split）和oracle_labels（5维二值标签）。

#### `src/automine_loop/dataset/nuscenes_adapter.py` — nuScenes真实数据适配器
当用户有nuScenes数据集时，本模块将真实数据转换为统一的数据格式。

**算法逻辑**：
1. 调用 `nuscenes-devkit` 解析nuScenes的scene/sample/sample_data层级结构。
2. 按场景随机打乱后，按相同比例划分labeled/unlabeled/val/test（场景级别切分，避免时序泄漏）。
3. 遍历每个场景的帧链（从 `first_sample_token` 沿 `next` 指针直到末尾），提取指定相机（默认`CAM_FRONT`）的图像路径。
4. 从nuScenes标注中提取真实标签（`_oracle_from_annotations`函数）：
   - `is_vru`：标注类别是否包含pedestrian/bicycle/motorcycle
   - `is_large_vehicle`：是否包含bus/truck/trailer/construction
   - `is_dense`：标注框数量≥12
   - `is_occlusion`：标注可见度是否为低
   - `rare_class`：是否包含construction/trailer稀有类别
5. **重要安全约束**：真实标签存储在`oracle_labels`中，仅用于评估和模拟标注释放，**绝不直接用于训练**，以真实模拟无标签数据挖掘场景。

---

### 3.4 图像质量评估层（`quality/`）
在无标签场景下，图像质量本身可作为粗筛选的规则基线。

#### `src/automine_loop/quality/features.py` — 质量特征提取器
对manifest中的每一帧图像计算4个质量特征：

| 特征  | 算法 | 用途 |
|------|------|------|
| `brightness`（亮度） | 转为灰度图(64×64)，取像素均值 | 识别过暗/过亮图像 |
| `blur_score`（模糊度） | 应用拉普拉斯边缘检测滤波后取方差 | 清晰度越高方差越大，模糊图像方差低 |
| `phash`（感知哈希） | **平均哈希（aHash）**：将64×64灰度图缩为8×8（每8×8块取均值），与全局均值比较生成64位二进制指纹 | 图像去重、近似重复检测 |
| `is_corrupted`（损坏标记） | 图像打开异常时标记为1 | 排除无法读取的损坏帧 |

**平均哈希算法细节**（`average_hash`函数）：
1. 输入64×64灰度矩阵
2. 重塑为8×8×8×8，在axis=(1,3)取均值，得到一个8×8缩略图
3. 将每个像素与缩略图全局均值比较，大于均值为1，反之为0
4. 将64个bit拼接为16位十六进制字符串

#### `src/automine_loop/quality/rules.py` — 规则评分引擎
基于质量特征计算综合规则分数，用于粗筛选：

```python
score = 0.55 × dark_or_bright + 0.25 × sharp + 0.20 × usable
```

- **dark_or_bright**（权重0.55）：亮度<70或>200的"极端光照"帧→1，此类帧对模型鲁棒性有价值
- **sharp**（权重0.25）：模糊度在前25%分位数以上的"清晰"帧→1，确保样本质量
- **usable**（权重0.20）：图像未损坏→1

---

### 3.5 嵌入提取层（`embedding/`）
将图像转化为可用于语义检索和聚类的向量表示。

#### `src/automine_loop/embedding/encoders.py` — 颜色直方图编码器（`ColorHistogramEncoder`）
**MVP默认编码器**，无需GPU，纯NumPy实现，保证在任何环境下可运行。

**算法原理**：
1. 将图像统一resize为224×224 RGB。
2. 对R、G、B三个通道分别计算32-bin直方图（范围0-255，密度归一化）。
3. 拼接三个通道的直方图，得到32×3=96维特征向量。
4. L2归一化处理，使向量模长为1。
5. 若图像加载失败，返回全零向量。

**`encode_manifest()`**：批量处理manifest中的所有帧，返回(N, 96)的嵌入矩阵和元数据DataFrame。

#### `src/automine_loop/embedding/clip_encoder.py` & `dinov2_encoder.py`
- **CLIP编码器**和**DINOv2编码器**是占位类，继承自`ColorHistogramEncoder`。
- 目的是**保持模块接口一致**，使得未来替换为真正的CLIP/DINOv2模型时，流水线其他模块无需修改。
- 当前使用不同的`name`属性区分（`clip_fallback_color_histogram` / `dinov2_fallback_color_histogram`）。

---

### 3.6 向量检索层（`retrieval/`）
构建基于嵌入向量的相似图像检索引擎。

#### `src/automine_loop/retrieval/index.py` — 平面内积索引（`FlatIPIndex`）
**算法设计**（不使用FAISS等外部库，纯NumPy实现）：

1. **初始化**：存储嵌入矩阵 + record_id数组，对嵌入执行L2归一化（向量模长为1）。
2. **搜索（`search_by_vector`）**：对查询向量做L2归一化后，计算`embeddings @ query`（内积），取Top-K最高得分的帧。**归一化后内积等价于余弦相似度**。
3. **按ID搜索（`search_by_record_id`）**：先在record_ids中定位查询帧的索引，再调用search_by_vector。
4. **持久化**：使用`np.savez_compressed`压缩存储嵌入和ID映射。

`build_index()` 工厂函数从嵌入矩阵和元数据构建FlatIPIndex实例。

---

### 3.7 聚类层（`clustering/`）
通过无监督聚类实现**数据去重**和**稀有度标注**。

#### `src/automine_loop/clustering/cluster.py` — 聚类引擎
**算法流程**：

1. **PCA降维**：若嵌入维度>32且样本数>32，使用PCA降至32维（减少K-Means在高维空间的维度灾难问题）。
2. **聚类**：
   - 默认**K-Means**：聚类数`k = min(max(2, n_clusters), N)`，使用scikit-learn实现。
   - 可选**HDBSCAN**（密度聚类）：当`method="hdbscan"`时尝试，若库不可用则回退为K-Means。
3. **输出特征**：
   - `cluster_id`：每个样本的簇标签（HDBSCAN中-1表示噪声）
   - `distance_to_center`：到簇中心的欧氏距离
   - `cluster_size`：所属簇的样本数量
   - `is_noise`：是否为噪声点
4. **稀有度设计思路**：`cluster_size`越小的簇，其样本越"稀有"——这些在数据集中不常见的场景对模型训练更有价值。

---

### 3.8 分类器层（`classifier/`）
在主动学习的每轮迭代中，使用已标注样本训练一个轻量分类器，用于预测未标注样本的不确定性。

#### `src/automine_loop/classifier/linear_probe.py` — 多标签逻辑回归探针
**`MultiLabelProbe`类**：
- 对5个标签分别训练一个独立的**逻辑回归分类器**（`LogisticRegression`）。
- 使用`StandardScaler`标准化输入特征。
- 使用`class_weight="balanced"`处理类别不平衡。
- 若某标签在训练集中只有单一取值，则跳过训练，直接返回常量值。
- `predict_proba()`返回(N, 5)的概率矩阵。

**`predict_probabilities()`**：
- 对每个标签计算预测概率`p_{label}`。
- **不确定性度量**：计算每个样本的**平均标准化交叉熵**：
  ```
  H = Σ(-p·log(p) - (1-p)·log(1-p)) / (k·log(2))
  ```
  其中k为标签数。熵值越高说明分类器对该样本越不确定，越值得人工标注。

**`train_linear_probe()`**：从元数据筛选`split="labeled"`的样本作为训练集，关联oracle标签训练探针。

#### `src/automine_loop/classifier/mlp.py`
- 占位入口函数，当前直接调用`train_linear_probe`。
- 保留MLP的模块边界，方便未来替换为基于PyTorch的多层感知机。

---

### 3.9 主动学习采样层（`active_learning/`）
**核心算法模块**，决定每轮应选择哪些样本送标注。

#### `src/automine_loop/active_learning/sampler.py` — 主动采样引擎
**`select_samples()`函数**：
支持5种采样策略：

| 策略 | 算法 | 适用场景 |
|------|------|----------|
| `random` | 从未标注池随机采样 | 基线对照 |
| `uncertainty` | 选分类器熵值最高的样本 | 最大化单轮信息增益 |
| `coreset` | 选与已标注集合距离最远的样本（diversity=min-dist-to-labeled） | 覆盖数据分布 |
| `rule` | 选稀有度最高的样本（基于聚类簇大小） | 发现长尾场景 |
| **`combined`（默认）** | **加权融合4维得分**（见下方公式） | 兼顾不确定性+多样性+稀有度+去冗余 |

**组合策略的核心公式**（`combined`模式）：
```
score = 0.35 × uncertainty + 0.25 × diversity + 0.20 × rarity - 0.15 × redundancy
```
所有分量均经过**Min-Max归一化**至[0,1]区间：

- **uncertainty（不确定性，权重0.35）**：分类器的平均熵（来自`predict_probabilities`），越大越不确定
- **diversity（多样性，权重0.25）**：样本到已标注集合的**最小欧氏距离**（来自`pairwise_distances`），距离越大越具多样性
- **rarity（稀有度，权重0.20）**：`1/√(cluster_size)`，小簇样本得分高
- **redundancy（冗余惩罚，权重0.15）**：`1 - diversity`，抑制与已标注样本过于相似的候选

**`component_scores()`函数**：计算上述4个分量得分，所有分量均做Min-Max归一化。

---

### 3.10 评估层（`evaluation/`）

#### `src/automine_loop/evaluation/metrics.py` — 评估指标
- `multilabel_metrics()`：计算多标签分类的**macro-F1**、**macro-Precision**、**macro-Recall**（阈值0.5）。
- `evaluate_predictions()`：在验证/测试集上评估预测结果与oracle标签的一致性。
- `precision_at_k()`：计算检索结果Top-K的标签一致性（衡量"相似帧是否确实共享同一标签"）。

---

### 3.11 流水线编排层（`pipeline/`）

#### `src/automine_loop/pipeline/run_demo.py` — 完整演示流水线
将一个项目的完整数据闭环串联起来：

```
生成演示数据 → 计算质量特征 → 提取嵌入 → 构建向量索引
                                             ↓
                                        聚类分析
                                             ↓
                          ┌── 主动学习迭代(3轮) ──┐
                          │  1. 训练线性探针       │
                          │  2. 预测概率+不确定性  │
                          │  3. 评估预测质量       │
                          │  4. 组合策略采样       │ 
                          │  5. 释放标注←扩充已标注 │
                          └────────────────────────┘
                                             ↓
                              输出指标报告+采样记录
```

每轮迭代后，`labeled_ids`集合扩展，下一轮在更大的标注池上训练更准确的分类器，形成**正向反馈循环**。

---

### 3.12 可视化层（`app/`）

#### `app/streamlit_app.py` — Streamlit交互界面
提供4个视图：
1. **Overview**：帧数量、场景数、聚类数、已选样本数统计，split分布柱状图，数据表预览。
2. **Image Search**：选择查询帧ID，Top-K相似图像检索并展示缩略图+相似度得分。
3. **Clusters**：按簇浏览聚类结果，展示每个簇中距离中心最近的帧。
4. **Active Sampling**：按轮次查看主动学习采样的候选帧和得分。

---

### 3.13 CLI命令行接口（`cli.py`）

- **`automine demo`**：运行完整演示流水线。
- **`automine search <record_id>`**：按帧ID搜索Top-K相似帧（从已保存的向量索引加载）。

---

### 3.14 测试层（`tests/`）

#### `tests/test_smoke.py`
端到端冒烟测试，验证`run_demo`在临时目录中完整运行并产出索引文件。

---

## 4. 算法核心思想总结

### 4.1 数据闭环全景
```
                 ┌──────────┐
                 │ 路采原始数据 │
                 └────┬─────┘
                      ▼
            ┌─────────────────┐
            │ 构建帧清单+隐藏标签 │  (dataset/)
            └────────┬────────┘
                     ▼
            ┌─────────────────┐
            │  质量特征+规则筛选  │  (quality/)
            └────────┬────────┘
                     ▼
            ┌─────────────────┐
            │  语义嵌入提取     │  (embedding/)
            └────────┬────────┘
                     ▼
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   ┌─────────┐ ┌─────────┐ ┌──────────┐
   │ 向量检索  │ │ 聚类去重  │ │ 场景分类器 │
   │(retrieval)│ │(clustering)│ │(classifier)│
   └─────────┘ └────┬─────┘ └────┬─────┘
                    │            │
                    └─────┬──────┘
                          ▼
                 ┌────────────────┐
                 │  主动学习采样    │  (active_learning/)
                 │  组合策略筛选    │
                 └───────┬────────┘
                         ▼
                 ┌────────────────┐
                 │  释放标注/评估   │  (evaluation/)
                 │  扩充已标注池    │
                 └───────┬────────┘
                         │
                         ▼ (迭代)
                  下一轮标注选择
```

### 4.2 关键设计原则

1. **按场景切分（scene-level split）**：同一场景的所有帧分配到同一数据子集，防止图像级随机切分导致的时序信息泄漏。
2. **Oracle标签隔离**：真实标签仅用于模拟标注释放和评估，训练过程中严格不可见，真实模拟无标签挖掘场景。
3. **渐进式依赖**：MVP可以在无GPU、无深度学习框架、无FAISS的环境下完整运行（使用颜色直方图+NumPy内积索引）；真实场景可无缝替换为DINOv2/CLIP+FAISS。
4. **组合采样策略**：同时考虑不确定性（模型弱点）、多样性（数据覆盖）、稀有度（长尾发现）和冗余惩罚（去重），比单一策略更高效。
5. **闭环迭代**：每轮标注释放后扩充已标注池，分类器在更大数据上重新训练，采样质量逐步提升。

---

## 5. 快速开始

```powershell
cd C:\path\to\automine-loop\data_select
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python -m automine_loop.cli demo --root .
python -m automine_loop.cli search 0 --root . --top-k 10
streamlit run app\streamlit_app.py
```

## 6. 后续扩展方向

- 替换基础编码器为真实的 DINOv2 / CLIP 语义编码器
- 添加基于检测器的ROI级裁剪
- 数据集规模扩大后引入FAISS向量索引
- 在nuScenes mini/trainval上运行5轮×3种子的正式实验
- 使用PySpark进行大规模元数据聚合
