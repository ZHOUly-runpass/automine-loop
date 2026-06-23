# 第一阶段总结：nuScenes mini 真实数据闭环已跑通

## 当前结论

第一阶段已经完成：项目不再只停留在 synthetic demo，而是已经在服务器 `zhou@10.20.31.213` 的个人目录中完成了 nuScenes mini 真实车载摄像头数据的无标签筛选闭环。

服务器工作目录：

```text
/home/zhou/automine-loop/data_select
```

nuScenes 数据目录：

```text
/home/zhou/datasets/nuscenes
```

Python 环境：

```text
/home/zhou/automine-loop/data_select/.venv
```

后续服务器命令建议继续使用：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B
```

原因是服务器 `.venv` 曾出现过损坏 `.pyc` 字节码缓存，使用 `-B` 可以避免再次读写缓存。

## 已实现内容

### 1. 服务器私有部署环境

已在 `zhou` 用户空间内完成部署，不依赖 root 运行训练，不向其他用户开放目录。

已完成：

- `/home/zhou/automine-loop/data_select` 项目代码部署。
- `/home/zhou/datasets/nuscenes` 数据目录创建。
- `.venv` Python 3.10 虚拟环境创建。
- 离线安装基础依赖、Streamlit、pytest、nuScenes devkit、OpenCV。
- 服务器端测试通过。

验证结果：

```text
2 passed
```

### 2. nuScenes mini 数据安装

已上传并解压：

```text
v1.0-mini.tgz
nuScenes-map-expansion-v1.3.zip
```

服务器上已具备：

```text
/home/zhou/datasets/nuscenes/v1.0-mini
/home/zhou/datasets/nuscenes/samples
/home/zhou/datasets/nuscenes/sweeps
/home/zhou/datasets/nuscenes/maps
```

nuScenes devkit 加载验证结果：

```text
samples 404
sample_data 31206
```

### 3. 构造无标签样本池

已将 nuScenes mini 的 `CAM_FRONT` 图像转换成项目内部 manifest 和 oracle labels。

执行命令：

```bash
python -B -m automine_loop.cli prepare-nuscenes \
  --root . \
  --data-root ~/datasets/nuscenes \
  --version v1.0-mini \
  --camera CAM_FRONT
```

生成结果：

```text
data/manifests/data_manifest.parquet
data/labels/oracle_labels.parquet
```

说明：

- `data_manifest.parquet` 是无标签样本池索引，包含真实图片路径、scene、split 等信息。
- `oracle_labels.parquet` 来自公开标注转换，只用于评估和模拟人工标注释放。
- pipeline 中未释放标签不会直接作为训练标签使用。

### 4. 真实数据 pipeline

已完成 5 轮主动学习闭环：

```bash
python -B -m automine_loop.cli pipeline \
  --root . \
  --rounds 5 \
  --budget 32 \
  --encoder color_histogram
```

生成结果：

```text
data/manifests/quality_features.parquet
data/embeddings/frame_embeddings.npy
data/embeddings/embedding_metadata.parquet
data/indexes/frame_index.npz
data/indexes/id_mapping.parquet
data/manifests/cluster_result.parquet
data/manifests/selected_samples.parquet
reports/experiment_metrics.parquet
reports/demo_report.md
```

报告结果：

```text
Dataset source: nuScenes.

round 0 macro_f1 0.619032
round 1 macro_f1 0.726855
round 2 macro_f1 0.784416
round 3 macro_f1 0.789676
round 4 macro_f1 0.796210
```

### 5. 检索和可视化验证

CLI 检索已验证可用：

```bash
python -B -m automine_loop.cli search 0 --root . --top-k 10
```

返回结果已经是 nuScenes 真实车载图像路径，例如：

```text
/home/zhou/datasets/nuscenes/samples/CAM_FRONT/...
```

Streamlit 已能显示真实车载画面：

```bash
python -B -m streamlit run app/streamlit_app.py \
  --server.address 127.0.0.1 \
  --server.port 8502 \
  --server.headless true
```

本地通过 SSH 隧道访问：

```powershell
ssh -L 8502:127.0.0.1:8502 zhou@10.20.31.213
```

浏览器：

```text
http://127.0.0.1:8502
```

## 对应脚本和模块

### CLI 入口

```text
data_select/src/automine_loop/cli.py
```

实现命令：

- `prepare-nuscenes`
- `quality`
- `embed`
- `cluster`
- `train`
- `pipeline`
- `search`

### nuScenes 适配

```text
data_select/src/automine_loop/dataset/nuscenes_adapter.py
```

作用：

- 读取 nuScenes devkit 数据。
- 选择指定相机，例如 `CAM_FRONT`。
- 生成真实图片 manifest。
- 从公开标注构造 oracle labels。

### pipeline 编排

```text
data_select/src/automine_loop/pipeline/stages.py
```

作用：

- `prepare_nuscenes`
- `run_quality`
- `run_embedding`
- `run_clustering`
- `run_training`
- `run_existing_pipeline`
- 生成报告 `reports/demo_report.md`。

### 图像质量特征

```text
data_select/src/automine_loop/quality/features.py
```

作用：

- 计算亮度、模糊度、图像 hash、损坏图像标记等质量特征。

### Embedding

```text
data_select/src/automine_loop/embedding/encoders.py
data_select/src/automine_loop/embedding/dinov2_encoder.py
data_select/src/automine_loop/embedding/clip_encoder.py
```

当前第一阶段实际使用：

```text
color_histogram
```

说明：

- `color_histogram` 已完成真实数据闭环验证。
- `dinov2` 和 `clip` 已保留接口，但需要后续安装深度学习依赖和模型权重后再验证。

### 检索索引

```text
data_select/src/automine_loop/retrieval/index.py
```

作用：

- 构建归一化 flat inner-product index。
- 支持按 `record_id` 做 Top-K 相似图像检索。

### 聚类

```text
data_select/src/automine_loop/clustering/cluster.py
```

作用：

- 基于 embedding 做 K-Means 聚类。
- 输出 cluster id、cluster size、distance_to_center。

### 主动采样

```text
data_select/src/automine_loop/active_learning/sampler.py
```

作用：

- 综合 uncertainty、coreset distance、rarity、rule score 选择下一轮样本。
- 输出 `selected_samples.parquet`。

### 轻量训练与评估

```text
data_select/src/automine_loop/classifier/linear_probe.py
data_select/src/automine_loop/evaluation/metrics.py
```

作用：

- 使用释放后的 labeled 样本训练轻量分类器。
- 输出每轮 macro F1、precision、recall。

### Streamlit 可视化

```text
data_select/app/streamlit_app.py
```

页面：

- Overview
- Image Search
- Clusters
- Active Sampling

说明：

- `Image Search` 的 Top-K 是 embedding 检索结果。
- `Clusters` 是无监督聚类结果。
- `Active Sampling` 和 `reports/experiment_metrics.parquet` 更接近训练/主动学习闭环结果。

### 批量实验脚本

```text
data_select/scripts/run_nuscenes_experiment.sh
```

作用：

- 一键执行 `prepare-nuscenes -> quality -> embed -> cluster -> train`。
- 支持通过环境变量设置 `DATA_ROOT`、`VERSION`、`CAMERA`、`ENCODER`、`ROUNDS`、`BUDGET`、`SEEDS`。

### 服务器训练脚本

```text
data_select/scripts/server_train.sh
```

作用：

- 提供服务器端训练入口。
- 支持 demo 和 nuScenes 模式。

### 结果同步脚本

```text
data_select/scripts/sync_results.ps1
```

作用：

- 从服务器拉回 `reports/`、`data/manifests/`、`data/embeddings/`、`data/indexes/` 等结果。
- 建议后续用它或 `scp` 同步结果，不把数据产物提交 Git。

## 第一阶段边界

已经完成：

- 真实 nuScenes mini 数据接入。
- 无标签样本池构造。
- 质量特征计算。
- color histogram embedding。
- Top-K 相似检索。
- K-Means 聚类。
- 主动采样。
- 模拟人工标注释放。
- 5 轮轻量训练与评估。
- Streamlit 真实车载图像可视化。

尚未完成：

- DINOv2/CLIP 真实深度 embedding 验证。
- GPU/CUDA/PyTorch 服务器环境部署。
- 更大规模 nuScenes trainval 数据实验。
- 长时间多 seed 实验结果汇总。
- 更正式的训练日志管理和自动同步。

## 后续建议

1. 固化服务器运行命令，优先使用 `scripts/run_nuscenes_experiment.sh` 管理多 seed 实验。
2. 增加结果同步流程，只同步报告和小型 parquet，不上传原始数据和 embedding 到 GitHub。
3. 第二阶段再安装 PyTorch/transformers，验证 `--encoder dinov2` 或 `--encoder clip`。
4. Streamlit 页面可以继续增强：显示数据源、图片绝对路径、当前 encoder、round/budget、最新报告摘要，避免混淆 demo 和真实数据。
