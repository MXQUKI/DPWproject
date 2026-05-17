# IMDB Movies Dataset Project

中文说明在本文件中，English version is available at [README_EN.md](/Users/m/Desktop/DPW%20project/DPWproject/README_EN.md)。

## 项目简介

这是一个基于 Kaggle / IMDB 电影数据集的课程型数据分析项目，整合了：

- 数据清洗与质量校验
- 探索性数据分析（EDA）
- Matplotlib 图表导出
- 票房预测验证
- Tkinter 桌面交互界面

项目当前以“清洗后的统一数据集”为核心运行，命令行报告、UI、预测模块和图表模块都共享同一份清洗结果，便于结果一致、启动更快、维护更简单。

## 当前功能

### 1. 数据清洗

- 从原始 Kaggle 数据构建项目数据表
- 自动合并 `movies_metadata.csv`、`keywords.csv`、`credits.csv`
- 清洗并标准化 `release_date`
- 删除 `2017-07-31` 之后的记录
- 自动生成 `year`
- 删除缺失标题和非法日期记录
- 严格处理重复电影
- 重新编排清洗后 `id`
- 保留 `source_id` 用于追踪原始记录
- 统一补齐清洗后必需字段，保证清洗结果无空值

### 2. 探索性数据分析

- 统计电影数量、类型数量、平均评分、中位时长、总票房
- 按类型汇总电影数量、平均评分、平均票房
- 按年份汇总电影数量、平均评分、平均票房
- 生成 Top Movies 列表
- 自动生成文字洞察

### 3. 可视化

- `genre_distribution.png`
- `genre_comparison.png`
- `yearly_rating_trend.png`
- `budget_revenue_scatter.png`
- `forecast_backtest_yearly_comparison.png`

其中最后一张预测图是固定图，展示 `2003-2017` 各验证年份的“预测总票房 vs 实际总票房”。

### 4. 高级分析

- 电影产量趋势
- 预算与票房相关性分析
- 年代比较
- 票房预测验证

### 5. 图形界面

- 类型筛选
- 年份区间筛选
- 最低评分筛选
- 标题 / 关键词搜索
- 结果表格分页浏览
- 洞察与高级分析摘要
- 图表预览
- 数据质量报告
- UI 操作日志记录

## 项目结构

主要文件如下：

- [main.py](/Users/m/Desktop/DPW%20project/DPWproject/main.py)：统一入口，支持 `report` / `ui`
- [data_preprocessing.py](/Users/m/Desktop/DPW%20project/DPWproject/data_preprocessing.py)：数据清洗、导出、质量报告、缓存
- [data_analysis.py](/Users/m/Desktop/DPW%20project/DPWproject/data_analysis.py)：筛选、统计、洞察
- [data_visualization.py](/Users/m/Desktop/DPW%20project/DPWproject/data_visualization.py)：图表生成
- [advanced_analytics.py](/Users/m/Desktop/DPW%20project/DPWproject/advanced_analytics.py)：高级分析
- [box_office_forecasting.py](/Users/m/Desktop/DPW%20project/DPWproject/box_office_forecasting.py)：票房预测与验证
- [imdb_ui.py](/Users/m/Desktop/DPW%20project/DPWproject/imdb_ui.py)：Tkinter 图形界面
- [ui_log_report.py](/Users/m/Desktop/DPW%20project/DPWproject/ui_log_report.py)：UI 日志汇总
- [requirements.txt](/Users/m/Desktop/DPW%20project/DPWproject/requirements.txt)：依赖列表

常见目录：

- `raw_data/`：原始 Kaggle 数据
- `outputs/`：图表、预测结果、导出的项目数据
- `logs/`：UI 日志
- `.cache/`：Matplotlib / fontconfig 缓存

## 数据来源与运行策略

项目优先读取清洗后的数据：

1. 根目录下的 `cleaned_imdb_movies.csv`
2. 通过 `--dataset` 显式指定的清洗后 CSV

当前主流程和 UI 默认不再每次启动都重跑整套原始清洗，而是：

- 优先读取清洗后数据
- 在运行时做必要的再校验
- 通过进程内缓存复用结果

这样可以保证：

- 命令行、UI、预测图表基于同一份数据
- UI 响应更快
- 数据问题更容易排查

## 清洗后数据字段

当前清洗后的核心字段为：

- `id`
- `source_id`
- `title`
- `original_title`
- `primary_genre`
- `budget`
- `budget_observed`
- `revenue`
- `revenue_observed`
- `release_date`
- `year`
- `runtime`
- `runtime_observed`
- `vote_average`
- `vote_count`
- `popularity`
- `language`
- `original_language`
- `country`
- `keyword`

说明：

- `overview` 已不再保留到清洗后数据中
- 清洗后数据已做严格处理，不应出现空值
- `id` 是清洗后的顺序编号
- `source_id` 是原始数据中的电影标识

## 环境要求

- Python 3.9 或更高版本
- macOS / Linux / Windows
- 图形界面模式需要可用桌面显示环境

## 安装依赖

推荐使用项目虚拟环境：

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

主要依赖：

- `numpy`
- `pandas`
- `matplotlib`
- `Pillow`
- `kagglehub`

## 运行方式

### 1. 命令行报告

```bash
./.venv/bin/python main.py --mode report
```

作用：

- 读取清洗后数据
- 运行 EDA
- 运行高级分析
- 运行 2017 留出集票房预测验证
- 导出图表到 `outputs/`
- 打印结果摘要

指定数据文件：

```bash
./.venv/bin/python main.py --mode report --dataset cleaned_imdb_movies.csv --output-dir outputs
```

### 2. 图形界面

```bash
./.venv/bin/python main.py --mode ui
```

界面包含四个标签页：

- `Results`：结果表格与分页
- `Insights`：自动洞察与高级分析摘要
- `Charts`：图表预览
- `Data Quality`：清洗报告与缺失值统计

左侧筛选支持：

- 类型
- 起始年份
- 结束年份
- 最低评分
- 标题 / 关键词

说明：

- 普通图表可以基于当前筛选结果重建
- `Yearly Forecast vs Actual` 是固定图
- 这张固定预测图始终基于完整清洗数据展示
- 它不随筛选结果变化，也不需要刷新

### 3. 单独重建清洗文件

```bash
./.venv/bin/python data_preprocessing.py
```

会在根目录生成：

- `cleaned_imdb_movies.csv`

## 输出文件说明

默认输出目录为 `outputs/`。

常见输出包括：

- `cleaned_imdb_movies_project.csv`：主流程导出的项目数据
- `genre_distribution.png`：类型分布图
- `genre_comparison.png`：类型数量与平均评分对比图
- `yearly_rating_trend.png`：年度评分趋势图
- `budget_revenue_scatter.png`：预算与票房散点图
- `forecast_backtest_yearly_comparison.png`：`2003-2017` 年每年预测总票房与实际总票房对比图
- `forecast_2017_genre_summary.csv`：2017 验证集分类型预测指标
- `forecast_2017_movie_predictions.csv`：2017 验证集逐电影预测结果
- `forecast_2017_overview.json`：2017 验证集总体预测信息
- `chart_manifest.json`：图表批次元数据

## 时间范围说明

项目使用的数据截止到 `2017-07-31`。

这意味着：

- 清洗阶段会主动删除更晚日期的电影
- 所有分析都基于这一有效范围
- 2017 预测验证实际上对应 `2017-01-01` 到 `2017-07-31` 的可用电影

## 搜索与筛选行为

当前标题 / 关键词搜索已做收紧处理：

- 优先精确标题匹配
- 然后才考虑标题前缀和关键词匹配
- 避免把不相关系列电影或子串误匹配进来

这使得像 `Toy Story`、`Star Wars`、`It` 这类查询更符合直觉。

## 缓存机制

当前版本包含两层缓存：

- 数据缓存：首次读取 `cleaned_imdb_movies.csv` 后在进程内复用
- UI 分析缓存：重复筛选条件复用分析结果

目标是减少重复读盘和重复计算，提高 UI 响应速度。

## 数据质量报告包含什么

在 `Data Quality` 标签页中可以看到：

- 数据源文件名
- 原始载入行数
- 清洗后行数
- 删除的重复数量
- 删除的缺失标题数量
- 删除的非法日期数量
- 删除的超出截止日期记录数量
- 原始时间范围
- 清洗后时间范围
- 每列缺失值数量
- 数值列摘要

## UI 日志与调试

图形界面会把日志写到：

- `logs/ui_operation_log.jsonl`

查看最近一次摘要：

```bash
./.venv/bin/python ui_log_report.py
```

查看完整历史：

```bash
./.venv/bin/python ui_log_report.py --all
```

## 已验证的当前行为

基于当前仓库中的清洗数据，已验证：

- 清洗后数据共有 `45,293` 行
- 清洗后字段不含 `overview`
- 清洗后数据不应包含空值
- 清洗后 `id` 已重排
- 清洗后不存在重复 `title + year`
- `forecast_backtest_yearly_comparison.png` 为固定全局预测图
- 该图在 UI 中始终可直接显示

## 已知限制

- Tkinter UI 适合本地桌面，不适合纯无头服务器
- 预算和票房原始字段噪声较大，相关分析更适合作趋势参考
- 某些年份样本较少，年度统计可能波动较大
- `language` / `country` 已在分析层支持，但尚未暴露为 UI 控件

## 后续可扩展方向

- 在 UI 中加入语言和国家筛选
- 支持导出当前筛选结果
- 增加更多预测模型对比
- 增加更完整的自动化测试
- 补齐英文 README 与当前实现的一致性

## English Version

完整英文版请查看：

- [README_EN.md](/Users/m/Desktop/DPW%20project/DPWproject/README_EN.md)
