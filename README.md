# 商品期货回测框架

这是从 `D:\回测框架测试` 分支 `refactor/framework`（`3439dcc`）剥离出的交易框架。
原仓库没有改动。这里保留行情读取、时点协议、定手记账、换月成本、标准落盘，以及
把外部因子表转换为交易掩码所需的最小执行适配器。
因子公式、因子注册表、实验脚本和历史结果不在本目录。

调用方向是单向的：外部代码算出多空掩码或权重，再交给本框架执行。
框架不识别因子编号，也不生成交易信号。

## 目录

| 路径 | 作用 |
|---|---|
| `data/` | 读取 `configs/data.yaml`、`configs/universe.yaml`，加载 Mongo 行情和交易规格 |
| `research/protocol.py` | 信号滞后、研究标签收益、年化、样本内外切分 |
| `research/metrics.py` | 收益风险、IC、分位收益 |
| `research/runtime.py` | 读取 `configs/backtest.yaml` 与 `configs/costs.yaml` |
| `backtest/ledger.py` | 按名义金额定手、重叠持仓、滑点手续费、换月成本 |
| `backtest/records.py` | 写出 `run.json`、`daily.csv`、`trades.csv`、`positions.csv`、`summary.csv` |
| `backtest/reconcile.py` | 换月价差不计入盈亏的对照 |
| `backtest/masks.py` | 时序和截面因子值转多空掩码 |
| `factors/signals/liquidity.py` | 主力成交额流动性过滤 |
| `strategies/schedule.py`、`strategies/factor_file.py` | 日历调仓和周频因子文件适配 |
| `validation/` | 样本切分、试验登记和稳健性表 |

## 口径

- Parquet 行情目录由 `configs/data.yaml` 指定。需要兼容旧 MongoDB 读取代码时，通过
  `QA_MONGO_URI` 和 `QA_MONGO_DB` 环境变量提供连接信息，不把连接配置写入仓库。
- 品种池只从 `configs/universe.yaml` 读取。
- T 日信号由调用方用 `adj_close` 算完后传入。框架在 T+`signal_lag`（默认 1）用调用方给出的价格成交和盯市。
- 研究标签收益是 `adj_settle[T+1+H] / adj_settle[T+1] - 1`，由 `research.protocol.forward_return` 计算。
- 手数按价格乘以合约乘数的名义金额计算，不读取保证金。
- 样本外起点只来自 `configs/data.yaml` 的 `protocol.oos_start`。该日及之后为样本外。

## 调用

掩码是布尔表，索引为交易日，列为品种。`lag` 由框架处理，调用方不要再自行移位。

```python
from backtest.ledger import research_backtest
from backtest.records import write_research_run

result = research_backtest(
    long_mask,
    short_mask,
    price,
    specs,
    holding=5,
    lag=1,
    cost_bps=5.0,
    capital=100_000_000,
)
write_research_run("runs/example", metadata, result)
```

`price` 是成交和盯市用的价格。研究标签用后复权结算价，账户成交用原始结算价，两套价格不要混在同一次记账里。
`metadata` 须包含 `research.repro.RUN_CONTRACT_FIELDS` 里的数据契约字段。

外部因子接入时，调用顺序必须固定为：先计算日期 × 品种因子表，再用
`factors.signals.liquidity.research_main_liquidity_mask` 做主力流动性过滤，
然后用 `backtest.masks.long_short_masks` 生成多空掩码。日频调仓用
`strategies.schedule.gate_signal_days`；周频调仓用
`strategies.factor_file.apply_weekly_rebalance`，最后交给
`research_backtest` 或 `delta_rebalance_backtest`。因子表代表 T 日收盘信号，
不要在调用方自行移位，成交滞后由账本的 `lag` 处理。

## 安装与测试

依赖见 `requirements.txt`。可以用本目录新建虚拟环境，也可以直接用原项目的解释器：

```powershell
D:\回测框架测试\.venv\Scripts\python.exe -m pytest -q
```

在本目录下执行。测试不连接 MongoDB，行情读取用假数据。

## 跟随源仓库

本目录是独立仓库，提交号与 `D:\回测框架测试` 不同。源仓库有新提交后，在本目录执行：

```powershell
D:\回测框架测试\.venv\Scripts\python.exe sync_framework.py
```

脚本只读取源仓库已经提交的内容。同步范围是行情、协议、指标、记账、校验、对应配置，以及不引用因子或策略的测试。`backtest/run.py`、研究信号、因子、策略、脚本和结果目录保持在源仓库。已跟随到的源提交写在 `source-sync.json`。同步不会自动提交，确认差异后再在本仓库提交。
