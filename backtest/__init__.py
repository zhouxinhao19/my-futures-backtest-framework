"""研究回测框架。只做定手、重叠持仓和逐日逐笔记账。"""

from .ledger import (
    CAPITAL,
    COST_BPS,
    LEG_NOTIONAL,
    continuous_backtest,
    research_backtest,
    signal_lots,
    tradable_specs,
)

__all__ = [
    "CAPITAL",
    "COST_BPS",
    "LEG_NOTIONAL",
    "continuous_backtest",
    "research_backtest",
    "signal_lots",
    "tradable_specs",
]
