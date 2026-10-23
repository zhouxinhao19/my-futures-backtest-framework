import pandas as pd

from data.trading_calendar import last_session_of_week, next_session, sessions


def test_calendar_uses_actual_bars():
    dates = pd.to_datetime(["2024-02-01", "2024-02-02", "2024-02-05", "2024-02-07"])
    frame = pd.DataFrame({"date": dates})
    assert sessions(frame).tolist() == list(dates)
    assert last_session_of_week(frame).tolist() == [
        pd.Timestamp("2024-02-02"), pd.Timestamp("2024-02-07")
    ]
    assert next_session(sessions(frame), pd.Timestamp("2024-02-02")) == pd.Timestamp("2024-02-05")
