from adsl.early_stopping import EarlyStoppingMonitor


def test_early_stopping_waits_for_min_steps_before_plateau_stop():
    monitor = EarlyStoppingMonitor(
        enabled=True,
        min_steps=5_000,
        patience_evals=2,
        min_delta=0.01,
        smoothing_window=1,
    )

    first = monitor.update(step=1_000, eval_return=100.0)
    second = monitor.update(step=2_000, eval_return=100.5)

    assert first.should_stop is False
    assert second.should_stop is False
    assert second.stale_evaluations == 1

    third = monitor.update(step=5_000, eval_return=100.7)

    assert third.should_stop is True
    assert third.reason == "plateau"
    assert third.stale_evaluations == 2


def test_early_stopping_resets_patience_after_meaningful_improvement():
    monitor = EarlyStoppingMonitor(
        enabled=True,
        min_steps=1_000,
        patience_evals=2,
        min_delta=0.01,
        smoothing_window=1,
    )

    monitor.update(step=1_000, eval_return=100.0)
    stale = monitor.update(step=2_000, eval_return=100.5)
    improved = monitor.update(step=3_000, eval_return=102.0)

    assert stale.stale_evaluations == 1
    assert improved.should_stop is False
    assert improved.stale_evaluations == 0
    assert improved.best_smoothed_return == 102.0


def test_disabled_early_stopping_never_stops():
    monitor = EarlyStoppingMonitor(
        enabled=False,
        min_steps=1_000,
        patience_evals=1,
        min_delta=0.01,
        smoothing_window=1,
    )

    monitor.update(step=1_000, eval_return=100.0)
    decision = monitor.update(step=2_000, eval_return=100.0)

    assert decision.should_stop is False
    assert decision.reason == ""
