"""rl/timing.py: per-stage timers that accumulate across resumed sessions, and the whole-run totals."""
import time

import pytest

from rl.timing import StageTimer, pipeline_timing, read_timing, timing_rows


def test_sections_accumulate_and_flatten(tmp_path):
    timer = StageTimer(tmp_path)
    with timer.section('load_data'):
        time.sleep(0.02)
    timer.add('train', 6.0)
    timer.add('train', 6.0)
    assert timer.sections['train'] == pytest.approx(0.2)            # 12 s
    assert timer.sections['load_data'] > 0
    assert timer.total_minutes == pytest.approx(sum(timer.sections.values()))
    flat = timer.flat()
    assert set(flat) == {'time_load_data_min', 'time_train_min', 'time_total_min', 'time_sessions'}
    saved = read_timing(tmp_path)                                    # written after every section
    assert saved['sections'] == timer.sections and saved['finished_at'] is None


def test_resume_adds_to_previous_session(tmp_path):
    first = StageTimer(tmp_path)
    first.add('ppo_rollout', 60.0)
    started = first.started_at
    second = StageTimer(tmp_path)                                     # simulated resume
    second.add('ppo_rollout', 30.0)
    second.add('final_eval', 6.0)
    second.finish()
    saved = read_timing(tmp_path)
    assert saved['sessions'] == 2 and saved['started_at'] == started and saved['finished_at']
    assert saved['sections'] == {'ppo_rollout': pytest.approx(1.5), 'final_eval': pytest.approx(0.1)}
    assert saved['total_minutes'] == pytest.approx(1.6)


def test_pipeline_totals_and_rows(tmp_path):
    dirs = {s: tmp_path / s for s in ('predictor', 'agent', 'final')}
    for stage, minutes in (('predictor', 2.0), ('agent', 3.0)):
        t = StageTimer(dirs[stage])
        t.add('work', minutes * 60)
        t.finish()
    totals = pipeline_timing(dirs)                                    # final stage not run yet
    assert totals['predictor_minutes'] == pytest.approx(2.0) and totals['final_minutes'] is None
    assert totals['pipeline_minutes_total'] == pytest.approx(5.0) and totals['pipeline_started_at']
    rows = timing_rows(dirs, fc='FC21', test='GF')
    assert [(r['stage'], r['component']) for r in rows] == [
        ('predictor', 'work'), ('predictor', 'total'), ('agent', 'work'), ('agent', 'total'), ('pipeline', 'total')]
    assert rows[-1]['minutes'] == pytest.approx(5.0) and rows[0]['fc'] == 'FC21'
