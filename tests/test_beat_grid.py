"""Pure-function checks: tempo detection without librosa or real audio."""
import unittest

import numpy as np

from vocalpitchlab.analysis.beat_grid import TARGET_RATE, estimate_bpm, onset_envelope
from vocalpitchlab.ui.plot_support import bar_phase, beat_period, grid_ticks


def click_track(bpm, seconds=20.):
    """Exponentially decaying noise bursts plus floor noise: the worst case for tempo."""
    rng = np.random.default_rng(0)
    total = int(seconds * TARGET_RATE)
    data = np.zeros(total, dtype=np.float32)
    length = int(.04 * TARGET_RATE)
    click = (rng.standard_normal(length) * np.exp(-np.arange(length) / (.010 * TARGET_RATE))).astype(np.float32)
    position = 0.
    while position < seconds:
        start = int(position * TARGET_RATE)
        end = min(total, start + length)
        if end > start:
            data[start:end] += click[:end - start]
        position += 60. / bpm
    data += rng.standard_normal(total).astype(np.float32) * .002
    return data


class TempoTests(unittest.TestCase):
    def test_clicks_recover_tempo(self):
        for bpm in (72, 90, 120, 138):
            with self.subTest(bpm=bpm):
                envelope, fps = onset_envelope(click_track(bpm), TARGET_RATE)
                self.assertAlmostEqual(estimate_bpm(envelope, fps), bpm, delta=2.)

    def test_silence_has_no_tempo(self):
        envelope, fps = onset_envelope(np.zeros(TARGET_RATE * 5, np.float32), TARGET_RATE)
        self.assertIsNone(estimate_bpm(envelope, fps))

    def test_short_input_has_no_tempo(self):
        envelope, fps = onset_envelope(np.zeros(100, np.float32), TARGET_RATE)
        self.assertIsNone(estimate_bpm(envelope, fps))


class PhaseTests(unittest.TestCase):
    def test_phase_recovers_downbeat(self):
        accent = np.tile([1., .2, .6, .2], 8)
        self.assertEqual(bar_phase(accent, 4), 0)
        self.assertEqual(bar_phase(np.r_[accent[2:], accent[:2]], 4), 2)

    def test_near_tie_prefers_earliest(self):
        # Kick on 1 and 3 make a near tie; the earlier phase must win.
        self.assertEqual(bar_phase(np.tile([1., .1, .99, .1], 8), 4), 0)

    def test_missing_accent_is_zero(self):
        self.assertEqual(bar_phase(None, 4), 0)
        self.assertEqual(bar_phase([], 4), 0)


class GridTests(unittest.TestCase):
    def test_bars_only_when_beats_are_dense(self):
        beats = np.arange(0., 40., .5)
        times, is_bar = grid_ticks(0., 40., 40., beats, .5, 0, 4, 300.)
        self.assertTrue(is_bar.all())
        self.assertEqual(len(times), 20)

    def test_grid_disappears_when_bars_are_dense(self):
        beats = np.arange(0., 300., .5)
        times, _ = grid_ticks(0., 300., 300., beats, .5, 0, 4, 300.)
        self.assertEqual(len(times), 0)

    def test_uniform_grid_without_beats(self):
        times, is_bar = grid_ticks(0., 10., 10., None, .5, 0, 4, 1000.)
        self.assertGreater(len(times), 0)
        self.assertEqual(len(is_bar), len(times))


class ReleaseTests(unittest.TestCase):
    def test_only_unreferenced_folders_are_removed(self):
        from unittest import mock
        import tempfile
        from pathlib import Path as _Path
        from vocalpitchlab.analysis import analysis_cache

        with tempfile.TemporaryDirectory() as tmp:
            root = _Path(tmp) / 'results' / 'stages'
            kept = root / 'decode' / 'keep'
            dropped = root / 'decode' / 'drop'
            kept.mkdir(parents=True)
            dropped.mkdir(parents=True)
            (kept / 'f.txt').write_text('k', encoding='utf-8')
            (dropped / 'f.txt').write_text('d', encoding='utf-8')
            with mock.patch.object(analysis_cache, 'DATA_ROOT', _Path(tmp)):
                analysis_cache.release([('decode', 'keep'), ('decode', 'drop')], {'keep'})
            self.assertTrue(kept.is_dir())
            self.assertFalse(dropped.is_dir())


class PeriodTests(unittest.TestCase):
    def test_real_beats_win(self):
        self.assertAlmostEqual(beat_period([0., .48, .96], 100.), .48)

    def test_falls_back_to_tempo(self):
        self.assertAlmostEqual(beat_period(None, 120.), .5)

    def test_unknown_tempo(self):
        self.assertEqual(beat_period(None, None), 0.)


if __name__ == '__main__':
    unittest.main()
