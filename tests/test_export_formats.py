"""Check the MIDI and OpenUtau writers against the bytes they promise."""
import struct
import unittest

import numpy as np

from vocalpitchlab.analysis.export_formats import PPQ, to_midi, to_ustx


def _vlq(data, index):
    value = 0
    while True:
        byte = data[index]
        index += 1
        value = (value << 7) | (byte & 0x7f)
        if not byte & 0x80:
            return value, index


def read_events(blob):
    """Minimal Standard MIDI File reader: (tick, kind, a, b) with absolute ticks."""
    assert blob[:4] == b'MThd'
    self_length = struct.unpack('>I', blob[4:8])[0]
    assert self_length == 6
    fmt, count, division = struct.unpack('>HHH', blob[8:14])
    assert (fmt, count, division) == (0, 1, PPQ)
    assert blob[14:18] == b'MTrk'
    size = struct.unpack('>I', blob[18:22])[0]
    data = blob[22:22 + size]
    assert len(data) == size
    index, tick, events = 0, 0, []
    while index < len(data):
        delta, index = _vlq(data, index)
        tick += delta
        status = data[index]
        index += 1
        if status == 0xff:
            kind = data[index]
            index += 1
            length, index = _vlq(data, index)
            events.append((tick, 'meta', kind, data[index:index + length]))
            index += length
        elif status & 0xf0 == 0xc0:
            events.append((tick, 'program', data[index], None))
            index += 1
        else:
            kind = {0x80: 'note_off', 0x90: 'note_on', 0xb0: 'control', 0xe0: 'bend'}[status & 0xf0]
            events.append((tick, kind, data[index], data[index + 1]))
            index += 2
    return events


def bends(events):
    return [(a | (b << 7)) - 8192 for _, kind, a, b in events if kind == 'bend']


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.notes = [dict(start=0., end=.5, midi=60), dict(start=.6, end=1.2, midi=62)]
        times = np.arange(0., 1.3, .01)
        tone = np.where(times < .55, 60., 62.)
        values = tone + .3 * np.sin(2 * np.pi * 6 * times)
        values[::97] = np.nan
        self.curve = (times, values)

    def test_midi_writes_tempo_program_and_a_usable_bend_range(self):
        events = read_events(to_midi(self.notes, self.curve, bpm=120.))

        tempo = next(payload for _, kind, meta, payload in events if kind == 'meta' and meta == 0x51)
        self.assertEqual(int.from_bytes(tempo, 'big'), 500000)
        self.assertEqual(next(a for _, kind, a, _ in events if kind == 'program'), 54)
        # The bend range must be declared in semitones, not cents.
        controls = [(a, b) for _, kind, a, b in events if kind == 'control']
        self.assertEqual(controls, [(101, 0), (100, 0), (6, 2), (38, 0)])

    def test_midi_keeps_notes_whole_and_adds_bends_inside_them(self):
        events = read_events(to_midi(self.notes, self.curve, bpm=120.))
        starts = [tick for tick, kind, _, _ in events if kind == 'note_on']
        ends = [tick for tick, kind, _, _ in events if kind == 'note_off']
        self.assertEqual(starts, [0, 576])
        self.assertEqual(ends, [480, 1152])
        self.assertEqual([a for _, kind, a, _ in events if kind == 'note_on'], [60, 62])
        values = bends(events)
        self.assertGreater(len(values), 100)
        self.assertTrue(all(-8192 <= value <= 8191 for value in values))
        # 0.3 semitone of vibrato over a two-semitone range.
        self.assertTrue(800 < max(abs(value) for value in values) < 1600)

    def test_midi_survives_an_empty_curve_and_out_of_range_notes(self):
        events = read_events(to_midi([dict(start=0., end=.5, midi=140)], (np.array([]), np.array([])), bpm=90.))
        self.assertEqual(bends(events), [])
        self.assertFalse([1 for _, kind, _, _ in events if kind == 'note_on'])

    def test_ustx_satisfies_the_openutau_loader(self):
        text = to_ustx(self.notes, self.curve, bpm=120., name='示例 song')
        lines = text.splitlines()
        self.assertEqual(lines[0], 'ustx_version: "0.10"')
        self.assertIn('name: "示例 song"', text)
        for required in ('voice_parts:', '  track_no: 0', '- position: 0', '  bpm: 120',
                         '  - abbr: pitd', '  masked_curves: []', '  wave_parts: []'):
            self.assertIn(required, text)
        self.assertNotIn(' parts:', text)
        self.assertEqual(text.count('  - position: '), 2)
        self.assertEqual(text.count('    lyric: "a"'), 2)
        self.assertEqual(text.count('      data:'), 2)
        self.assertTrue(text.endswith('\n'))

    def test_ustx_pitch_curve_keeps_the_frame_grid(self):
        text = to_ustx(self.notes, self.curve, bpm=120.)
        lines = text.splitlines()
        xs = eval(next(line for line in lines if line.strip().startswith('xs:')).split('xs:')[1])
        ys = eval(next(line for line in lines if line.strip().startswith('ys:')).split('ys:')[1])
        self.assertEqual(len(xs), len(ys))
        self.assertGreater(len(xs), 20)
        self.assertEqual(xs, sorted(xs))
        self.assertTrue(all(x % 5 == 0 for x in xs))
        self.assertTrue(all(-1200 <= y <= 1200 for y in ys))

    def test_ustx_without_a_curve_still_opens(self):
        text = to_ustx(self.notes, (np.array([]), np.array([])), bpm=100.)
        self.assertIn('  curves: []', text)
        self.assertEqual(text.count('  - position: '), 2)
        self.assertIn('  bpm: 100', text)


if __name__ == '__main__':
    unittest.main()
