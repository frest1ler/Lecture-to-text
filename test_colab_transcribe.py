import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import colab_transcribe as app


class FakeModel:
    device = 'cpu'

    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def transcribe(self, audio, **options):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('simulated interruption')
        return {'segments': [{'start': 0, 'end': 1, 'text': ' Пример.',
                              'words': [{'start': 0, 'end': 1, 'word': ' Пример.'}]}]}


class PipelineTests(unittest.TestCase):
    def test_links(self):
        for link in ['https://youtu.be/kgccc0I_kkw?t=5',
                     'https://www.youtube.com/watch?v=kgccc0I_kkw&t=2s',
                     'https://youtube.com/live/kgccc0I_kkw']:
            self.assertEqual(app.youtube_url(link)[1], 'kgccc0I_kkw')
        for link in ['https://youtube.com.evil.org/watch?v=kgccc0I_kkw',
                     'rm -rf /', 'https://youtube.com/playlist?list=abc']:
            with self.assertRaises(ValueError):
                app.youtube_url(link)

    def test_chunk_coverage_and_edges(self):
        chunks = app.make_chunks(1200.5)
        self.assertEqual([(c.start, c.end) for c in chunks],
                         [(0, 600), (600, 1200), (1200, 1200.5)])
        self.assertEqual(chunks[1].audio_start, 595)
        self.assertEqual(chunks[-1].audio_end, 1200.5)
        self.assertEqual(len(app.make_chunks(1200)), 2)

    def test_overlap_ownership_and_absolute_timestamps(self):
        chunks = app.make_chunks(1200)
        absolute = [(598, 599, ' до'), (599.8, 600.2, ' граница'), (601, 602, ' после')]
        results = []
        for c in chunks:
            result = {'segments': [{'words': [dict(start=s-c.audio_start,
                       end=e-c.audio_start, word=t) for s, e, t in absolute]}]}
            results.extend(app.owned_segments(result, c))
        text = ' '.join(s['text'] for s in results)
        self.assertEqual(text, 'до граница после')
        self.assertEqual(results[1]['start'], 600)
        self.assertEqual(results[1]['end'], 602)

    def test_srt_rounding(self):
        self.assertEqual(app.timestamp(59.9996, True), '00:01:00,000')
        self.assertEqual(app.timestamp(3661.123, True), '01:01:01,123')

    def test_export_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = app.export_files(tmp, {'title': 'Лекция'},
                [{'start': 1.25, 'end': 2.5, 'text': 'Теорема <script>.'}])
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(len(z.namelist()), 5)
                self.assertIn('00:00:01,250 --> 00:00:02,500',
                              z.read('transcript.srt').decode())
                self.assertIn('Теорема', z.read('transcript.txt').decode())

    def test_real_ffmpeg_and_resume_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / 'input.wav'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                            'sine=frequency=440:duration=3.2', str(audio)], check=True)
            self.assertAlmostEqual(app.duration(audio), 3.2, places=1)
            chunks = app.make_chunks(app.duration(audio), 2, 0)
            out = root / 'result'
            failing = FakeModel(fail_at=2)
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                app.transcribe_chunks(audio, out, chunks, {}, lambda: failing)
            self.assertTrue((out / 'chunk-0000.json').exists())
            self.assertFalse((out / 'chunk-0001.json').exists())
            self.assertIn('Пример', (out / 'partial.txt').read_text())
            resumed = FakeModel()
            result = app.transcribe_chunks(audio, out, chunks, {}, lambda: resumed)
            self.assertEqual(resumed.calls, 1)
            self.assertEqual([s['start'] for s in result], [0, 2])
            # Fully cached run never loads model.
            app.transcribe_chunks(audio, out, chunks, {},
                                  lambda: self.fail('should not load model'))
            (out / 'chunk-0001.json').write_text('{broken')
            repaired = FakeModel()
            app.transcribe_chunks(audio, out, chunks, {}, lambda: repaired)
            self.assertEqual(repaired.calls, 1)

    def test_download_failure_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(app, 'command', side_effect=RuntimeError('HTTP 403')):
                with self.assertRaisesRegex(RuntimeError, 'UPLOAD_LOCAL_FILE'):
                    app.download_audio('https://youtube.com/watch?v=kgccc0I_kkw', tmp)

    def test_notebook_embeds_engine_and_compiles(self):
        notebook = json.loads(Path('YouTube_Whisper_Free_Colab.ipynb').read_text())
        code = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        for i, source in enumerate(code):
            compile(source, f'cell-{i}', 'exec')
        self.assertIn(Path('colab_transcribe.py').read_text(), code[3])
        self.assertTrue(all(not c.get('outputs') for c in notebook['cells']))
        self.assertNotIn('OPENAI_API_KEY', '\n'.join(code))

    def test_notebook_result_cell_shows_safe_text_and_download(self):
        notebook = json.loads(Path('YouTube_Whisper_Free_Colab.ipynb').read_text())
        source = ''.join(notebook['cells'][6]['source'])
        shown, downloads = [], []
        modules = {'IPython.display': types.SimpleNamespace(HTML=lambda x: x,
                                                            display=shown.append),
                   'google.colab': types.SimpleNamespace(
                       files=types.SimpleNamespace(download=downloads.append))}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            folder = Path(tmp)
            app.export_files(folder, {}, [{'start': 0, 'end': 1,
                             'text': '</textarea><script>bad()</script>'}])
            exec(compile(source, 'result-cell', 'exec'), {'RESULT_DIR': folder})
            self.assertIn('&lt;script&gt;', ''.join(shown))
            self.assertNotIn('<script>bad()', ''.join(shown))
            self.assertEqual(downloads, [str(folder / 'transcript.zip')])


if __name__ == '__main__':
    unittest.main()
