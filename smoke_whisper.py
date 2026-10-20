"""Optional real-model integration test. Downloads tiny and a public speech sample.

Run with the same dependencies as the notebook plus nbformat. Requires network.
All downloaded files and generated output go to /tmp, not the repository.
"""
import os
from pathlib import Path
import urllib.request
import zipfile

os.environ.setdefault('XDG_CACHE_HOME', '/tmp/lecture-whisper-cache')

import nbformat
import torch
from colab_transcribe import pipeline


def main():
    notebook = nbformat.read('YouTube_Whisper_Free_Colab.ipynb', as_version=4)
    nbformat.validate(notebook)
    print('Notebook schema: OK', flush=True)
    torch.set_num_threads(2)
    sample = Path('/tmp/lecture-whisper-jfk.flac')
    if not sample.exists():
        with urllib.request.urlopen(
                'https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac',
                timeout=60) as response:
            sample.write_bytes(response.read())
    folder = pipeline('', '/tmp/lecture-whisper-smoke', model_size='tiny',
                      language='en', chunk_seconds=10, local_file=sample)
    text = (folder / 'transcript.txt').read_text(encoding='utf-8').lower()
    assert 'country' in text and 'you' in text, text
    with zipfile.ZipFile(folder / 'transcript.zip') as archive:
        assert archive.testzip() is None
    print('Actual Whisper inference + chunk merge + ZIP: OK', flush=True)
    second = pipeline('', '/tmp/lecture-whisper-smoke', model_size='tiny',
                      language='en', chunk_seconds=10, local_file=sample)
    assert folder == second
    print('Actual pipeline resume: OK', flush=True)


if __name__ == '__main__':
    main()
