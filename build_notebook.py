"""Build a portable notebook containing the engine; no repo imports in Colab."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def cell(kind, source, **metadata):
    c = {'cell_type': kind, 'metadata': metadata,
         'source': source.splitlines(keepends=True)}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c


intro = '''# YouTube → расшифровка в бесплатном Google Colab

**Без API-ключей и платных запросов.** Вся обработка идёт на машине Colab.

1. Откройте этот `.ipynb` в [Google Colab](https://colab.research.google.com/) через **Файл → Загрузить блокнот**.
2. В меню **Среда выполнения → Сменить среду выполнения** выберите **GPU**, если она доступна.
3. Вставьте ссылку в `VIDEO_URL` ниже. Выберите **Среда выполнения → Выполнить все**.
4. Дождитесь результата в последней ячейке: текст можно скопировать, а архив скачать.

Обрабатывается **всё видео с начала**, даже если ссылка содержит `t=…`. По умолчанию язык — русский.
Модель `auto`: `medium` на GPU, `small` на CPU. Это распознавание речи, а не анализ слайдов.

Бесплатная GPU и длительность сеанса [не гарантированы](https://research.google.com/colaboratory/faq.html#resource-limits).
Не закрывайте сеанс до окончания. Готовые части сохраняются; повторный запуск продолжит их,
пока файлы среды существуют. Для сохранения между сеансами включите `SAVE_TO_DRIVE`
**до запуска**: Google один раз попросит разрешить доступ к Drive. Без этого при удалении
машины Colab прогресс теряется. На Drive также сохраняется аудио и требуется свободное место.

YouTube иногда блокирует IP Colab. Автоматически обойти это гарантированно нельзя.
В таком случае включите `UPLOAD_LOCAL_FILE`, снова выполните все ячейки и выберите
аудио/видео на компьютере. Обычно после запуска не нужны никакие действия до результата.
'''

settings = '''#@title 1. Настройки — вставьте ссылку и выполните все ячейки
VIDEO_URL = "https://www.youtube.com/watch?v=kgccc0I_kkw" #@param {type:"string"}
LANGUAGE = "ru" #@param {type:"string"}
MODEL_SIZE = "auto" #@param ["auto", "small", "medium", "large-v3", "turbo"]
SAVE_TO_DRIVE = False #@param {type:"boolean"}
UPLOAD_LOCAL_FILE = False #@param {type:"boolean"}
# Пустой LANGUAGE — определение языка автоматически для каждого фрагмента.
'''

setup = '''#@title 2. Установка зависимостей и проверка среды
import os
import shutil
import subprocess
import sys
from pathlib import Path

def install(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *args], check=True)

# Nightly-загрузчик содержит исправления при изменениях YouTube.
install("--upgrade", "--pre", "yt-dlp[default]")
install("openai-whisper==20250625", "deno")
if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=True)
if not shutil.which("deno"):
    raise RuntimeError("Не найден JavaScript runtime deno. Повторите установку зависимостей.")

import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "нет; используем CPU, это медленнее")
print("Установка завершена. Платный API не используется.")
'''

storage = '''#@title 3. Каталог результатов (и необязательная загрузка своего файла)
from pathlib import Path
LOCAL_FILE = None
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_ROOT = Path("/content/drive/MyDrive/lecture_transcripts")
else:
    OUTPUT_ROOT = Path("/content/lecture_transcripts")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
if UPLOAD_LOCAL_FILE:
    from google.colab import files
    print("Выберите один аудио- или видеофайл.")
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("Нужно загрузить ровно один файл")
    LOCAL_FILE = str(Path(next(iter(uploaded))).resolve())
    del uploaded  # не держать аудио ещё и в памяти Python
print("Каталог:", OUTPUT_ROOT)
'''

execute = '''#@title 5. Скачать и распознать — дождитесь «Готово»
# Сброс исключает показ результата предыдущего видео после ошибки нового запуска.
RESULT_DIR = None
RESULT_DIR = pipeline(VIDEO_URL, OUTPUT_ROOT, model_size=MODEL_SIZE,
                      language=LANGUAGE, local_file=LOCAL_FILE)
'''

display = '''#@title 6. Результат: скопируйте текст или скачайте архив
import html
from IPython.display import HTML, display
if RESULT_DIR is None:
    raise RuntimeError("Распознавание ещё не завершено. Смотрите ошибку в предыдущей ячейке.")
transcript = (RESULT_DIR / "transcript.txt").read_text(encoding="utf-8")
timed = (RESULT_DIR / "transcript_timestamps.txt").read_text(encoding="utf-8")
prompt = (RESULT_DIR / "prompt_for_chatgpt.txt").read_text(encoding="utf-8")
display(HTML("<h3>Расшифровка готова</h3><p>Выделите текст внутри поля (Ctrl+A), "
             "скопируйте (Ctrl+C) и отправьте в ChatGPT. Длинный текст можно "
             "передать файлом transcript.txt.</p>"))
display(HTML('<textarea style="width:100%;height:420px" readonly>' +
             html.escape(transcript) + '</textarea>'))
display(HTML('<details><summary>Текст с таймкодами</summary><textarea '
             'style="width:100%;height:300px" readonly>' + html.escape(timed) +
             '</textarea></details>'))
display(HTML('<h4>Запрос для красивого конспекта</h4><textarea '
             'style="width:100%;height:140px" readonly>' + html.escape(prompt) + '</textarea>'))
from google.colab import files
try:
    files.download(str(RESULT_DIR / "transcript.zip"))
except Exception as exc:
    print("Браузер не начал скачивание:", exc)
print("В архиве: обычный текст, таймкоды, SRT, JSON и запрос для ChatGPT.")
print("Если скачивание не началось: откройте панель «Файлы» слева, найдите", RESULT_DIR,
      "и скачайте transcript.zip через меню файла.")
'''

cells = [cell('markdown', intro), cell('code', settings, cellView='form'),
         cell('code', setup, cellView='form'), cell('code', storage, cellView='form'),
         cell('code', '#@title 4. Модуль обработки (встроен в блокнот)\n' +
              (ROOT / 'colab_transcribe.py').read_text(encoding='utf-8'), cellView='form'),
         cell('code', execute, cellView='form'), cell('code', display, cellView='form'),
         cell('markdown', '''## Если что-то пошло не так

- **403 / 429 / Sign in to confirm:** это ограничение загрузки YouTube; используйте `UPLOAD_LOCAL_FILE`.
- **CUDA out of memory:** выберите `small`, перезапустите среду и выполните всё заново.
- **Сеанс оборвался:** повторите запуск с теми же настройками. Сохранённые части используются снова.
  Если машина была удалена, это работает только при включённом заранее `SAVE_TO_DRIVE`.
- **Ошибка скачивания модели:** повторите запуск позже; для первой загрузки нужен доступ к серверам моделей.
- **Медленно без GPU:** дождитесь доступной GPU или оставьте `auto` для работы на CPU.

Прогресс сохраняется после каждых 10 минут исходного аудио. `partial.txt` содержит
готовую часть, даже если обработка прервалась. Таймкоды приблизительные; на стыках
возможны ошибки распознавания. Формулы и схемы, видимые только на слайдах, сюда не попадут.

Источники: [Whisper](https://github.com/openai/whisper),
[yt-dlp](https://github.com/yt-dlp/yt-dlp),
[ограничения Colab](https://research.google.com/colaboratory/faq.html).
''')]
notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
    'colab': {'name': 'YouTube_Whisper_Free_Colab.ipynb', 'provenance': []},
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'}, 'accelerator': 'GPU'}, 'cells': cells}
for i, c in enumerate(cells):
    c['id'] = f'lecture-cell-{i}'
(ROOT / 'YouTube_Whisper_Free_Colab.ipynb').write_text(
    json.dumps(notebook, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('Created YouTube_Whisper_Free_Colab.ipynb')
