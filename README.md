# Comando di build

-   Windows: pyinstaller --onefile --add-data "./.venv/Lib/site-packages/whisper/assets/mel_filters.npz;whisper/assets" --optimize 2 main.py
-   Linux: pyinstaller --onefile --add-data "./.venv/lib/python3.12/site-packages/whisper/assets/multilingual.tiktoken:whisper/assets" --add-data "./.venv/lib/python3.12/site-packages/whisper/assets/mel_filters.npz:whisper/assets" --optimize 2 main.py
