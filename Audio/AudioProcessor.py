import os
from typing import Generator
from pydub import AudioSegment
import tempfile


class AudioProcessor:
    """
    Classe per la gestione e la suddivisione di file audio in frammenti.

    :param file_path: Il percorso del file audio.
    :param fragment_size: La dimensione del frammento (in campioni) da cui dividere il file audio. Default è 32000.
    """

    def __init__(self, file_path: str, fragment_size: int = 32000) -> None:
        self._file_path: str = file_path
        self._fragment_size: int = fragment_size

    def split_audio(self) -> Generator[AudioSegment, None, None]:
        """
        Generatore che divide l'audio in frammenti di dimensione specificata.
        Se un frammento è inferiore alla dimensione richiesta, viene riempito con silenzio.

        :return: Un generatore di oggetti AudioSegment.
        """
        self._file_path = self._convert_audio_if_needed(self._file_path)
        audio: AudioSegment = AudioSegment.from_file_using_temporary_files(
            self._file_path
        )

        for start in range(0, len(audio), self._fragment_size):
            end: int = min(start + self._fragment_size, len(audio))
            fragment: AudioSegment = audio[start:end]

            # Aggiungi silenzio se il frammento è più piccolo della dimensione minima
            if len(fragment) < self._fragment_size:
                fragment = fragment + AudioSegment.silent(
                    duration=self._fragment_size - len(fragment)
                )

            yield fragment

    def _convert_audio_if_needed(self, file_path: str) -> str:
        """
        Converte l'audio in formato mono e a 16000 Hz se necessario.
        Se il formato non è WAV, lo converte in WAV.

        :param file_path: Il percorso del file audio originale.
        :return: Il percorso del file audio convertito o l'originale se già conforme.
        """
        audio: AudioSegment = AudioSegment.from_file(file_path)

        # Controlla se il file non è WAV e lo converte in un file WAV temporaneo
        if (
            not file_path.lower().endswith(".wav")
            or audio.channels != 1
            or audio.frame_rate != 16000
        ):
            temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(temp_wav_file.name, format="wav")
            return temp_wav_file.name

        return file_path

    # Getter e Setter con controlli per gli attributi
    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, file_path: str) -> None:
        if not isinstance(file_path, str):
            raise TypeError("Il percorso del file deve essere una stringa.")
        if not file_path:
            raise ValueError("Il percorso del file non può essere vuoto.")
        self._file_path = file_path

    @property
    def fragment_size(self) -> int:
        return self._fragment_size

    @fragment_size.setter
    def fragment_size(self, fragment_size: int) -> None:
        if not isinstance(fragment_size, int):
            raise TypeError("La dimensione del frammento deve essere un numero intero.")
        if fragment_size <= 0:
            raise ValueError("La dimensione del frammento deve essere positiva.")
        self._fragment_size = fragment_size
