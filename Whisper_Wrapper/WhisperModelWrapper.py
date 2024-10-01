import whisper
import threading
from typing import Any, Optional


class WhisperModelWrapper:
    """
    Classe per la gestione del modello Whisper per la trascrizione audio.

    :param model_type: Il tipo di modello Whisper da caricare (es: "tiny", "small", "medium", etc.).
    """

    Lock: threading.Lock = threading.Lock()

    def __init__(self, model_type: Optional[str] = "medium") -> None:
        self._model_type: str = model_type
        self._model: whisper.Whisper = None

    def _load_model(self) -> None:
        """
        Carica il modello Whisper ìn modo lazy.

        :return: None.
        """
        self._model = whisper.load_model(self._model_type)

    def transcribe(self, audio_file_path: str) -> str:
        """
        Trascrive il file audio specificato.

        :param audio_file_path: Il percorso del file audio da trascrivere.
        :return: Il testo trascritto.
        """

        try:
            with WhisperModelWrapper.Lock:
                # Assicurati di attendere il caricamento del modello
                if self._model == None:
                    self._load_model()
                # Trascrivi l'audio usando il modello caricato
                result: dict[str, Any] = self._model.transcribe(audio_file_path)
                return result["text"]
        except Exception as e:
            print(f"Errore durante la trascrizione: {e}")
            return ""

    # Getter e Setter per l'attributo model_type
    @property
    def model_type(self) -> str:
        return self._model_type

    @model_type.setter
    def model_type(self, model_type: str) -> None:
        if not isinstance(model_type, str):
            raise TypeError("Il tipo di modello deve essere una stringa.")
        if model_type not in ["tiny", "base", "small", "medium", "large"]:
            raise ValueError(
                "Tipo di modello non valido. Scegli tra: 'tiny', 'base', 'small', 'medium', 'large'."
            )
        self._model_type = model_type

    # Getter e Setter per l'attributo local_data (thread-local)
    @property
    def local_data(self) -> threading.local:
        return self._local_data

    @local_data.setter
    def local_data(self, local_data: threading.local) -> None:
        if not isinstance(local_data, threading.local):
            raise TypeError("local_data deve essere un'istanza di threading.local.")
        self._local_data = local_data
