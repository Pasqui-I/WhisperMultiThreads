import os
import concurrent.futures
import sys
import tempfile
from queue import Queue
import threading
import time
import warnings
from Audio.AudioProcessor import AudioProcessor
from Whisper_Wrapper.WhisperModelWrapper import WhisperModelWrapper

# Crea un evento per terminare il messaggio di caricamento
stop_event: threading.Event = threading.Event()


def show_loading_message() -> None:
    """
    Mostra un messaggio di caricamento che si aggiorna ogni secondo.
    Il ciclo si interrompe quando l'evento stop_event è impostato.
    """
    while not stop_event.is_set():
        sys.stdout.write("\rLoading")
        sys.stdout.flush()
        # Stampa i puntini per simulare l'aggiornamento del caricamento
        for i in range(3):
            time.sleep(1)
            sys.stdout.write(".")
            sys.stdout.flush()
        # Cancella la linea precedente per evitare che i puntini si accumulino
        sys.stdout.write("\r           \r")
        sys.stdout.flush()


def main() -> None:
    """
    Funzione principale che gestisce la trascrizione di un file audio.
    Divide l'audio in frammenti, crea file temporanei e utilizza un ThreadPoolExecutor
    per trascrivere i frammenti in parallelo utilizzando Whisper. I risultati vengono scritti
    in un file di output e i file temporanei vengono eliminati.
    """

    # Prende il numero di core e lo divide a meta per impostare il massimo numero di thread nella pool
    max_threads: int = os.cpu_count() // 2

    # Coda per mantenere i risultati delle trascrizioni
    threads_queue: Queue[concurrent.futures.Future] = Queue()

    # Lista per tracciare i file temporanei
    temp_files: list[str] = []

    # Thread che mostra il messaggio di caricamento
    loading_message: threading.Thread = threading.Thread(target=show_loading_message)

    # Richiede il percorso del file audio all'utente
    file_path: str = input("Inserire il percorso al file audio: ").strip(' "')
    audio_processor: AudioProcessor = AudioProcessor(file_path=file_path)

    # Richiede il tipo di modello Whisper da usare (opzionale)
    model_type: str = input("Inserire il tipo di modello (opzionale): ")
    model_wrapper: WhisperModelWrapper = (
        WhisperModelWrapper(model_type) if model_type else WhisperModelWrapper()
    )

    print(f"Trascrizione in esecuzione su {max_threads} processi paralleli")

    # Crea la cartella di output se non esiste
    os.makedirs("./Output/", exist_ok=True)

    # Svuota il file di output
    open("./Output/Output.txt", "w").close()

    # Crea un ThreadPoolExecutor con metà dei core disponibili del sistema
    loading_message.start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Itera sui frammenti dell'audio e crea un file temporaneo per ogni frammento
        for fragment in audio_processor.split_audio():
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                fragment.export(temp_file.name, format="wav")
                temp_files.append(temp_file.name)  # Traccia il file temporaneo

                # Sottomette il task di trascrizione al pool di thread
                future: concurrent.futures.Future = executor.submit(
                    model_wrapper.transcribe,
                    audio_processor._convert_audio_if_needed(temp_file.name),
                )
                threads_queue.put(future)

    # Scrive i risultati delle trascrizioni nel file di output
    with open("./Output/Output.txt", "a", encoding="utf-8") as output_file:
        while not threads_queue.empty():
            future: concurrent.futures.Future = threads_queue.get()
            result: str = future.result()  # Attende il risultato della trascrizione
            output_file.write(f"{result}\n")

    # Elimina i file temporanei
    for temp_file in temp_files:
        os.remove(temp_file)

    # Ferma il thread del messaggio di caricamento
    stop_event.set()
    loading_message.join()

    print("Trascrizione completata.")


if __name__ == "__main__":
    # Ignora i warning relativi a FP16 non supportato su CPU
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

    # Misura il tempo impiegato per completare la trascrizione
    start_time: float = time.perf_counter()
    main()
    end_time: float = time.perf_counter()
    print(f"Tempo impiegato: {end_time - start_time:.2f} secondi")
