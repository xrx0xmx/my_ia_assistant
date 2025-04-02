from pathlib import Path
from pydub import AudioSegment
import tempfile
from tqdm import tqdm
import openai
import os
import argparse
from typing import List
from dataclasses import dataclass
import logging

# Configuración del logging al inicio del archivo, después de los imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

@dataclass
class AudioConfig:
    """Configuración para el procesamiento de audio"""
    max_size_mb: int = 24
    model_name: str = "whisper-1"
    default_output: str = "transcripcion.txt"
    prompt: str = "Realizauna transcripción de una conversación en español. Es una entrevista que realiza un consultor de sexo masculino a dos trabajadores de la agencia Paladina Marketing. Se trata de la reunión inicial para realizar un diagnostico inicial de una servicio de asesoramiento en procesos de negocio"

class AudioFormatConverter:
    """Maneja la conversión de formatos de audio a WAV"""
    
    @staticmethod
    def to_wav(audio_path: str) -> str:
        """
        Convierte cualquier formato de audio soportado a WAV.
        Returns:
            str: Ruta del archivo WAV temporal.
        Raises:
            Exception: Si hay un error en la conversión
        """
        try:
            audio_format = Path(audio_path).suffix.lower().replace('.', '')
            logging.info(f"Iniciando conversión de archivo {audio_path} (formato: {audio_format})")
            
            audio = AudioSegment.from_file(audio_path, format=audio_format)
            
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            audio.export(temp_wav.name, format='wav')
            logging.info(f"Archivo convertido exitosamente a WAV: {temp_wav.name}")
            return temp_wav.name
            
        except Exception as e:
            logging.error(f"Error en la conversión del audio: {str(e)}")
            raise Exception(f"Error en la conversión del audio: {str(e)}")

class AudioSegmenter:
    """Divide archivos de audio en segmentos más pequeños"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.converter = AudioFormatConverter()

    def get_segments(self, audio_path: str) -> List[str]:
        """
        Divide un archivo de audio en segmentos si excede el tamaño máximo.
        Returns:
            List[str]: Lista de rutas a los segmentos de audio
        """
        try:
            wav_path = self._ensure_wav_format(audio_path)
            audio = AudioSegment.from_file(wav_path, format="wav")
            file_size_mb = Path(wav_path).stat().st_size / (1024 * 1024)
            duration_seconds = len(audio) / 1000  # Convertir milisegundos a segundos
            
            logging.info(f"Tamaño del archivo de audio: {file_size_mb:.2f} MB")
            logging.info(f"Duración del audio: {duration_seconds:.2f} segundos")
            
            if file_size_mb <= self.config.max_size_mb:
                logging.info("El archivo no requiere segmentación")
                return [wav_path]
            else:
                logging.info(f"Iniciando segmentación del archivo (tamaño máximo por segmento: {self.config.max_size_mb} MB)")
                return self._create_segments(audio, file_size_mb)
            
        except FileNotFoundError:
            logging.error(f"Archivo de audio no encontrado: {audio_path}")
            raise FileNotFoundError(f"Archivo de audio no encontrado: {audio_path}")
        except Exception as e:
            logging.error(f"Error al segmentar el audio: {str(e)}")
            raise Exception(f"Error al segmentar el audio: {str(e)}")

    def _ensure_wav_format(self, audio_path: str) -> str:
        """Asegura que el archivo esté en formato WAV"""
        return (audio_path if audio_path.lower().endswith('.wav') 
                else self.converter.to_wav(audio_path))

    def _create_segments(self, audio: AudioSegment, file_size_mb: float) -> List[str]:
        """Crea segmentos de audio basados en el tamaño máximo permitido"""
        segment_duration = (self.config.max_size_mb / file_size_mb) * len(audio)
        segment_duration_seconds = segment_duration / 1000  # Convertir a segundos
        segments = []
        total_segments = len(audio) // int(segment_duration) + 1
        
        logging.info(f"Creando {total_segments} segmentos...")
        logging.info(f"Duración aproximada por segmento: {segment_duration_seconds:.2f} segundos")
        
        for i, start in enumerate(range(0, len(audio), int(segment_duration)), 1):
            end = min(start + int(segment_duration), len(audio))
            segment = audio[start:end]
            segment_length_seconds = len(segment) / 1000
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                segment.export(temp_file.name, format='wav')
                segment_size_mb = Path(temp_file.name).stat().st_size / (1024 * 1024)
                segments.append(temp_file.name)
                logging.info(f"Segmento {i}/{total_segments} creado:")
                logging.info(f"  - Archivo: {temp_file.name}")
                logging.info(f"  - Duración: {segment_length_seconds:.2f} segundos")
                logging.info(f"  - Tamaño: {segment_size_mb:.2f} MB")
        
        return segments

class WhisperTranscriber:
    """Maneja la transcripción de audio usando OpenAI Whisper"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self._setup_api_key()

    def _setup_api_key(self):
        """Configura la API key de OpenAI"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("API key no configurada. Establece la variable de entorno OPENAI_API_KEY")
        openai.api_key = api_key

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe un archivo de audio usando la API de OpenAI Whisper.
        Returns:
            str: Texto transcrito
        """
        try:
            self._validate_file_exists(audio_path)
            return self._process_transcription(audio_path)
            
        except Exception as e:
            raise Exception(f"Error en la transcripción: {str(e)}")

    def _validate_file_exists(self, audio_path: str):
        """Valida que el archivo exista"""
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"No se encontró el archivo: {audio_path}")

    def _process_transcription(self, audio_path: str) -> str:
        """Procesa la transcripción del audio"""
        with open(audio_path, "rb") as audio:
            return openai.audio.transcriptions.create(
                model=self.config.model_name,
                file=audio,
                response_format="text",
                language="es",
                prompt=self.config.prompt
            )

class TranscriptionManager:
    """Coordina el proceso completo de transcripción"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.segmenter = AudioSegmenter(config)
        self.transcriber = WhisperTranscriber(config)

    def process_audio(self, input_path: str, output_path: str) -> None:
        """Procesa el audio completo y guarda la transcripción"""
        segments = self.segmenter.get_segments(input_path)
        transcription = self._transcribe_segments(segments)
        
        self._save_transcription(transcription, output_path)
        self._cleanup_segments(segments, input_path)

    def _transcribe_segments(self, segments: List[str]) -> str:
        """Transcribe todos los segmentos de audio"""
        transcriptions = []
        for segment_path in tqdm(segments, desc="Transcribiendo segmentos"):
            result = self.transcriber.transcribe(segment_path)
            if result:
                transcriptions.append(result)
        return " ".join(transcriptions)

    def _save_transcription(self, text: str, output_path: str):
        """Guarda la transcripción en un archivo"""
        print("\nTranscripción exitosa:")
        print(text)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

    def _cleanup_segments(self, segments: List[str], original_path: str):
        """Limpia los archivos temporales"""
        for segment in segments:
            if os.path.exists(segment) and segment != original_path:
                os.unlink(segment)

def parse_arguments() -> argparse.Namespace:
    """Configura y procesa los argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Transcribe un archivo de audio a texto.')
    parser.add_argument('input_file', 
                       help='Ruta al archivo de audio de entrada (formatos soportados: mp3, wav, ogg, flac, m4a, etc)')
    parser.add_argument('-o', '--output', default='output.txt',
                        help='Ruta del archivo de salida para la transcripción (por defecto: transcripcion.txt)')
    parser.add_argument('-p', '--prompt',
                        help='Prompt para guiar la transcripción (opcional)')
    return parser.parse_args()

def main():
    """Función principal del programa"""
    args = parse_arguments()
    config = AudioConfig()
    if args.prompt:
        config.prompt = args.prompt
    
    try:
        manager = TranscriptionManager(config)
        manager.process_audio(args.input_file, args.output)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
