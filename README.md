# Transcriptor de Audio

Una herramienta simple para transcribir archivos de audio a texto utilizando la API de OpenAI Whisper.

## Requisitos

- Python 3.x
- OpenAI API key
- Dependencias listadas en `requirements.txt`:
  - openai
  - python-dotenv

## Instalación

1. Clona este repositorio o descarga los archivos
2. Instala las dependencias:

pip install -r requirements.txt

3. Crea un archivo `.env` en el directorio raíz y añade tu API key de OpenAI:

OPENAI_API_KEY=tu_api_key_aquí

## Uso

1. Coloca tu archivo de audio en el directorio del proyecto
2. Ejecuta el script:

```bash
python transcript.py
```

3. Sigue las instrucciones en pantalla para seleccionar el archivo de audio
4. La transcripción se guardará en un archivo de texto con el mismo nombre que el archivo de audio

## Características

- Soporta múltiples formatos de audio (mp3, wav, m4a, etc.)
- Utiliza el modelo Whisper de OpenAI para transcripción precisa
- Manejo seguro de credenciales mediante variables de entorno
- Interfaz simple de línea de comandos

## Notas

- Asegúrate de tener una conexión a internet estable
- El tiempo de procesamiento dependerá del tamaño del archivo de audio
- Se requiere una API key válida de OpenAI

## Licencia

Este proyecto está bajo la Licencia MIT.
