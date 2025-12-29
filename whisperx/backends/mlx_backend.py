'''MLX backend for Apple Silicon acceleration.'''
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from faster_whisper.tokenizer import Tokenizer
    from faster_whisper.transcribe import TranscriptionOptions

MODEL_REPOS = {
    'tiny': 'mlx-community/whisper-tiny',
    'tiny.en': 'mlx-community/whisper-tiny.en',
    'base': 'mlx-community/whisper-base',
    'base.en': 'mlx-community/whisper-base.en',
    'small': 'mlx-community/whisper-small',
    'small.en': 'mlx-community/whisper-small.en',
    'medium': 'mlx-community/whisper-medium',
    'medium.en': 'mlx-community/whisper-medium.en',
    'large': 'mlx-community/whisper-large-v3',
    'large-v2': 'mlx-community/whisper-large-v2',
    'large-v3': 'mlx-community/whisper-large-v3',
    'turbo': 'mlx-community/whisper-large-v3-turbo',
}


class _ModelProxy:
    '''Proxy providing ctranslate2 model interface attributes.'''

    def __init__(self, is_multilingual: bool = True):
        self.device = 'mps'
        self.device_index = [0]
        self.is_multilingual = is_multilingual


class MLXWhisperModel:
    '''Drop-in replacement for faster_whisper.WhisperModel using MLX.'''

    def __init__(
        self,
        model_size_or_path: str,
        device: str = 'mps',
        device_index: int = 0,
        compute_type: str = 'float16',
        cpu_threads: int = 4,
        download_root: Optional[str] = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        from transformers import WhisperTokenizer

        self._repo = MODEL_REPOS.get(model_size_or_path, model_size_or_path)
        is_multilingual = not model_size_or_path.endswith('.en')

        self.model = _ModelProxy(is_multilingual)
        self.feat_kwargs = {'feature_size': 80}
        self.max_length = 448
        self.time_precision = 0.02

        base_name = model_size_or_path.split('.')[0]
        if base_name in ('turbo',):
            base_name = 'large-v3'
        self.hf_tokenizer = WhisperTokenizer.from_pretrained(
            f'openai/whisper-{base_name}'
        )

        self._pending_audio: Optional[np.ndarray] = None
        self._pending_language: Optional[str] = None

    def set_audio_for_batch(
        self, audio: np.ndarray, language: Optional[str], task: str
    ):
        self._pending_audio = audio
        self._pending_language = language

    def encode(self, features: np.ndarray) -> np.ndarray:
        return features

    def generate_segment_batched(
        self,
        features: np.ndarray,
        tokenizer: 'Tokenizer',
        options: 'TranscriptionOptions',
        encoder_output=None,
    ) -> List[str]:
        import mlx_whisper

        if self._pending_audio is None:
            raise RuntimeError('Audio not set - call set_audio_for_batch() first')

        result = mlx_whisper.transcribe(
            self._pending_audio,
            path_or_hf_repo=self._repo,
            language=self._pending_language
            or (tokenizer.language_code if tokenizer else None),
            initial_prompt=options.initial_prompt if options else None,
            word_timestamps=False,
            condition_on_previous_text=options.condition_on_previous_text
            if options
            else False,
        )

        self._pending_audio = None

        return [result.get('text', '').strip()]

    def get_prompt(
        self,
        tokenizer: 'Tokenizer',
        previous_tokens: List[int],
        without_timestamps: bool = False,
        prefix: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> List[int]:
        prompt = list(tokenizer.sot_sequence)
        if without_timestamps:
            prompt.append(tokenizer.no_timestamps)
        return prompt
