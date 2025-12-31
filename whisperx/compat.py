"""Compatibility utilities for pyannote-audio versions."""


def get_pyannote_auth_kwargs(use_auth_token):
    """Return auth kwargs for pyannote-audio 3.x (use_auth_token) or 4.x (token)."""
    if use_auth_token is None:
        return {}

    try:
        import pyannote.audio
        major = int(pyannote.audio.__version__.split('.')[0])
        if major >= 4:
            return {"token": use_auth_token}
    except (AttributeError, ValueError):
        pass

    return {"use_auth_token": use_auth_token}
