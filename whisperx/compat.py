"""Compatibility utilities for different library versions."""


def get_pyannote_auth_kwargs(use_auth_token):
    """
    Return authentication kwargs compatible with both pyannote-audio 3.x and 4.x.

    pyannote-audio 4.0 renamed the authentication parameter:
    - 3.x: use_auth_token=...
    - 4.x: token=...

    This helper enables backward compatibility by checking the installed version.
    """
    if use_auth_token is None:
        return {}

    try:
        import pyannote.audio
        from packaging.version import Version

        pyannote_version = Version(pyannote.audio.__version__)
        if pyannote_version >= Version("4.0.0"):
            return {"token": use_auth_token}
    except (AttributeError, ImportError):
        # pyannote.audio.__version__ not available, or packaging not installed
        # Fall back to 3.x API for backward compatibility
        pass

    return {"use_auth_token": use_auth_token}
