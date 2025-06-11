import os
import logging

logger = logging.getLogger("dinov2")


def _xformers_is_available(layer):

    XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
    xformers = None
    try:
        if XFORMERS_ENABLED:
            import xformers

            logger.info(f"xFormers is available ({layer})")
        else:
            logger.warning(f"xFormers is disabled ({layer})")
            raise ImportError
    except ImportError:
        logger.warning(f"xFormers is not available ({layer})")

    return xformers is not None
