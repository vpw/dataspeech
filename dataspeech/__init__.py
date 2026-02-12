from .cpu_enrichments import rate_apply
from .gpu_enrichments import snr_apply, squim_apply

try:
    from .gpu_enrichments import pitch_apply
except (FileNotFoundError, ImportError):
    pitch_apply = None