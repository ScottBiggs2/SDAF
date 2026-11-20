"""Print the versions the concept scripts run with (the env-build job of submit_concepts.sh)."""
import torch
import transformers

import specdec_af
from specdec_af.concepts.system import load_system  # noqa: F401  (import check)

print("ENV_OK", torch.__version__, transformers.__version__, "cuda", torch.cuda.is_available(),
      "specdec_af", specdec_af.__file__)
