from typing import MutableMapping, Union
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
import numpy as np

FeatureDict = MutableMapping[str, np.ndarray]
TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]
