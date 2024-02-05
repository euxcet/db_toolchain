from __future__ import annotations
from ...utils.version import check_library_version
check_library_version('torch', '2.0.1')
from typing import Any
import torch
import torch.nn as nn
from db_graph.framework.node import Node

class TorchNode(Node):
  def __init__(
      self,
      name: str,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
      model: nn.Module = None,
      checkpoint_file: str = None
  ) -> None:
    super(TorchNode, self).__init__(
      name=name,
      input_edges=input_edges,
      output_edges=output_edges,
    )
    self.name = name
    # model
    if model is not None:
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.model: nn.Module = model
      self.model.load_state_dict(torch.load(checkpoint_file, map_location=self.device))
      self.model.eval()