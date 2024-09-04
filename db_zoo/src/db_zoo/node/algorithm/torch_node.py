from __future__ import annotations
from ...utils.version import check_library_version
check_library_version('torch', '2.0.1')
from typing import Any
from typing_extensions import override
import torch
import torch.nn as nn
from thop import profile
from thop import clever_format
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node

class TorchNode(Node):
  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
      model: nn.Module = None,
      checkpoint_file: str = None
  ) -> None:
    super(TorchNode, self).__init__(
      name=name,
      graph=graph,
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
      # inp = torch.randn(1, 6, 1, 200)
      # flops, params = profile(model, inputs=(inp,))
      # flops, params = clever_format([flops, params], '%.3f')
      # print(flops, params)

  @override
  def start(self):
    ...
