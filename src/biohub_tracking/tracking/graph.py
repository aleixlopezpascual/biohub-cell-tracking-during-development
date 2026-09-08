"""Typed graph primitives."""
from dataclasses import dataclass
from typing import Hashable
NodeId = Hashable
@dataclass(frozen=True, slots=True)
class Node:
    id: NodeId; t: int; z: float; y: float; x: float
    @property
    def centroid(self) -> tuple[float,float,float]: return self.z,self.y,self.x
@dataclass(frozen=True, slots=True)
class Edge:
    source_id: NodeId; target_id: NodeId
