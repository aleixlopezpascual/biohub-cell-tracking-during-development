"""Hungarian frame linking."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from .graph import Edge, Node
@dataclass(slots=True)
class Tracker:
    max_distance_um: float=7.0
    allow_divisions: bool=True
    def link(self, source:list[Node], target:list[Node], scale:tuple[float,float,float]) -> list[Edge]:
        if not source or not target: return []
        a=np.asarray([n.centroid for n in source])*np.asarray(scale); b=np.asarray([n.centroid for n in target])*np.asarray(scale)
        d=np.linalg.norm(a[:,None]-b[None,:],axis=2); r,c=linear_sum_assignment(np.where(d<=self.max_distance_um,d,1e12))
        return [Edge(source[i].id,target[j].id) for i,j in zip(r,c) if d[i,j]<=self.max_distance_um]
