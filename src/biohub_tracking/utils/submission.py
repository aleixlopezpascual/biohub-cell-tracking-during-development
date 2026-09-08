"""Strict deterministic CSV exporter."""
import pandas as pd
from ..tracking.graph import Node,Edge
SCHEMA=["row_type","node_id","t","z","y","x","source_id","target_id","id","dataset"]
def export_submission(nodes:list[Node],edges:list[Edge],path:str,dataset:str)->pd.DataFrame:
    rows=[{"row_type":"node","node_id":str(n.id),"t":n.t,"z":n.z,"y":n.y,"x":n.x,"source_id":"","target_id":"","id":str(n.id),"dataset":dataset} for n in nodes]
    rows += [{"row_type":"edge","node_id":"","t":"","z":"","y":"","x":"","source_id":str(e.source_id),"target_id":str(e.target_id),"id":f"{e.source_id}->{e.target_id}","dataset":dataset} for e in edges]
    out=pd.DataFrame(rows,columns=SCHEMA); out.to_csv(path,index=False); return out
