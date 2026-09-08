from biohub_tracking.tracking.graph import Node,Edge
from biohub_tracking.tracking.tracker import Tracker
from biohub_tracking.metrics.evaluator import evaluate
def test_core():
    a=[Node("a",0,0,0,0)];b=[Node("b",1,0,1,1)]; e=Tracker().link(a,b,(1,1,1))
    assert e==[Edge("a","b")] and evaluate(a+b,e,a+b,e,scale=(1,1,1)).edge_tp==1
