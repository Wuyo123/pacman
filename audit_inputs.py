"""Deterministic encoder identifiability audit; no solver trial or training."""
import json
import numpy as np
from belief_mita.runtime import ROOT, initial_state, MapTopology, ObservationEncoder

state=initial_state()
observation=state.makeObservation(0)
topology=MapTopology(state.getWalls())
def encoded(obs):
    encoder=ObservationEncoder(topology,[0,2],True)
    encoder.update(obs,0)
    return encoder.history_array()
base=encoded(observation)
checks={}
changed=observation.deepCopy()
changed.data.score=10
changed.data.agentStates[0].numReturned=10
checks['score_and_returned_omitted']=bool(np.array_equal(base,encoded(changed)))
changed=observation.deepCopy()
changed.data.agentStates[1].scaredTimer=40
checks['enemy_scared_timer_omitted']=bool(np.array_equal(base,encoded(changed)))
changed=observation.deepCopy()
changed.data.agentStates[0].scaredTimer=40
checks['own_scared_timer_omitted']=bool(np.array_equal(base,encoded(changed)))
from game import Configuration
full=state.deepCopy()
point=next(point for point in topology.cells if point[0]>=topology.width//2)
for actor in (0,2):
    full.data.agentStates[actor].configuration=Configuration(point,'Stop')
    full.data.agentStates[actor].isPacman=True
food=full.getBlueFood().asList()
for x,y in food[:9]:
    full.data.food[x][y]=False
full.data.agentStates[0].numCarrying=3
full.data.agentStates[2].numCarrying=2
full.data.agentStates[0].numReturned=4
full.data.score=4
first=full.makeObservation(0)
other=first.deepCopy()
other.data.agentStates[0].numCarrying=4
other.data.agentStates[0].numReturned=3
other.data.score=3
checks['colocated_first_agent_carry_overwritten']=bool(np.array_equal(encoded(first),encoded(other)))
assert all(checks.values()),checks
path=ROOT/'runs/root_cause_20261105'
path.mkdir(exist_ok=True)
(path/'input_audit.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks))
