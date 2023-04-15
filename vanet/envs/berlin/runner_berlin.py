import os, sys

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import traci.constants as tc
from sumolib import checkBinary


def run():
    tot = 0
    vehicleList = traci.vehicle.getIDList()
    # print(len(vehicleList), file=f)
    for vehID in vehicleList:
        position = traci.vehicle.getPosition(vehID)
        print(vehID, position[0], position[1], file=f)


configName = "berlin.sumocfg"
sumoBinary = checkBinary('sumo-gui')
sumoCmd = [sumoBinary, "-c", configName]

traci.start(sumoCmd)
with open("vehicle_trace.txt", "w") as f:
    pass

step = 0
while traci.simulation.getMinExpectedNumber() > 0:
    with open("vehicle_trace.txt", "a+") as f:
        print('step', step, file=f)
        run()
        traci.simulationStep()
        step += 1

traci.close()