import os
import sys
import math

if 'SUMO_HOME' not in os.environ:
    raise EnvironmentError("未检测到 SUMO_HOME 环境变量，请先配置 SUMO_HOME")

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

import traci

from uav import UAV


class UAVEnv:
    def __init__(self, sumocfg_file, uav_start=(1600, 1600),
                 uav_radius=200, step_size=20,
                 x_min=0, x_max=5000, y_min=0, y_max=5000,
                 max_steps=20, gui=False, move_cost=0.0):
        self.sumocfg_file = sumocfg_file
        self.gui = gui
        self.move_cost = move_cost

        self.uav = UAV(
            x=uav_start[0],
            y=uav_start[1],
            radius=uav_radius,
            step_size=step_size,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max
        )

        self.uav_start = uav_start
        self.step_count = 0
        self.max_steps = max_steps
        self.started = False

    def start_sumo(self, gui=False):
        if gui:
            sumo_binary = "sumo-gui"
        else:
            sumo_binary = "sumo"

        sumo_cmd = [
            sumo_binary,
            "-c", self.sumocfg_file
        ]

        traci.start(sumo_cmd)
        self.started = True

    def close_sumo(self):
        if self.started:
            traci.close()
            self.started = False

    def reset_uav(self):
        self.uav.reset(self.uav_start[0], self.uav_start[1])
        self.step_count = 0

    def get_vehicle_positions(self):
        vehicle_positions = []
        for veh_id in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(veh_id)
            vehicle_positions.append((veh_id, x, y))
        return vehicle_positions

    def count_covered_vehicles(self):
        uav_x, uav_y = self.uav.get_position()
        count = 0

        for veh_id in traci.vehicle.getIDList():
            veh_x, veh_y = traci.vehicle.getPosition(veh_id)
            distance = math.hypot(veh_x - uav_x, veh_y - uav_y)

            if distance <= self.uav.radius:
                count += 1

        return count

    def get_state(self):
        return self.uav.get_position()

    def simulation_step(self):
        traci.simulationStep()
        self.step_count += 1

    def reset(self):
        if self.started:
            self.close_sumo()

        self.start_sumo(self.gui)
        self.reset_uav()
        return self.get_state()

    def step(self, action):
        self.uav.move(action)
        self.simulation_step()

        next_state = self.get_state()
        covered_count = self.count_covered_vehicles()

        if action == 'STAY':
            move_penalty = 0
        else:
            move_penalty = self.move_cost

        reward = covered_count - move_penalty

        done = self.step_count >= self.max_steps
        return next_state, reward, done