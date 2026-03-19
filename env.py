import os
import sys
import math
import traci

from uav import UAV


class UAVEnv:
    def __init__(self, sumocfg_file, uav_start=(1600, 1600),
                 uav_radius=200, step_size=20,
                 x_min=0, x_max=5000, y_min=0, y_max=5000,
                 max_steps=20, gui=False):
        self.sumocfg_file = sumocfg_file
        self.gui = gui

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

    def start_sumo(self):
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            if tools not in sys.path:
                sys.path.append(tools)
        else:
            print("警告：未检测到 SUMO_HOME 环境变量，请确认 SUMO 已正确安装")

        sumo_binary = "sumo-gui" if self.gui else "sumo"

        traci.start([
            sumo_binary,
            "-c", self.sumocfg_file,
            "--no-step-log", "true",
            "--duration-log.disable", "true"
        ])
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

        self.start_sumo()
        self.reset_uav()
        return self.get_state()

    def step(self, action):
        self.uav.move(action)
        self.simulation_step()

        next_state = self.get_state()
        covered_count = self.count_covered_vehicles()

        if action == 'STAY':
            move_cost = 0
        else:
            move_cost = 0.1
        #移动代价让stay更有吸引力，鼓励无人机在覆盖较多车辆的位置停留，而不是频繁移动，减少能量损耗
        reward = covered_count - move_cost

        done = self.step_count >= self.max_steps
        return next_state, reward, done