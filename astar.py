import heapq


class AStarPlanner:
    def __init__(self, x_min=0, x_max=5000, y_min=0, y_max=5000, step_size=20):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.step_size = step_size

        self.cols = int((x_max - x_min) / step_size) + 1
        self.rows = int((y_max - y_min) / step_size) + 1

    def pos_to_grid(self, pos):
        """
        连续坐标 -> 网格坐标
        """
        x, y = pos
        gx = round((x - self.x_min) / self.step_size)
        gy = round((y - self.y_min) / self.step_size)

        gx = max(0, min(gx, self.cols - 1))
        gy = max(0, min(gy, self.rows - 1))
        return gx, gy

    def grid_to_pos(self, grid):
        """
        网格坐标 -> 连续坐标
        """
        gx, gy = grid
        x = self.x_min + gx * self.step_size
        y = self.y_min + gy * self.step_size
        return x, y

    def heuristic(self, a, b):
        """
        曼哈顿距离启发函数
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, node):
        """
        四邻域
        """
        x, y = node
        neighbors = []

        directions = [
            (x, y + 1),   # UP
            (x, y - 1),   # DOWN
            (x - 1, y),   # LEFT
            (x + 1, y)    # RIGHT
        ]

        for nx, ny in directions:
            if 0 <= nx < self.cols and 0 <= ny < self.rows:
                neighbors.append((nx, ny))

        return neighbors

    def astar_search(self, start, goal):
        """
        A* 搜索
        :param start: 起点网格 (gx, gy)
        :param goal: 终点网格 (gx, gy)
        :return: 路径列表 [start, ..., goal]
        """
        if start == goal:
            return [start]

        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def reconstruct_path(self, came_from, current):
        """
        还原路径
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def path_to_action(self, path):
        """
        根据路径的前两点，转换成动作
        """
        if len(path) < 2:
            return 'STAY'

        (x1, y1), (x2, y2) = path[0], path[1]

        if x2 == x1 and y2 == y1 + 1:
            return 'UP'
        elif x2 == x1 and y2 == y1 - 1:
            return 'DOWN'
        elif x2 == x1 - 1 and y2 == y1:
            return 'LEFT'
        elif x2 == x1 + 1 and y2 == y1:
            return 'RIGHT'
        else:
            return 'STAY'

    def get_vehicle_center(self, vehicle_positions):
        """
        计算车辆几何中心
        :param vehicle_positions: [(veh_id, x, y), ...]
        :return: (center_x, center_y)
        """
        if not vehicle_positions:
            return None

        xs = [x for _, x, _ in vehicle_positions]
        ys = [y for _, _, y in vehicle_positions]

        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        return center_x, center_y

    def choose_action(self, uav_pos, vehicle_positions):
        """
        A* 规划当前一步动作
        :param uav_pos: 无人机当前位置 (x, y)
        :param vehicle_positions: 当前车辆位置列表
        :return: action, target_pos, path
        """
        if not vehicle_positions:
            return 'STAY', None, []

        target_pos = self.get_vehicle_center(vehicle_positions)

        start_grid = self.pos_to_grid(uav_pos)
        goal_grid = self.pos_to_grid(target_pos)

        path = self.astar_search(start_grid, goal_grid)
        action = self.path_to_action(path)

        return action, target_pos, path