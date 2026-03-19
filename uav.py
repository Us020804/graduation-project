class UAV:
    def __init__(self, x=0, y=0, radius=200, step_size=20,
                 x_min=0, x_max=1000, y_min=0, y_max=1000):
        self.x = x
        self.y = y
        self.radius = radius
        self.step_size = step_size

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def get_position(self):
        return self.x, self.y

    def move(self, action):
        new_x, new_y = self.x, self.y

        if action == 'UP':
            new_y += self.step_size
        elif action == 'DOWN':
            new_y -= self.step_size
        elif action == 'LEFT':
            new_x -= self.step_size
        elif action == 'RIGHT':
            new_x += self.step_size
        elif action == 'STAY':
            pass
        else:
            print(f"警告：无效动作 {action}，无人机保持不动")

        new_x = max(self.x_min, min(new_x, self.x_max))
        new_y = max(self.y_min, min(new_y, self.y_max))

        self.x, self.y = new_x, new_y

    def reset(self, x, y):
        self.x = x
        self.y = y