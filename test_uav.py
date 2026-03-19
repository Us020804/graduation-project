from uav import UAV

# 创建一个无人机对象
uav = UAV(x=100, y=100, step_size=20, x_min=0, x_max=200, y_min=0, y_max=200)

print("初始位置：", uav.get_position())

uav.move('UP')
print("执行 UP 后：", uav.get_position())

uav.move('RIGHT')
print("执行 RIGHT 后：", uav.get_position())

uav.move('DOWN')
print("执行 DOWN 后：", uav.get_position())

uav.move('LEFT')
print("执行 LEFT 后：", uav.get_position())

uav.move('STAY')
print("执行 STAY 后：", uav.get_position())

# 测试边界
uav.reset(0, 0)
print("重置到边界位置：", uav.get_position())

uav.move('LEFT')
print("在左边界执行 LEFT 后：", uav.get_position())

uav.move('DOWN')
print("在下边界执行 DOWN 后：", uav.get_position())

uav.reset(200, 200)
print("重置到右上角：", uav.get_position())

uav.move('RIGHT')
print("在右边界执行 RIGHT 后：", uav.get_position())

uav.move('UP')
print("在上边界执行 UP 后：", uav.get_position())