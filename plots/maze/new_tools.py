import numpy as np




# def no_collision(old_xy, new_xy):
#     """
#     xys: shape of N x 2 np.array
#     """
#     old_x, old_y = old_xy
#     x, y = new_xy
#     x_samples = np.linspace(old_x, x, 8)
#     y_samples = np.linspace(old_y, y, 8)

#     result = np.all(np.greater(22.09, x_samples) * np.greater(y_samples, -2.09) * np.greater(14.09, y_samples))
#     result = result * (~np.any(np.greater(20.41, x_samples) * np.greater(y_samples, 2.09) * np.greater(12.91, y_samples)))

#     return result


def no_collision(old_xy, new_xy): # bigger tunnel
    """
    xys: shape of N x 2 np.array
    """
    old_x, old_y = old_xy
    x, y = new_xy
    x_samples = np.linspace(old_x, x, 8)
    y_samples = np.linspace(old_y, y, 8)

    result = np.all(np.greater(22.09, x_samples) * np.greater(y_samples, -2.09) * np.greater(14.59, y_samples))
    result = result * (~np.any(np.greater(20.41, x_samples) * np.greater(y_samples, 2.09) * np.greater(12.91, y_samples)))

    return result




# print(no_collision((20.808480496526926, 12.027405999862161), (17.560829741916802, 15.059090862729363)))

# a = np.array([
#     [20.5, 13],
#     [16, 2.03],
#     [-2, 0.8],
#     [20.6, 0.83]
# ])
# print(all_valid(np.zeros((4, 2)), a))
# print(valid(a))


# def nearest_distance(x, y):
#     first_corner = np.sqrt((x - 20.41)**2 + (y - 2.09)**2)
#     second_corner = np.sqrt((x - 20.41)**2 + (y - 12.91)**2)

#     distances = np.zeros((len(x), 6), dtype=np.float32)
#     distances[:, 0] = y + 2.09
#     distances[:, 1] = 22.09 - x
#     distances[:, 2] = 14.09 - y

#     distances[:, 3] = np.where(x < 20.41, 2.09 - y, first_corner)
#     distances[:, 4] = np.where(x < 20.41, y - 12.91, second_corner)
#     distances[:, 5] = np.where(np.greater(y, 2.09) * np.greater(12.91, y), x - 20.41, second_corner)

#     return np.min(np.abs(distances), axis=-1)


def nearest_distance(x, y): # bigger tunnel
    first_corner = np.sqrt((x - 20.41)**2 + (y - 2.09)**2)
    second_corner = np.sqrt((x - 20.41)**2 + (y - 12.91)**2)

    distances = np.zeros((len(x), 6), dtype=np.float32)
    distances[:, 0] = y + 2.09
    distances[:, 1] = 22.09 - x
    distances[:, 2] = 14.59 - y

    distances[:, 3] = np.where(x < 20.41, 2.09 - y, first_corner)
    distances[:, 4] = np.where(x < 20.41, y - 12.91, second_corner)
    distances[:, 5] = np.where(np.greater(y, 2.09) * np.greater(12.91, y), x - 20.41, second_corner)

    return np.min(np.abs(distances), axis=-1)




# def projection(x, y):
#     in_first_stage = np.int32(np.greater(20.41, x) * np.greater(2.09, y))
#     in_third_stage = np.int32(np.greater(y, 0.84) * np.greater(12.91, y) * np.greater(x, 20.41))
#     in_fifth_stage = np.int32(np.greater(y, 12.91) * np.greater(20.66, x))
#     in_second_stage = np.int32(np.greater(x, 20.41) * np.greater(0.84, y))
#     in_fourth_stage = 1 - in_first_stage - in_second_stage - in_third_stage - in_fifth_stage

#     result = in_first_stage * x + in_third_stage * (y + 20.8895) + in_fifth_stage * (55.3862 - x)
#     result += in_second_stage * (20.41 + 0.84*np.arcsin((x-20.41)/np.sqrt((x-20.41)**2 + 1e-6 + (y-0.84)**2)))
#     result += in_fourth_stage * (33.7995 + 0.59*np.arcsin((y-12.91)/np.sqrt((x-20.66)**2 + 1e-6 + (y-12.91)**2)))

#     return result



def projection(x, y): # bigger tunnel
    in_first_stage = np.int32(np.greater(20.41, x) * np.greater(2.09, y))
    in_third_stage = np.int32(np.greater(y, 0.84) * np.greater(12.91, y) * np.greater(x, 20.41))
    in_fifth_stage = np.int32(np.greater(y, 12.91) * np.greater(20.41, x))
    in_second_stage = np.int32(np.greater(x, 20.41) * np.greater(0.84, y))
    in_fourth_stage = 1 - in_first_stage - in_second_stage - in_third_stage - in_fifth_stage

    result = in_first_stage * x + in_third_stage * (y + 20.8895) + in_fifth_stage * (55.529 - x)
    result += in_second_stage * (20.41 + 0.84*np.arcsin((x-20.41)/np.sqrt((x-20.41)**2 + 1e-6 + (y-0.84)**2)))
    result += in_fourth_stage * (33.7995 + 0.84*np.arcsin((y-12.91)/np.sqrt((x-20.41)**2 + 1e-6 + (y-12.91)**2)))

    return result




# print(projection(a))

# print(penalty(a))
# print(benifit(a))


def reward(old_xy, new_xys):
    """
    xys: shape of N x 2 np.array
    """
    old_x, old_y = old_xy
    x, y = new_xys.T
    x_samples = np.linspace(old_x, x, 8)
    y_samples = np.linspace(old_y, y, 8)
    
    valid = np.all(np.greater(22.09, x_samples) * np.greater(y_samples, -2.09) * np.greater(14.59, y_samples), axis=0)
    valid = valid * (~np.any(np.greater(20.41, x_samples) * np.greater(y_samples, 2.09) * np.greater(12.91, y_samples), axis=0))
    valid = np.int32(valid)

    # if old_x < 20.41 and old_y < 2.09:
    #     valid = valid * np.where(y < 2.09, 1, 0)
    # elif old_y < 12.91:
    #     valid = valid * np.where(x > 20.41, 1, 0)
    # else:
    #     valid = valid * np.where(y > 12.91, 1, 0)

    d = nearest_distance(x, y)
    benifit = projection(x, y)

    # penalty = 0.3125 * np.power(d + 0.5, -4)
    penalty = 0.162 * np.power(d + 0.18, -2)
    # penalty = 2.048 * np.power(d + 0.8, -4)
    
    result = valid*(benifit - penalty) - 100*(1 - valid)
    # result = valid*(benifit) - 100*(1 - valid)

    return result


# print(reward((0, 0), a))
# print(projection(a))


