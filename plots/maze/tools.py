import numpy as np



# def valid(x, y):
#     if x < 22.09 and y > -2.09:
#         if x < 20.41 and y > 2.09:
#             return False
#         else:
#             return True
#     else:
#         return False


def all_valid(old_xy, new_xy):
    """
    xys: shape of N x 2 np.array
    """
    old_x, old_y = old_xy
    x, y = new_xy
    x_samples = np.linspace(old_x, x, 8)
    y_samples = np.linspace(old_y, y, 8)

    result = np.where(x_samples < 22.09, 1, 0) * np.where(y_samples > -2.09, 1, 0) - \
        np.where(x_samples < 20.41, 1, 0) * np.where(y_samples > 2.09, 1, 0)

    return np.all(result)



# a = np.array([
#     [20.5, 17],
#     [16, 2.03],
#     [-2, 0.8],
#     [21, 0.83]
# ])
# print(all_valid(np.zeros((4, 2)), a))
# print(valid(a))


# def penalty(xys):
#     x = xys[:, 0]
#     y = xys[:, 1]
#     in_first_part = np.where(x < 20.41, 1, 0)
#     in_third_part = np.where(y > 2.09, 1, 0)
#     in_second_part = 1 - in_first_part - in_third_part

#     distances = np.zeros((len(xys), 3), dtype=np.float32)
#     distances[:, 0] = y + 2.09
#     distances[:, 1] = 22.09 - x
#     distances[:, 2] = in_first_part * (2.09 - y) + \
#         in_second_part * np.sqrt((x - 20.41)**2 + (y - 2.09)**2) + \
#             in_third_part * (x - 20.41)
    
#     d = np.min(np.abs(distances), axis=-1)
#     return 0.75*np.power(d + 0.5, -3)


def projection(xys):
    """
    xys: shape of N x 2 np.array
    """
    x = xys[:, 0]
    y = xys[:, 1]
    in_first_stage = np.where(x < 20.41, 1, 0)
    in_third_stage = (1 - in_first_stage) * np.where(y > 0.84, 1, 0)
    in_second_stage = 1 - in_first_stage - in_third_stage

    result = in_first_stage * x + \
        in_second_stage * (20.41 + 0.84*np.arcsin((x-20.41)/np.sqrt((x-20.41)**2 + 1e-6 + (y-0.84)**2))) + \
              in_third_stage * (20.8895 + y)
    
    return result


# print(penalty(a))
# print(benifit(a))


def reward(old_xy, new_xys):
    """
    xys: shape of N x 2 np.array
    """
    # still_valid = valid(xys)
    old_x, old_y = old_xy
    x, y = new_xys.T
    x_samples = np.linspace(old_x, x, 8)
    y_samples = np.linspace(old_y, y, 8)

    # still_valid = np.where(x < 22.09, 1, 0)*np.where(y > -2.09, 1, 0) - \
    #     np.where(x < 20.41, 1, 0)*np.where(y > 2.09, 1, 0)

    # valid = np.int32(np.all(x_samples < 22.09, axis=0) * np.all(y_samples > -2.09, axis=0)) - \
    #     np.int32(np.any(x_samples < 20.41, axis=0) * np.any(y_samples > 2.09, axis=0))
    
    valid = np.all(np.greater(22.09, x_samples) * np.greater(y_samples, -2.09), axis=0)^np.any(np.greater(20.41, x_samples) * np.greater(y_samples, 2.09), axis=0)
    # print(np.all(np.greater(22.09, x_samples) * np.greater(y_samples, -2.09), axis=0))

    in_first_part = np.where(x < 20.41, 1, 0)
    in_third_part = np.where(y > 2.09, 1, 0)
    in_second_part = 1 - in_first_part - in_third_part

    in_first_stage = np.where(x < 20.41, 1, 0)
    in_third_stage = (1 - in_first_stage) * np.where(y > 0.84, 1, 0)
    in_second_stage = 1 - in_first_stage - in_third_stage

    distances = np.zeros((len(new_xys), 3), dtype=np.float32)
    distances[:, 0] = y + 2.09
    distances[:, 1] = 22.09 - x
    distances[:, 2] = in_first_part * (2.09 - y) + \
        in_second_part * np.sqrt((x - 20.41)**2 + (y - 2.09)**2) + \
            in_third_part * (x - 20.41)
    
    d = np.min(np.abs(distances), axis=-1)
    penalty = 6 * np.power(d + 1, -4)
    
    benifit = in_first_stage * x + \
        in_second_stage * (20.41 + 0.84*np.arcsin((x-20.41)/np.sqrt((x-20.41)**2 + 1e-6 + (y-0.84)**2))) + \
            in_third_stage * (20.8895 + y)
    
    result = valid*(benifit - penalty) - 100*(1 - valid)

    return result


# print(reward((0, 0), a))
# print(projection(a))


