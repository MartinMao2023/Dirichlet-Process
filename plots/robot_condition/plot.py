import numpy as np
import matplotlib.pyplot as plt

intact_image = np.load("intact.npy")
damaged_image = np.load("damaged.npy")
fig, ax = plt.subplots()

# # intact
# resize_intact_image = intact_image[100:, 140: 1140, :]
# # print(resize_intact_image.shape)
# ax.imshow(resize_intact_image)
# ax.axis("off")
# # plt.show()
# plt.savefig("intact_robot.pdf", dpi=500)


# damage
resize_damaged_image = damaged_image[100:, 140: 1140, :]
ax.imshow(resize_damaged_image)
ax.axis("off")
# plt.show()
plt.savefig("damaged_robot.pdf", dpi=500)


