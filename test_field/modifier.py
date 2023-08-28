import os
import re

density_pattern = re.compile(r"\$\$density\$\$")
friction_pattern = re.compile(r"\$\$friction\$\$")

xml_path = r"/home/runjun/python_project/pybullet/lib/python3.8/site-packages/pybullet_data/mjcf"


def modify(density, friction):
    density = "{:.2f}".format(density)
    friction = "{:.2f}".format(friction)

    with open(os.path.join(xml_path, "ant(mod).xml"), "r") as file:
        content = file.read()
    
    content = density_pattern.sub(density, content)
    content = friction_pattern.sub(friction, content)

    with open(os.path.join(xml_path, "custom_ant.xml"), "w") as file:
        file.write(content)



