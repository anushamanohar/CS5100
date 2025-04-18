# generate_urdfs.py

import os
import re

urdf_directory = os.path.join(os.path.dirname(__file__), "urdf_files")
os.makedirs(urdf_directory, exist_ok=True)

#   This function generates a URDF file for a robotic arm consisting of:
#   - A base
#   - Three revolute arm joints
#   - A wrist with pitch and roll
#   - A fixed gripper palm
#   - Two prismatic finger joints
def create_rigid_robotic_arm_urdf():
    urdf_dir = os.path.join(os.path.dirname(__file__), "urdf_files")
    os.makedirs(urdf_dir, exist_ok=True)
    file_path = os.path.join(urdf_dir, "robotic_arm.urdf")

    urdf_content = """<?xml version="1.0"?>
<robot name="robotic_arm">

    <!-- Base -->
    <link name="base">
        <visual>
            <geometry><cylinder radius="0.12" length="0.05"/></geometry>
        </visual>
        <collision>
            <geometry><cylinder radius="0.12" length="0.05"/></geometry>
        </collision>
        <inertial>
            <mass value="5"/>
            <inertia ixx="0.2" iyy="0.2" izz="0.2" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>

    <!-- Arm Base -->
    <link name="arm_base">
        <visual>
            <geometry><cylinder radius="0.1" length="0.1"/></geometry>
        </visual>
        <collision>
            <geometry><cylinder radius="0.1" length="0.1"/></geometry>
        </collision>
        <inertial>
            <mass value="3"/>
            <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>

    <joint name="base_to_arm" type="revolute">
        <parent link="base"/>
        <child link="arm_base"/>
        <origin xyz="0 0 0.1"/>
        <axis xyz="0 1 0"/>
        <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
    </joint>

    <!-- Arm Link 1 -->
    <link name="arm_link1">
        <visual>
            <geometry><box size="0.1 0.1 0.4"/></geometry>
        </visual>
        <collision>
            <geometry><box size="0.1 0.1 0.4"/></geometry>
        </collision>
        <inertial>
            <mass value="2"/>
            <inertia ixx="0.05" iyy="0.05" izz="0.05" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>

    <joint name="arm_joint1" type="revolute">
        <parent link="arm_base"/>
        <child link="arm_link1"/>
        <origin xyz="0 0 0.1"/>
        <axis xyz="0 1 0"/>
        <limit lower="-1.57" upper="1.57"/>
    </joint>

    <!-- Arm Link 2 -->
    <link name="arm_link2">
        <visual>
            <geometry><box size="0.1 0.1 0.4"/></geometry>
        </visual>
        <collision>
            <geometry><box size="0.1 0.1 0.4"/></geometry>
        </collision>
        <inertial>
            <mass value="1.5"/>
            <inertia ixx="0.04" iyy="0.04" izz="0.04" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>

    <joint name="arm_joint2" type="revolute">
        <parent link="arm_link1"/>
        <child link="arm_link2"/>
        <origin xyz="0 0 0.4"/>
        <axis xyz="0 1 0"/>
        <limit lower="-1.57" upper="1.57"/>
    </joint>

    <!-- Arm Link 3 -->
    <link name="arm_link3">
        <visual>
            <geometry><box size="0.1 0.1 0.3"/></geometry>
        </visual>
        <collision>
            <geometry><box size="0.1 0.1 0.3"/></geometry>
        </collision>
        <inertial>
            <mass value="1"/>
            <inertia ixx="0.03" iyy="0.03" izz="0.03" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>

    <joint name="arm_joint3" type="revolute">
        <parent link="arm_link2"/>
        <child link="arm_link3"/>
        <origin xyz="0 0 0.4"/>
        <axis xyz="0 1 0"/>
        <limit lower="-1.57" upper="1.57"/>
    </joint>

    <!-- Wrist Pitch Joint -->
    <link name="wrist_pitch">
        <visual>
            <geometry><cylinder radius="0.05" length="0.05"/></geometry>
        </visual>
        <collision>
            <geometry><cylinder radius="0.05" length="0.05"/></geometry>
        </collision>
        <inertial>
            <mass value="0.3"/>
            <inertia ixx="0.005" iyy="0.005" izz="0.005" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>

    <joint name="wrist_pitch_joint" type="revolute">
        <parent link="arm_link3"/>
        <child link="wrist_pitch"/>
        <origin xyz="0 0 0.3"/>
        <axis xyz="1 0 0"/>
        <limit lower="-1.57" upper="1.57" effort="10" velocity="0.5"/>
    </joint>

    <!-- Wrist Roll Joint -->
    <link name="wrist_roll">
        <visual>
            <geometry><cylinder radius="0.05" length="0.05"/></geometry>
        </visual>
        <collision>
            <geometry><cylinder radius="0.05" length="0.05"/></geometry>
        </collision>
        <inertial>
            <mass value="0.5"/>
            <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>

    <joint name="wrist_roll_joint" type="revolute">
        <parent link="wrist_pitch"/>
        <child link="wrist_roll"/>
        <origin xyz="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10" velocity="0.5"/>
    </joint>

    <!-- Gripper Palm -->
    <link name="gripper_palm">
        <visual>
            <geometry><box size="0.06 0.12 0.06"/></geometry>
        </visual>
        <collision>
            <geometry><box size="0.06 0.12 0.06"/></geometry>
        </collision>
        <inertial>
            <mass value="0.5"/>
            <inertia ixx="0.01" iyy="0.01" izz="0.01"/>
        </inertial>
    </link>

    <joint name="gripper_palm_joint" type="fixed">
        <parent link="wrist_roll"/>
        <child link="gripper_palm"/>
        <origin xyz="0 0 0.06"/>
    </joint>

    <!-- Left Finger -->
    <link name="left_finger">
        <visual>
            <geometry><box size="0.04 0.12 0.04"/></geometry>
        </visual>
        <collision>
            <geometry><box size="0.04 0.12 0.04"/></geometry>
        </collision>
        <inertial>
            <mass value="0.1"/>
            <inertia ixx="0.01" iyy="0.01" izz="0.01"/>
        </inertial>
    </link>

    <joint name="left_finger_joint" type="prismatic">
        <parent link="gripper_palm"/>
        <child link="left_finger"/>
        <origin xyz="0.02 0.035 0.02" rpy="0 0 0"/>   <!-- left -->
        <axis xyz="1 0 0"/>
        <limit lower="0" upper="0.05" effort="10" velocity="1.0"/>
    </joint>

    <!-- Right Finger -->
    <link name="right_finger">
        <visual>
            <geometry><box size="0.04 0.12 0.04"/></geometry>
        </visual>
        <collision>
            <geometry><box size="0.04 0.12 0.04"/></geometry>
        </collision>
        <inertial>
            <mass value="0.1"/>
            <inertia ixx="0.01" iyy="0.01" izz="0.01"/>
        </inertial>
    </link>

    <joint name="right_finger_joint" type="prismatic">
        <parent link="gripper_palm"/>
        <child link="right_finger"/>
        <origin xyz="-0.02 -0.035 0.02" rpy="0 0 0"/> <!-- right -->
        <axis xyz="1 0 0"/>
        <limit lower="0" upper="0.05" effort="10" velocity="1.0"/>
    </joint>

</robot>
"""
    with open(file_path, "w") as f:
        f.write(urdf_content)
    print(f"Robotic arm URDF with gripper saved at {file_path}")
    return file_path


robotic_arm_urdf = create_rigid_robotic_arm_urdf()

#   Generates a URDF file for a simple rectangular table composed of:
#   - A brown tabletop with four vertical legs connected with fixed joints
#   - The tabletop has dimensions 1m x 1m x 0.1m and a brown material color defined
#     using RGBA values
#   - Each leg is 0.1m x 0.1m x 0.6m and placed at a corner of the tabletop
#   - Legs are connected using fixed joints, so the table is static in simulation
def create_table_urdf():
    file_path = os.path.join(urdf_directory, "table.urdf")

    urdf_content = """<?xml version="1.0"?>
<robot name="table">
    
    <!-- Tabletop -->
    <link name="tabletop">
        <visual>
            <geometry><box size="1 1 0.1"/></geometry>
            <material name="brown">
                <color rgba="0.545 0.271 0.075 1"/>
            </material>
        </visual>
        <collision><geometry><box size="1 1 0.1"/></geometry></collision>
        <inertial><mass value="50"/><inertia ixx="1" iyy="1" izz="1"/></inertial>
    </link>

    <!-- Legs -->
    <link name="leg1">
        <visual><geometry><box size="0.1 0.1 0.6"/></geometry></visual>
        <collision><geometry><box size="0.1 0.1 0.6"/></geometry></collision>
    </link>
    <joint name="leg1_joint" type="fixed">
        <parent link="tabletop"/>
        <child link="leg1"/>
        <origin xyz="-0.45 -0.45 -0.35"/>
    </joint>

    <link name="leg2">
        <visual><geometry><box size="0.1 0.1 0.6"/></geometry></visual>
        <collision><geometry><box size="0.1 0.1 0.6"/></geometry></collision>
    </link>
    <joint name="leg2_joint" type="fixed">
        <parent link="tabletop"/>
        <child link="leg2"/>
        <origin xyz="0.45 -0.45 -0.35"/>
    </joint>

    <link name="leg3">
        <visual><geometry><box size="0.1 0.1 0.6"/></geometry></visual>
        <collision><geometry><box size="0.1 0.1 0.6"/></geometry></collision>
    </link>
    <joint name="leg3_joint" type="fixed">
        <parent link="tabletop"/>
        <child link="leg3"/>
        <origin xyz="-0.45 0.45 -0.35"/>
    </joint>

    <link name="leg4">
        <visual><geometry><box size="0.1 0.1 0.6"/></geometry></visual>
        <collision><geometry><box size="0.1 0.1 0.6"/></geometry></collision>
    </link>
    <joint name="leg4_joint" type="fixed">
        <parent link="tabletop"/>
        <child link="leg4"/>
        <origin xyz="0.45 0.45 -0.35"/>
    </joint>

</robot>
"""

    with open(file_path, "w") as f:
        f.write(urdf_content)

    print(f"Table URDF successfully created at {file_path}")
    return file_path

table_urdf = create_table_urdf()



#   This function generates a URDF file for objects placed on the table — either a box or
#   a sphere — with specified size and mass. These objects can be used for testing 
#   grasping and manipulation tasks in a simulated environment.
#
# Parameters:
#   - name:   The name of the object and the output URDF file (e.g., "box", "sphere")
#   - shape:  The shape of the object, either "box" or "sphere"
#   - size:   Size of the object (edge length for box, radius for sphere)
#   - mass:   Mass of the object in kilograms
def create_object_urdf(name, shape, size, mass):
    file_path = os.path.join(urdf_directory, f"{name}.urdf")
    
    if shape == "box":
        geometry = f"<box size=\"{size} {size} {size}\"/>"
    elif shape == "sphere":
        geometry = f"<sphere radius=\"{size}\"/>"
    else:
        raise ValueError("Unsupported shape. Use 'box' or 'sphere'.")

    urdf_content = f"""<?xml version="1.0"?>
<robot name="{name}">
    <link name="{name}">
        <visual><geometry>{geometry}</geometry></visual>
        <collision><geometry>{geometry}</geometry></collision>
        <inertial><mass value="{mass}"/><inertia ixx="1" iyy="1" izz="1"/></inertial>
    </link>
</robot>
"""

    with open(file_path, "w") as f:
        f.write(urdf_content)

    print(f"{name.capitalize()} URDF successfully created at {file_path}")
    return file_path

box_urdf = create_object_urdf("box", "box", 0.08, 1)
sphere_urdf = create_object_urdf("sphere", "sphere", 0.05, 0.5)

# # Function to scan for ArUco markers
parcel_ids_file = os.path.join(urdf_directory, "parcel_ids.txt")  # TXT file to store parcel IDs

# Function to scan for ArUco markers
def assign_aruco_marker_ids():
    """Scans the directory for ArUco marker images, extracts IDs, and stores them in a dictionary."""
    marker_id_dict = {}
    pattern = re.compile(r'aruco_marker_(\d+)\.png')

    for filename in os.listdir(urdf_directory):
        match = pattern.match(filename)
        if match:
            marker_id = int(match.group(1))
            marker_id_dict[marker_id] = filename  # Store marker ID with filename

    return marker_id_dict

# Function to create a parcel with an ArUco marker
def create_parcel_with_aruco_urdf(marker_id, marker_filename):
    """Creates a URDF for a parcel with an ArUco marker texture."""
    file_path = os.path.join(urdf_directory, f"parcel_{marker_id}.urdf")

    urdf_content = f"""<?xml version="1.0"?>
<robot name="parcel_{marker_id}">
    <link name="parcel">
        <visual>
            <origin xyz="0 0 0.05" rpy="0 0 0"/>
            <geometry>
                <box size="0.1 0.1 0.1"/>
            </geometry>
            <material name="aruco_marker_{marker_id}">
                
                <color rgba="1 1 1 1"/>

                <texture filename="{marker_filename.replace('.png', '.bmp')}"/>
            </material>
        </visual>
        <collision>
            <geometry>
                <box size="0.1 0.1 0.1"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="0.5"/>
            <inertia ixx="0.001" iyy="0.001" izz="0.001"/>
        </inertial>
    </link>
</robot>
"""

    with open(file_path, "w") as f:
        f.write(urdf_content)

    print(f"Parcel URDF with ArUco marker {marker_id} created at {file_path}")
    return file_path

# Scan for ArUco markers
aruco_markers = assign_aruco_marker_ids()

# Generate URDFs for detected markers
generated_urdfs = {}
parcel_ids = []  # List to store parcel IDs

for marker_id, marker_filename in aruco_markers.items():
    urdf_path = create_parcel_with_aruco_urdf(marker_id, marker_filename)
    generated_urdfs[marker_id] = urdf_path
    parcel_ids.append(str(marker_id))  # Store only marker IDs as strings

# Save Parcel IDs to a text file
parcel_ids_file = os.path.join(urdf_directory, "parcel_ids.txt")
with open(parcel_ids_file, "w") as f:
    f.write("\n".join(parcel_ids))

print(f"Parcel IDs saved to {parcel_ids_file}")


#   This function creates a URDF file for a rectangular tray, which includes a flat
#   base and four raised walls (front, back, left, right). The tray is used as pickup zone and
#   drop zone 
# Tray Description:
#   - Base size: 0.6m x 0.4m x 0.02m
#   - Side wall height: 0.05m
#   - Side wall thickness: 0.02m
#   - All walls are attached using fixed joints to ensure stability
#   - Base is shaded gray for better visibility
def create_tray_urdf():
    """Creates a URDF file for a tray with raised edges."""
    file_path = os.path.join(urdf_directory, "tray.urdf")

    urdf_content = """<?xml version="1.0"?>
<robot name="tray">

    <!-- Tray Base -->
    <link name="tray_base">
        <visual>
            <geometry>
                <box size="0.6 0.4 0.02"/>
            </geometry>
            <material name="gray">
                <color rgba="0.5 0.5 0.5 1"/>
            </material>
        </visual>
        <collision>
            <geometry>
                <box size="0.6 0.4 0.02"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="2"/>
            <inertia ixx="0.02" iyy="0.02" izz="0.02"/>
        </inertial>
    </link>

    <!-- Tray Side Walls -->
    <link name="tray_wall_front">
        <visual>
            <geometry>
                <box size="0.6 0.02 0.05"/>
            </geometry>
        </visual>
        <collision>
            <geometry>
                <box size="0.6 0.02 0.05"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="0.1"/>
            <inertia ixx="0.01" iyy="0.01" izz="0.01"/>
        </inertial>
    </link>
    <joint name="wall_front_joint" type="fixed">
        <parent link="tray_base"/>
        <child link="tray_wall_front"/>
        <origin xyz="0 0.21 0.035"/>
    </joint>

    <link name="tray_wall_back">
        <visual>
            <geometry>
                <box size="0.6 0.02 0.05"/>
            </geometry>
        </visual>
        <collision>
            <geometry>
                <box size="0.6 0.02 0.05"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="0.1"/>
            <inertia ixx="0.01" iyy="0.01" izz="0.01"/>
        </inertial>
    </link>
    <joint name="wall_back_joint" type="fixed">
        <parent link="tray_base"/>
        <child link="tray_wall_back"/>
        <origin xyz="0 -0.21 0.035"/>
    </joint>

    <link name="tray_wall_left">
        <visual>
            <geometry>
                <box size="0.02 0.4 0.05"/>
            </geometry>
        </visual>
        <collision>
            <geometry>
                <box size="0.02 0.4 0.05"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="0.1"/>
            <inertia ixx="0.01" iyy="0.01" izz="0.01"/>
        </inertial>
    </link>
    <joint name="wall_left_joint" type="fixed">
        <parent link="tray_base"/>
        <child link="tray_wall_left"/>
        <origin xyz="-0.29 0 0.035"/>
    </joint>

    <link name="tray_wall_right">
        <visual>
            <geometry>
                <box size="0.02 0.4 0.05"/>
            </geometry>
        </visual>
        <collision>
            <geometry>
                <box size="0.02 0.4 0.05"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="0.1"/>
            <inertia ixx="0.01" iyy="0.01" izz="0.01"/>
        </inertial>
    </link>
    <joint name="wall_right_joint" type="fixed">
        <parent link="tray_base"/>
        <child link="tray_wall_right"/>
        <origin xyz="0.29 0 0.035"/>
    </joint>

</robot>
"""

    with open(file_path, "w") as f:
        f.write(urdf_content)

    print(f"Tray URDF successfully created at {file_path}")
    return file_path
tray_urdf = create_tray_urdf()

