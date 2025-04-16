import trimesh
from trimesh.creation import box, cylinder
import numpy as np
import os

output_dir = "urdf_files/custom_claw_parts"
os.makedirs(output_dir, exist_ok=True)

def save(mesh, name):
    path = os.path.join(output_dir, f"{name}.stl")
    mesh.export(path)
    print(f"Saved: {path}")

# 1. Base block (palm of the claw)
base = box(extents=[0.08, 0.06, 0.02])
save(base, "claw_base")

# 2. Curved claw finger
def make_claw_finger():
    parts = []
    segment_len = 0.035
    segment_rad = 0.008
    segment = cylinder(radius=segment_rad, height=segment_len, sections=32)
    segment.apply_translation([0, 0, segment_len / 2])
    for i in range(6):
        seg = segment.copy()
        angle = i * 15  # tighter curve
        transform = trimesh.transformations.rotation_matrix(
            np.radians(angle), [0, 1, 0], point=[0, 0, 0]
        )
        seg.apply_transform(transform)
        seg.apply_translation([segment_len * i, 0, 0])
        parts.append(seg)
    return trimesh.util.concatenate(parts)

claw_left = make_claw_finger()
save(claw_left, "claw_left")

# 3. Mirror it to get the right claw
claw_right = claw_left.copy()
claw_right.apply_scale([-1, 1, 1])  # Mirror across X
save(claw_right, "claw_right")

# Optional: pins
pin = cylinder(radius=0.005, height=0.02)
save(pin, "pin")

print("✅ Custom claw components exported.")

mesh=trimesh.load("urdf_files/custom_claw_parts/claw_base.stl")
mesh.show(viewer='gl')  # or 'windowed'

