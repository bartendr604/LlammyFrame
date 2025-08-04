#!/usr/bin/env python3
import subprocess
import time

print("🎬 Starting visual session...")

blender_path = "/Applications/Blender 4.5.app/Contents/MacOS/Blender"
script_content = '''
import bpy
import time

print("🎨 Visual AI Session Starting...")

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Create something visual
bpy.ops.mesh.primitive_uv_sphere_add(radius=2, location=(0,0,0))
sphere = bpy.context.active_object
sphere.name = "AI_Creation"

# Add material
material = bpy.data.materials.new(name="AI_Material")
material.use_nodes = True
material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.8, 0.2, 0.9, 1.0)
sphere.data.materials.append(material)

print("✅ AI creation complete - displaying for 15 seconds...")
time.sleep(15)
print("🏁 Visual session complete")
'''

with open("/tmp/visual_session.py", "w") as f:
    f.write(script_content)

print("🚀 Launching Blender...")
subprocess.run([blender_path, "--python", "/tmp/visual_session.py"])
print("✅ Visual session complete!")
