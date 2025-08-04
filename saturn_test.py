#!/usr/bin/env python3
import subprocess

blender_path = "/Applications/Blender 4.5.app/Contents/MacOS/Blender"
script_content = '''
import bpy
print("🪐 Creating Saturn...")
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.mesh.primitive_uv_sphere_add(radius=3)
print("✅ Done!")
'''

with open("/tmp/saturn.py", "w") as f:
    f.write(script_content)

subprocess.run([blender_path, "--python", "/tmp/saturn.py"])
