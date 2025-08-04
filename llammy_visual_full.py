#!/usr/bin/env python3
import subprocess
import time
import random
import sys
from datetime import datetime

def run_single_session():
    print("🎬 Visual session starting...")
    
    # Test Blender path
    blender_path = "/Applications/Blender 4.5.app/Contents/MacOS/Blender"
    if subprocess.run([blender_path, "--version"], capture_output=True).returncode == 0:
        print("✅ Blender found!")
        
        # Create simple Blender script
        script = '''
import bpy
print("🪐 Creating Saturn system...")
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.mesh.primitive_uv_sphere_add(radius=3, location=(0,0,0))
print("✅ Saturn created!")
'''
        
        with open("/tmp/saturn_script.py", "w") as f:
            f.write(script)
        
        # Run Blender
        subprocess.run([blender_path, "--python", "/tmp/saturn_script.py"])
        print("✅ Saturn session complete!")
    else:
        print("❌ Blender not found")

if __name__ == "__main__":
    if "--single-session" in sys.argv:
        run_single_session()
    else:
        print("🚀 Full mode - use --single-session for test")
