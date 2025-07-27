# VisionLoraModule.py - Vision LoRA Integration for Llammy Framework
# Auto-discovered by the Master Coordinator

import os
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Tuple, Optional
import bpy

class VisionLoraService:
    """Vision LoRA Service - Auto-discovered by Llammy Master System"""
    
    def __init__(self):
        self.initialized = False
        self.addon_dir = Path(__file__).parent
        self.vision_script = self.addon_dir / "Scripts" / "VisionLoraIntegration.py"
        self.setup_script = self.addon_dir / "setup.sh"
        self.venv_path = self.addon_dir / ".venv"
        self.setup_completed = False
        
        print("👁️ Vision LoRA Service initialized")
    
    def initialize(self):
        """Initialize vision system - called by Master Coordinator"""
        try:
            # Check if vision environment exists
            if not self.venv_path.exists():
                print("🔧 Setting up vision environment...")
                success, message = self.setup_vision_environment()
                if not success:
                    print(f"⚠️ Vision setup failed: {message}")
                    return False
            
            # Verify vision script exists
            if not self.vision_script.exists():
                print(f"❌ Vision script not found: {self.vision_script}")
                return False
            
            self.initialized = True
            print("✅ Vision LoRA Service ready")
            return True
            
        except Exception as e:
            print(f"❌ Vision initialization failed: {e}")
            return False
    
    def setup_vision_environment(self) -> Tuple[bool, str]:
        """Setup the vision LoRA environment"""
        try:
            if not self.setup_script.exists():
                return False, "setup.sh not found - copy your LoRA package files"
            
            # Make setup script executable and run it
            os.chmod(str(self.setup_script), 0o755)
            result = subprocess.run([str(self.setup_script)], 
                                  cwd=str(self.addon_dir),
                                  capture_output=True, 
                                  text=True)
            
            if result.returncode == 0:
                self.setup_completed = True
                return True, "Vision environment setup complete"
            else:
                return False, f"Setup failed: {result.stderr}"
                
        except Exception as e:
            return False, f"Setup error: {e}"
    
    def analyze_reference_image(self, image_path: str) -> str:
        """Analyze reference image for creative context"""
        if not self.initialized:
            return "Vision system not initialized"
        
        try:
            # Run vision analysis
            python_exe = self.venv_path / "bin" / "python3"
            result = subprocess.run([
                str(python_exe), 
                str(self.vision_script), 
                image_path
            ], capture_output=True, text=True, cwd=str(self.addon_dir))
            
            if result.returncode == 0:
                vision_output = result.stdout.strip()
                return self._format_for_creative_ai(vision_output, image_path)
            else:
                return f"Vision analysis failed: {result.stderr}"
                
        except Exception as e:
            return f"Vision error: {e}"
    
    def analyze_blender_scene(self) -> str:
        """Analyze current Blender scene"""
        if not self.initialized:
            return "Vision system not initialized"
        
        try:
            # Capture viewport
            viewport_image = self.capture_viewport()
            if not viewport_image:
                return "Failed to capture viewport"
            
            # Analyze the captured image
            scene_analysis = self.analyze_reference_image(viewport_image)
            
            # Clean up temp file
            if os.path.exists(viewport_image):
                os.remove(viewport_image)
            
            return self._format_for_technical_ai(scene_analysis)
            
        except Exception as e:
            return f"Scene analysis error: {e}"
    
    def capture_viewport(self) -> Optional[str]:
        """Capture current Blender viewport to temp file"""
        try:
            # Create temp file
            temp_file = tempfile.mktemp(suffix=".png")
            
            # Capture viewport
            bpy.ops.render.opengl(write_still=True)
            
            # Save render result
            if 'Render Result' in bpy.data.images:
                bpy.data.images['Render Result'].save_render(filepath=temp_file)
                return temp_file
            else:
                return None
                
        except Exception as e:
            print(f"Viewport capture failed: {e}")
            return None
    
    def _format_for_creative_ai(self, vision_output: str, image_path: str) -> str:
        """Format vision analysis for creative AI (Llammy)"""
        image_name = os.path.basename(image_path)
        return f"""VISUAL REFERENCE ANALYSIS for {image_name}:

{vision_output}

CREATIVE DIRECTION NOTES:
- Use this visual analysis to understand the mood, style, and artistic elements
- Consider the composition, lighting, and color palette when generating creative concepts
- Let this visual context inspire the creative direction and artistic choices
- Adapt the technical implementation to match this visual aesthetic"""

    def _format_for_technical_ai(self, vision_output: str) -> str:
        """Format scene analysis for technical AI (Qwen)"""
        return f"""CURRENT BLENDER SCENE ANALYSIS:

{vision_output}

TECHNICAL IMPLEMENTATION CONTEXT:
- This shows what currently exists in the Blender scene
- Use this context to understand existing objects, materials, and setup
- Ensure new code integrates well with the current scene structure
- Consider existing naming conventions and object relationships"""

    def get_status(self) -> dict:
        """Get vision system status"""
        return {
            "initialized": self.initialized,
            "setup_completed": self.setup_completed,
            "vision_script_exists": self.vision_script.exists(),
            "venv_exists": self.venv_path.exists(),
            "setup_script_exists": self.setup_script.exists()
        }

# Global instance for the Master Coordinator to find
vision_lora_service = VisionLoraService()

# Blender operators for vision functionality
class LLAMMY_OT_UploadReference(bpy.types.Operator):
    """Upload reference images for vision analysis"""
    bl_idname = "llammy.upload_reference"
    bl_label = "Upload Reference Images"
    bl_description = "Select reference images for AI vision analysis"
    
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_image: bpy.props.BoolProperty(default=True, options={'HIDDEN'})
    
    def execute(self, context):
        if not self.filepath:
            self.report({'WARNING'}, "No image selected")
            return {'CANCELLED'}
        
        # Store reference images in scene
        if not hasattr(context.scene, 'llammy_reference_images'):
            context.scene.llammy_reference_images = ""
        
        # Add to list (semicolon separated)
        current_images = context.scene.llammy_reference_images
        if current_images:
            context.scene.llammy_reference_images = f"{current_images};{self.filepath}"
        else:
            context.scene.llammy_reference_images = self.filepath
        
        # Test vision analysis
        global vision_lora_service
        if vision_lora_service.initialized:
            analysis = vision_lora_service.analyze_reference_image(self.filepath)
            print(f"Vision Analysis Preview: {analysis[:200]}...")
            self.report({'INFO'}, f"✅ Reference image added and analyzed")
        else:
            self.report({'WARNING'}, "⚠️ Reference added but vision system not ready")
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class LLAMMY_OT_AnalyzeScene(bpy.types.Operator):
    """Analyze current Blender scene with vision"""
    bl_idname = "llammy.analyze_scene" 
    bl_label = "Analyze Current Scene"
    bl_description = "Analyze current Blender scene with vision AI"
    
    def execute(self, context):
        global vision_lora_service
        
        if not vision_lora_service.initialized:
            self.report({'ERROR'}, "Vision system not initialized")
            return {'CANCELLED'}
        
        # Analyze current scene
        scene_analysis = vision_lora_service.analyze_blender_scene()
        
        # Store in scene for use by other systems
        context.scene.llammy_scene_analysis = scene_analysis
        
        print(f"Scene Analysis: {scene_analysis}")
        self.report({'INFO'}, "✅ Scene analyzed with vision AI")
        
        return {'FINISHED'}

class LLAMMY_OT_SetupVision(bpy.types.Operator):
    """Setup vision LoRA environment"""
    bl_idname = "llammy.setup_vision"
    bl_label = "Setup Vision System"
    bl_description = "Install and setup the vision LoRA environment"
    
    def execute(self, context):
        global vision_lora_service
        
        self.report({'INFO'}, "Setting up vision environment... (this may take a minute)")
        
        success, message = vision_lora_service.setup_vision_environment()
        
        if success:
            vision_lora_service.initialize()
            self.report({'INFO'}, f"✅ {message}")
        else:
            self.report({'ERROR'}, f"❌ {message}")
        
        return {'FINISHED'}

# Operators list for registration by Master Coordinator
VISION_OPERATORS = [
    LLAMMY_OT_UploadReference,
    LLAMMY_OT_AnalyzeScene,
    LLAMMY_OT_SetupVision,
]

def register_vision_operators():
    """Register vision operators"""
    for cls in VISION_OPERATORS:
        try:
            bpy.utils.register_class(cls)
        except:
            pass  # Already registered

def unregister_vision_operators():
    """Unregister vision operators"""
    for cls in reversed(VISION_OPERATORS):
        try:
            bpy.utils.unregister_class(cls)
        except:
            pass  # Not registered

# Auto-register when module is imported
register_vision_operators()

print("👁️ VisionLoraModule loaded and ready for auto-discovery")

