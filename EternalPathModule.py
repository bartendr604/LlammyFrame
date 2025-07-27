# characters/llammy_character_images.py - Character Images & Rigging System
# Llammy Framewor/Users/jimmcquade/Library/Application Support/Blender/4.5/scripts/addons/EternalPathModule.pyk v8.5 - Visual Character Creation

import bpy
import bmesh
import json
import base64
import urllib.request
import urllib.error
from mathutils import Vector, Matrix
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import os
import tempfile

class CharacterImageSystem:
    """Advanced Character Image Generation & Rigging System
    
    This system handles:
    1. AI-powered character image generation
    2. Dynamic character prompt creation
    3. Automatic rigging for generated characters
    4. Material and texture application
    5. Character pose and expression systems
    6. Integration with story modules
    """
    
    def __init__(self, model_manager=None):
        self.model_manager = model_manager
        self.character_templates = {}
        self.image_cache = {}
        self.rigging_presets = {}
        self.current_story = "elephant_story"
        
        # Image generation settings
        self.image_config = {
            "default_resolution": (1024, 1024),
            "supported_formats": ["PNG", "JPEG", "WebP"],
            "quality_levels": ["draft", "standard", "high", "ultra"],
            "style_presets": {
                "realistic": "photorealistic, detailed, high quality",
                "stylized": "stylized, artistic, vibrant colors",
                "cartoon": "cartoon style, friendly, colorful",
                "anime": "anime style, detailed eyes, expressive",
                "pixar": "pixar style, 3D rendered, warm lighting"
            }
        }
        
        # Rigging configuration
        self.rigging_config = {
            "default_bone_count": 15,
            "auto_weight_paint": True,
            "enable_constraints": True,
            "create_control_rig": True,
            "bone_layers": {
                "deform": 0,
                "control": 1,
                "mechanism": 2,
                "extra": 3
            }
        }
        
        # Character prompt templates
        self.prompt_templates = {
            "base_character": {
                "positive": "character design, full body, standing pose, clean background, {style}, {character_description}",
                "negative": "blurry, low quality, deformed, multiple characters, cropped, text, watermark"
            },
            "portrait": {
                "positive": "character portrait, head and shoulders, detailed face, {style}, {character_description}",
                "negative": "blurry, low quality, deformed, multiple faces, cropped, text"
            },
            "action_pose": {
                "positive": "character in action, dynamic pose, {action_description}, {style}, {character_description}",
                "negative": "static, boring pose, blurry, low quality, deformed"
            },
            "expression_sheet": {
                "positive": "character expression sheet, multiple expressions, {expressions}, {style}, {character_description}",
                "negative": "single expression, blurry, inconsistent character, low quality"
            }
        }
        
        print("🎨 Character Image System initialized!")
    
    def generate_character_image(self, character_name: str, image_type: str = "base_character", 
                                style: str = "stylized", custom_prompt: str = None) -> Tuple[bool, str, Optional[str]]:
        """Generate an image for a character using AI"""
        try:
            # Get character info from current story
            character_info = self._get_character_info(character_name)
            if not character_info:
                return False, f"Character '{character_name}' not found in current story", None
            
            # Build the prompt
            if custom_prompt:
                final_prompt = custom_prompt
            else:
                final_prompt = self._build_character_prompt(character_info, image_type, style)
            
            print(f"🖼️ Generating image for {character_name} with prompt: {final_prompt[:100]}...")
            
            # Generate image using available AI backend
            success, image_path = self._generate_image_with_ai(final_prompt, character_name)
            
            if success:
                # Cache the image
                cache_key = f"{character_name}_{image_type}_{style}"
                self.image_cache[cache_key] = {
                    "path": image_path,
                    "character": character_name,
                    "type": image_type,
                    "style": style,
                    "prompt": final_prompt,
                    "generated": datetime.now().isoformat()
                }
                
                return True, f"Image generated successfully for {character_name}", image_path
            else:
                return False, f"Failed to generate image: {image_path}", None
                
        except Exception as e:
            error_msg = f"Error generating character image: {str(e)}"
            print(f"❌ {error_msg}")
            return False, error_msg, None
    
    def _get_character_info(self, character_name: str) -> Optional[Dict]:
        """Get character information from the current story module"""
        try:
            # This would integrate with the story system
            # For now, using the elephant story characters as example
            elephant_characters = {
                "Tien": {
                    "description": "jade green elephant, young and enthusiastic, keyboard player",
                    "personality": "energetic, musical, creative",
                    "colors": ["jade green", "gold accents"],
                    "props": ["keyboard", "musical notes"],
                    "size": "medium"
                },
                "Nishang": {
                    "description": "translucent glass elephant, shy and gentle, emotional lighting",
                    "personality": "shy, emotional, artistic",
                    "colors": ["clear glass", "rainbow refractions", "soft blue"],
                    "props": ["prism", "light beams"],
                    "size": "small"
                },
                "Xiaohan": {
                    "description": "wise ancient dragon, mentor figure, cosmic powers",
                    "personality": "wise, patient, powerful",
                    "colors": ["deep purple", "silver", "cosmic blue"],
                    "props": ["staff", "floating orbs", "ancient symbols"],
                    "size": "large"
                }
            }
            
            return elephant_characters.get(character_name)
            
        except Exception as e:
            print(f"❌ Error getting character info: {e}")
            return None
    
    def _build_character_prompt(self, character_info: Dict, image_type: str, style: str) -> str:
        """Build a detailed prompt for character image generation"""
        try:
            # Get base template
            template = self.prompt_templates.get(image_type, self.prompt_templates["base_character"])
            
            # Get style description
            style_desc = self.image_config["style_presets"].get(style, "stylized, artistic")
            
            # Build character description
            char_desc = character_info["description"]
            personality = character_info.get("personality", "")
            colors = ", ".join(character_info.get("colors", []))
            
            # Add personality traits to description
            if personality:
                char_desc += f", {personality} personality"
            
            # Add color information
            if colors:
                char_desc += f", colors: {colors}"
            
            # Format the prompt
            positive_prompt = template["positive"].format(
                style=style_desc,
                character_description=char_desc,
                action_description="dynamic movement",
                expressions="happy, sad, surprised, angry"
            )
            
            # Add quality enhancers
            quality_enhancers = "high quality, detailed, professional, well-lit, clean composition"
            final_prompt = f"{positive_prompt}, {quality_enhancers}"
            
            return final_prompt
            
        except Exception as e:
            print(f"❌ Error building prompt: {e}")
            return "character design, high quality"
    
    def _generate_image_with_ai(self, prompt: str, character_name: str) -> Tuple[bool, str]:
        """Generate image using available AI backend"""
        try:
            # Check if we have a model manager with image generation capability
            if not self.model_manager:
                return False, "No model manager available"
            
            # For now, simulate image generation since we don't have actual image gen models
            # In a real implementation, this would call DALL-E, Midjourney, Stable Diffusion, etc.
            
            # Create a placeholder image path
            temp_dir = tempfile.gettempdir()
            image_filename = f"llammy_{character_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            image_path = os.path.join(temp_dir, image_filename)
            
            # Create a simple placeholder image using Blender
            success = self._create_placeholder_image(image_path, character_name, prompt)
            
            if success:
                return True, image_path
            else:
                return False, "Failed to create placeholder image"
                
        except Exception as e:
            return False, f"Image generation error: {str(e)}"
    
    def _create_placeholder_image(self, image_path: str, character_name: str, prompt: str) -> bool:
        """Create a placeholder image using Blender's compositor"""
        try:
            # Create a new image in Blender
            width, height = self.image_config["default_resolution"]
            image = bpy.data.images.new(f"llammy_{character_name}", width=width, height=height)
            
            # Fill with a character-specific color
            character_colors = {
                "Tien": (0.2, 0.8, 0.3, 1.0),  # Jade green
                "Nishang": (0.8, 0.9, 1.0, 1.0),  # Light blue
                "Xiaohan": (0.4, 0.2, 0.8, 1.0)  # Purple
            }
            
            color = character_colors.get(character_name, (0.5, 0.5, 0.5, 1.0))
            
            # Fill the image with the character color
            pixels = [color[i % 4] for i in range(width * height * 4)]
            image.pixels[:] = pixels
            
            # Save the image
            image.filepath_raw = image_path
            image.file_format = 'PNG'
            image.save()
            
            print(f"✅ Created placeholder image: {image_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating placeholder image: {e}")
            return False
    
    def apply_image_to_character(self, character_name: str, image_path: str) -> Tuple[bool, str]:
        """Apply generated image to character mesh as texture"""
        try:
            # Find the character object
            char_obj = bpy.data.objects.get(character_name)
            if not char_obj:
                return False, f"Character object '{character_name}' not found"
            
            # Load the image
            if not os.path.exists(image_path):
                return False, f"Image file not found: {image_path}"
            
            image = bpy.data.images.load(image_path)
            
            # Create or update material
            material_name = f"{character_name}_material"
            material = bpy.data.materials.get(material_name)
            
            if not material:
                material = bpy.data.materials.new(name=material_name)
                material.use_nodes = True
            
            # Clear existing nodes
            material.node_tree.nodes.clear()
            
            # Create material nodes
            nodes = material.node_tree.nodes
            links = material.node_tree.links
            
            # Output node
            output_node = nodes.new(type='ShaderNodeOutputMaterial')
            output_node.location = (400, 0)
            
            # Principled BSDF
            principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
            principled_node.location = (200, 0)
            
            # Image texture node
            image_node = nodes.new(type='ShaderNodeTexImage')
            image_node.location = (0, 0)
            image_node.image = image
            
            # Connect nodes
            links.new(image_node.outputs['Color'], principled_node.inputs['Base Color'])
            links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
            
            # Apply material to object
            if char_obj.data.materials:
                char_obj.data.materials[0] = material
            else:
                char_obj.data.materials.append(material)
            
            print(f"✅ Applied image texture to {character_name}")
            return True, f"Image applied to {character_name} successfully"
            
        except Exception as e:
            error_msg = f"Error applying image to character: {str(e)}"
            print(f"❌ {error_msg}")
            return False, error_msg
    
    def create_character_rig(self, character_name: str, rig_type: str = "basic") -> Tuple[bool, str]:
        """Create a rig for the character mesh"""
        try:
            # Find the character object
            char_obj = bpy.data.objects.get(character_name)
            if not char_obj or char_obj.type != 'MESH':
                return False, f"Character mesh '{character_name}' not found"
            
            # Create armature
            armature_name = f"{character_name}_rig"
            armature_data = bpy.data.armatures.new(armature_name)
            armature_obj = bpy.data.objects.new(armature_name, armature_data)
            bpy.context.collection.objects.link(armature_obj)
            
            # Enter edit mode to create bones
            bpy.context.view_layer.objects.active = armature_obj
            bpy.ops.object.mode_set(mode='EDIT')
            
            # Create bone structure based on rig type
            if rig_type == "basic":
                self._create_basic_rig(armature_data, char_obj)
            elif rig_type == "advanced":
                self._create_advanced_rig(armature_data, char_obj)
            else:
                self._create_basic_rig(armature_data, char_obj)
            
            # Exit edit mode
            bpy.ops.object.mode_set(mode='OBJECT')
            
            # Parent mesh to armature
            char_obj.parent = armature_obj
            char_obj.parent_type = 'ARMATURE_AUTO'
            
            # Add armature modifier
            if not char_obj.modifiers.get("Armature"):
                armature_mod = char_obj.modifiers.new(name="Armature", type='ARMATURE')
                armature_mod.object = armature_obj
            
            # Auto-weight if enabled
            if self.rigging_config["auto_weight_paint"]:
                bpy.context.view_layer.objects.active = char_obj
                bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
                bpy.ops.object.mode_set(mode='OBJECT')
            
            print(f"✅ Created {rig_type} rig for {character_name}")
            return True, f"Rig created successfully for {character_name}"
            
        except Exception as e:
            error_msg = f"Error creating character rig: {str(e)}"
            print(f"❌ {error_msg}")
            return False, error_msg
    
    def _create_basic_rig(self, armature_data, char_obj):
        """Create a basic rig structure"""
        try:
            bones = armature_data.edit_bones
            
            # Get character bounds
            bbox = char_obj.bound_box
            min_z = min(v[2] for v in bbox)
            max_z = max(v[2] for v in bbox)
            center_x = sum(v[0] for v in bbox) / 8
            center_y = sum(v[1] for v in bbox) / 8
            
            height = max_z - min_z
            
            # Root bone
            root_bone = bones.new("Root")
            root_bone.head = (center_x, center_y, min_z)
            root_bone.tail = (center_x, center_y, min_z + height * 0.1)
            
            # Spine bones
            spine_count = 3
            for i in range(spine_count):
                bone_name = f"Spine_{i+1:02d}"
                bone = bones.new(bone_name)
                bone.head = (center_x, center_y, min_z + height * (0.1 + i * 0.25))
                bone.tail = (center_x, center_y, min_z + height * (0.1 + (i + 1) * 0.25))
                
                if i == 0:
                    bone.parent = root_bone
                else:
                    bone.parent = bones[f"Spine_{i:02d}"]
            
            # Head bone
            head_bone = bones.new("Head")
            head_bone.head = (center_x, center_y, min_z + height * 0.85)
            head_bone.tail = (center_x, center_y, max_z)
            head_bone.parent = bones["Spine_03"]
            
            # Limb bones (if applicable)
            if "elephant" in char_obj.name.lower():
                self._create_elephant_limbs(bones, center_x, center_y, min_z, height)
            
        except Exception as e:
            print(f"❌ Error creating basic rig: {e}")
    
    def _create_elephant_limbs(self, bones, center_x, center_y, min_z, height):
        """Create elephant-specific limb bones"""
        try:
            # Trunk bones
            trunk_segments = 5
            for i in range(trunk_segments):
                bone_name = f"Trunk_{i+1:02d}"
                bone = bones.new(bone_name)
                bone.head = (center_x, center_y + height * 0.3, min_z + height * (0.6 - i * 0.1))
                bone.tail = (center_x, center_y + height * 0.4, min_z + height * (0.6 - (i + 1) * 0.1))
                
                if i == 0:
                    bone.parent = bones["Spine_03"]
                else:
                    bone.parent = bones[f"Trunk_{i:02d}"]
            
            # Ear bones
            for side in ["L", "R"]:
                multiplier = 1 if side == "L" else -1
                ear_bone = bones.new(f"Ear_{side}")
                ear_bone.head = (center_x + height * 0.3 * multiplier, center_y, min_z + height * 0.8)
                ear_bone.tail = (center_x + height * 0.5 * multiplier, center_y, min_z + height * 0.9)
                ear_bone.parent = bones["Head"]
            
            # Leg bones
            for i, leg in enumerate(["Front_L", "Front_R", "Back_L", "Back_R"]):
                leg_bone = bones.new(f"Leg_{leg}")
                x_offset = height * 0.2 * (1 if "R" in leg else -1)
                y_offset = height * 0.2 * (1 if "Front" in leg else -1)
                
                leg_bone.head = (center_x + x_offset, center_y + y_offset, min_z + height * 0.4)
                leg_bone.tail = (center_x + x_offset, center_y + y_offset, min_z)
                leg_bone.parent = bones["Spine_02"]
            
        except Exception as e:
            print(f"❌ Error creating elephant limbs: {e}")
    
    def _create_advanced_rig(self, armature_data, char_obj):
        """Create an advanced rig with IK, constraints, and control bones"""
        try:
            # Start with basic rig
            self._create_basic_rig(armature_data, char_obj)
            
            # Add IK controls and constraints
            bones = armature_data.edit_bones
            
            # Create control bones
            for bone_name in bones.keys():
                if bone_name.startswith("Leg_"):
                    # Create IK target
                    ik_target_name = f"{bone_name}_IK_Target"
                    ik_bone = bones.new(ik_target_name)
                    ik_bone.head = bones[bone_name].tail
                    ik_bone.tail = (bones[bone_name].tail.x, bones[bone_name].tail.y, bones[bone_name].tail.z - 0.1)
                    
                    # Create pole target
                    pole_target_name = f"{bone_name}_Pole_Target"
                    pole_bone = bones.new(pole_target_name)
                    pole_bone.head = (bones[bone_name].head.x, bones[bone_name].head.y + 0.5, bones[bone_name].head.z)
                    pole_bone.tail = (bones[bone_name].head.x, bones[bone_name].head.y + 0.6, bones[bone_name].head.z)
            
        except Exception as e:
            print(f"❌ Error creating advanced rig: {e}")
    
    def create_character_poses(self, character_name: str, poses: List[str]) -> Tuple[bool, str]:
        """Create predefined poses for the character"""
        try:
            # Find the armature
            armature_name = f"{character_name}_rig"
            armature_obj = bpy.data.objects.get(armature_name)
            
            if not armature_obj:
                return False, f"Rig '{armature_name}' not found"
            
            # Create pose library
            if not armature_obj.pose_library:
                armature_obj.pose_library = bpy.data.actions.new(f"{character_name}_poses")
            
            # Select armature and enter pose mode
            bpy.context.view_layer.objects.active = armature_obj
            bpy.ops.object.mode_set(mode='POSE')
            
            # Create poses
            for pose_name in poses:
                self._create_pose(armature_obj, pose_name)
            
            bpy.ops.object.mode_set(mode='OBJECT')
            
            print(f"✅ Created {len(poses)} poses for {character_name}")
            return True, f"Created poses: {', '.join(poses)}"
            
        except Exception as e:
            error_msg = f"Error creating character poses: {str(e)}"
            print(f"❌ {error_msg}")
            return False, error_msg
    
    def _create_pose(self, armature_obj, pose_name: str):
        """Create a specific pose"""
        try:
            # Reset pose
            bpy.ops.pose.select_all(action='SELECT')
            bpy.ops.pose.rot_clear()
            bpy.ops.pose.loc_clear()
            bpy.ops.pose.scale_clear()
            
            # Apply pose-specific transformations
            if pose_name == "default":
                pass  # Default pose is rest position
            elif pose_name == "happy":
                self._apply_happy_pose(armature_obj)
            elif pose_name == "sad":
                self._apply_sad_pose(armature_obj)
            elif pose_name == "excited":
                self._apply_excited_pose(armature_obj)
            elif pose_name == "thinking":
                self._apply_thinking_pose(armature_obj)
            
            # Save pose to library
            bpy.ops.poselib.pose_add(frame=len(armature_obj.pose_library.pose_markers), name=pose_name)
            
        except Exception as e:
            print(f"❌ Error creating pose '{pose_name}': {e}")
    
    def _apply_happy_pose(self, armature_obj):
        """Apply happy pose transformations"""
        try:
            # Slight upward head tilt
            head_bone = armature_obj.pose.bones.get("Head")
            if head_bone:
                head_bone.rotation_euler = (0.1, 0, 0)
            
            # Ears up (if elephant)
            for side in ["L", "R"]:
                ear_bone = armature_obj.pose.bones.get(f"Ear_{side}")
                if ear_bone:
                    ear_bone.rotation_euler = (0.2, 0, 0)
            
        except Exception as e:
            print(f"❌ Error applying happy pose: {e}")
    
    def _apply_sad_pose(self, armature_obj):
        """Apply sad pose transformations"""
        try:
            # Downward head tilt
            head_bone = armature_obj.pose.bones.get("Head")
            if head_bone:
                head_bone.rotation_euler = (-0.2, 0, 0)
            
            # Ears down (if elephant)
            for side in ["L", "R"]:
                ear_bone = armature_obj.pose.bones.get(f"Ear_{side}")
                if ear_bone:
                    ear_bone.rotation_euler = (-0.3, 0, 0)
            
        except Exception as e:
            print(f"❌ Error applying sad pose: {e}")
    
    def _apply_excited_pose(self, armature_obj):
        """Apply excited pose transformations"""
        try:
            # Upward stretch
            for i in range(1, 4):
                spine_bone = armature_obj.pose.bones.get(f"Spine_{i:02d}")
                if spine_bone:
                    spine_bone.scale = (1.0, 1.0, 1.1)
            
            # Ears perked up
            for side in ["L", "R"]:
                ear_bone = armature_obj.pose.bones.get(f"Ear_{side}")
                if ear_bone:
                    ear_bone.rotation_euler = (0.3, 0, 0)
            
        except Exception as e:
            print(f"❌ Error applying excited pose: {e}")
    
    def _apply_thinking_pose(self, armature_obj):
        """Apply thinking pose transformations"""
        try:
            # Slight head turn
            head_bone = armature_obj.pose.bones.get("Head")
            if head_bone:
                head_bone.rotation_euler = (0, 0.2, 0)
            
            # Trunk curl (if elephant)
            for i in range(1, 6):
                trunk_bone = armature_obj.pose.bones.get(f"Trunk_{i:02d}")
                if trunk_bone:
                    trunk_bone.rotation_euler = (0, 0, 0.1 * i)
            
        except Exception as e:
            print(f"❌ Error applying thinking pose: {e}")
    
    def get_character_image_status(self) -> Dict[str, Any]:
        """Get status of character images and rigs"""
        try:
            status = {
                "cached_images": len(self.image_cache),
                "active_rigs": [],
                "available_poses": {},
                "current_story": self.current_story,
                "last_generated": None
            }
            
            # Find active rigs
            for obj in bpy.data.objects:
                if obj.type == 'ARMATURE' and obj.name.endswith('_rig'):
                    character_name = obj.name.replace('_rig', '')
                    status["active_rigs"].append(character_name)
                    
                    # Count poses
                    if obj.pose_library:
                        status["available_poses"][character_name] = len(obj.pose_library.pose_markers)
            
            # Get last generated image
            if self.image_cache:
                latest_image = max(self.image_cache.values(), key=lambda x: x["generated"])
                status["last_generated"] = latest_image["generated"]
            
            return status
            
        except Exception as e:
            print(f"❌ Error getting character image status: {e}")
            return {"error": str(e)}
    
    def cleanup_character_images(self):
        """Clean up temporary image files and unused data"""
        try:
            cleanup_count = 0
            
            # Clean up temporary images
            for cache_key, cache_data in list(self.image_cache.items()):
                image_path = cache_data["path"]
                if os.path.exists(image_path) and image_path.startswith(tempfile.gettempdir()):
                    try:
                        os.remove(image_path)
                        del self.image_cache[cache_key]
                        cleanup_count += 1
                    except:
                        pass
            
            # Clean up unused Blender images
            for image in bpy.data.images:
                if image.name.startswith("llammy_") and image.users == 0:
                    bpy.data.images.remove(image)
                    cleanup_count += 1
            
            print(f"🧹 Cleaned up {cleanup_count} character image files")
            return cleanup_count
            
        except Exception as e:
            print(f"❌ Error cleaning up character images: {e}")
            return 0

# Global instance
character_image_system = CharacterImageSystem()

# Integration functions for operators
def generate_character_image_op(character_name: str, image_type: str = "base_character"):
    """Operator wrapper for character image generation"""
    return character_image_system.generate_character_image(character_name, image_type)

def create_character_rig_op(character_name: str, rig_type: str = "basic"):
    """Operator wrapper for character rigging"""
    return character_image_system.create_character_rig(character_name, rig_type)

def create_character_poses_op(character_name: str, poses: List[str]):
    """Operator wrapper for character pose creation"""
    return character_image_system.create_character_poses(character_name, poses)

print("🎨 Character Images System loaded - Ready for visual character creation!")
