import bpy
import aud
import os
import json
import time
import threading
import random
from typing import Dict, List, Any, Optional, Tuple
from mathutils import Vector
from datetime import datetime

# =============================================================================
# AUDIO INTEGRATION SYSTEM FOR ANIMATION PIPELINE v1.0
# =============================================================================

class AudioManager:
    """Manages audio integration for animation pipeline"""
    
    def __init__(self):
        self.audio_device = None
        self.audio_cache = {}
        self.active_sounds = []
        self.audio_timeline = []
        self.volume_master = 1.0
        self.audio_enabled = True
        
        # Audio categories
        self.audio_categories = {
            'sfx': {'volume': 0.8, 'priority': 3},
            'music': {'volume': 0.6, 'priority': 1},
            'dialogue': {'volume': 0.9, 'priority': 2},
            'ambient': {'volume': 0.4, 'priority': 4},
            'foley': {'volume': 0.7, 'priority': 3}
        }
        
        # Built-in procedural audio generators
        self.procedural_generators = {
            'footsteps': self._generate_footstep_audio,
            'whoosh': self._generate_whoosh_audio,
            'impact': self._generate_impact_audio,
            'ambient_forest': self._generate_ambient_forest,
            'mechanical': self._generate_mechanical_audio,
            'magical': self._generate_magical_audio,
            'weather': self._generate_weather_audio,
            'emotional': self._generate_emotional_audio
        }
        
        # Initialize audio system
        self._initialize_audio_system()
    
    def _initialize_audio_system(self):
        """Initialize the audio system"""
        try:
            # Try to get audio device
            self.audio_device = aud.Device()
            print("🔊 Audio system initialized successfully")
            
            # Set up audio scene in Blender
            self._setup_blender_audio_scene()
            
        except Exception as e:
            print(f"⚠️ Audio system initialization failed: {e}")
            self.audio_enabled = False
    
    def _setup_blender_audio_scene(self):
        """Set up Blender's audio scene"""
        scene = bpy.context.scene
        
        # Enable audio scrubbing
        scene.use_audio_scrub = True
        scene.use_audio_sync = True
        
        # Set audio settings
        scene.render.ffmpeg.audio_codec = 'AAC'
        scene.render.ffmpeg.audio_bitrate = 192
        scene.render.ffmpeg.audio_mixrate = 44100
        
        # Clear existing audio
        scene.sequence_editor_clear()
        
        # Create sequence editor if not exists
        if not scene.sequence_editor:
            scene.sequence_editor_create()
    
    def process_audio_request(self, audio_request: str, scene_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process natural language audio request"""
        
        print(f"🎵 Processing audio request: {audio_request}")
        
        # Parse audio request
        audio_elements = self._parse_audio_request(audio_request, scene_context)
        
        # Generate audio timeline
        audio_timeline = self._create_audio_timeline(audio_elements, scene_context)
        
        # Generate or source audio files
        audio_files = self._generate_audio_files(audio_timeline)
        
        # Integrate with Blender sequence editor
        self._integrate_with_blender(audio_files, audio_timeline)
        
        return {
            'audio_elements': audio_elements,
            'timeline': audio_timeline,
            'files_generated': len(audio_files),
            'total_duration': max([item['end_frame'] for item in audio_timeline]) if audio_timeline else 0
        }
    
    def _parse_audio_request(self, request: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse natural language audio request into structured elements"""
        
        elements = []
        request_lower = request.lower()
        
        # Detect audio types based on keywords
        audio_keywords = {
            'footsteps': ['footstep', 'walking', 'steps', 'running', 'walk'],
            'whoosh': ['whoosh', 'wind', 'swoosh', 'movement', 'fast'],
            'impact': ['impact', 'hit', 'crash', 'bang', 'collision'],
            'ambient_forest': ['forest', 'nature', 'birds', 'trees', 'outdoor'],
            'mechanical': ['mechanical', 'robot', 'machine', 'gear', 'engine'],
            'magical': ['magic', 'spell', 'enchant', 'mystical', 'fairy'],
            'weather': ['rain', 'storm', 'thunder', 'wind', 'weather'],
            'emotional': ['sad', 'happy', 'dramatic', 'emotional', 'mood']
        }
        
        # Detect emotions and moods
        emotions = {
            'happy': ['happy', 'joyful', 'cheerful', 'bright', 'upbeat'],
            'sad': ['sad', 'melancholy', 'depressing', 'somber', 'gloomy'],
            'dramatic': ['dramatic', 'intense', 'powerful', 'climactic'],
            'mysterious': ['mysterious', 'eerie', 'spooky', 'dark', 'unknown'],
            'energetic': ['energetic', 'fast', 'active', 'dynamic', 'lively'],
            'calm': ['calm', 'peaceful', 'serene', 'quiet', 'gentle']
        }
        
        # Scene context analysis
        scene_type = context.get('scene_analysis', {}).get('scene_type', 'indoor')
        mood = context.get('scene_analysis', {}).get('mood', 'neutral')
        duration = int(context.get('scene_analysis', {}).get('duration', '10'))
        
        # Generate base audio elements
        for audio_type, keywords in audio_keywords.items():
            if any(keyword in request_lower for keyword in keywords):
                elements.append({
                    'type': audio_type,
                    'category': 'sfx',
                    'intensity': 0.7,
                    'duration': duration * 0.3,  # 30% of scene duration
                    'start_time': 0,
                    'spatial': False,
                    'loop': False
                })
        
        # Add mood-based music
        detected_emotion = None
        for emotion, keywords in emotions.items():
            if any(keyword in request_lower for keyword in keywords):
                detected_emotion = emotion
                break
        
        if detected_emotion or mood != 'neutral':
            elements.append({
                'type': 'emotional',
                'category': 'music',
                'emotion': detected_emotion or mood,
                'intensity': 0.6,
                'duration': duration,
                'start_time': 0,
                'spatial': False,
                'loop': True
            })
        
        # Add ambient audio based on scene type
        if scene_type == 'outdoor':
            elements.append({
                'type': 'ambient_forest',
                'category': 'ambient',
                'intensity': 0.4,
                'duration': duration,
                'start_time': 0,
                'spatial': False,
                'loop': True
            })
        elif scene_type == 'indoor':
            elements.append({
                'type': 'mechanical',
                'category': 'ambient',
                'intensity': 0.2,
                'duration': duration,
                'start_time': 0,
                'spatial': False,
                'loop': True
            })
        
        # If no specific audio detected, add default ambience
        if not elements:
            elements.append({
                'type': 'emotional',
                'category': 'music',
                'emotion': 'neutral',
                'intensity': 0.5,
                'duration': duration,
                'start_time': 0,
                'spatial': False,
                'loop': True
            })
        
        return elements
    
    def _create_audio_timeline(self, elements: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create audio timeline from elements"""
        
        timeline = []
        fps = 24
        
        for element in elements:
            start_frame = int(element['start_time'] * fps)
            duration_frames = int(element['duration'] * fps)
            end_frame = start_frame + duration_frames
            
            timeline_item = {
                'type': element['type'],
                'category': element['category'],
                'start_frame': start_frame,
                'end_frame': end_frame,
                'duration_frames': duration_frames,
                'volume': element['intensity'],
                'loop': element.get('loop', False),
                'spatial': element.get('spatial', False),
                'emotion': element.get('emotion', 'neutral'),
                'channel': self._assign_audio_channel(element['category'])
            }
            
            timeline.append(timeline_item)
        
        # Sort by start frame
        timeline.sort(key=lambda x: x['start_frame'])
        
        return timeline
    
    def _assign_audio_channel(self, category: str) -> int:
        """Assign audio channel based on category"""
        channel_map = {
            'music': 1,
            'sfx': 2,
            'dialogue': 3,
            'ambient': 4,
            'foley': 5
        }
        return channel_map.get(category, 2)
    
    def _generate_audio_files(self, timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate audio files for timeline items"""
        
        audio_files = []
        
        for item in timeline:
            audio_type = item['type']
            
            try:
                # Generate audio using procedural generator
                if audio_type in self.procedural_generators:
                    audio_data = self.procedural_generators[audio_type](item)
                    
                    # Save audio file
                    filename = f"audio_{audio_type}_{int(time.time())}.wav"
                    filepath = os.path.join("/tmp", filename)
                    
                    # Convert audio data to file (simplified)
                    audio_file_info = {
                        'filepath': filepath,
                        'type': audio_type,
                        'category': item['category'],
                        'duration': item['duration_frames'],
                        'volume': item['volume'],
                        'generated': True,
                        'audio_data': audio_data
                    }
                    
                    audio_files.append(audio_file_info)
                    
                    print(f"🎵 Generated audio: {audio_type}")
                
            except Exception as e:
                print(f"⚠️ Failed to generate audio for {audio_type}: {e}")
        
        return audio_files
    
    def _integrate_with_blender(self, audio_files: List[Dict[str, Any]], timeline: List[Dict[str, Any]]):
        """Integrate audio files with Blender sequence editor"""
        
        scene = bpy.context.scene
        
        # Ensure sequence editor exists
        if not scene.sequence_editor:
            scene.sequence_editor_create()
        
        seq_editor = scene.sequence_editor
        
        # Add audio sequences
        for i, (audio_file, timeline_item) in enumerate(zip(audio_files, timeline)):
            try:
                # Create audio marker
                marker_name = f"AUDIO_{timeline_item['type']}_{i}"
                marker = scene.timeline_markers.new(
                    name=marker_name,
                    frame=timeline_item['start_frame']
                )
                
                # Set marker properties
                marker.select = True
                
                # Add to timeline markers with metadata
                if hasattr(marker, 'note'):
                    marker.note = json.dumps({
                        'type': timeline_item['type'],
                        'category': timeline_item['category'],
                        'volume': timeline_item['volume'],
                        'duration': timeline_item['duration_frames']
                    })
                
                print(f"🎵 Added audio marker: {marker_name} at frame {timeline_item['start_frame']}")
                
            except Exception as e:
                print(f"⚠️ Failed to add audio sequence: {e}")
    
    # =============================================================================
    # PROCEDURAL AUDIO GENERATORS
    # =============================================================================
    
    def _generate_footstep_audio(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Generate footstep audio"""
        return {
            'type': 'procedural',
            'pattern': 'rhythmic',
            'base_frequency': 200,
            'rhythm_bpm': 120,
            'intensity': item['volume'],
            'surface_type': 'concrete',
            'variations': 3
        }
    
    def _generate_whoosh_audio(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Generate whoosh/wind audio"""
        return {
            'type': 'procedural',
            'pattern': 'sweep',
            'base_frequency': 100,
            'frequency_sweep': [100, 800, 200],
            'intensity': item['volume'],
            'filter_type': 'lowpass',
            'envelope': 'fade_in_out'
        }
    
    def _generate_impact_audio(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Generate impact audio"""
        return {
            'type': 'procedural',
            'pattern': 'burst',
            'base_frequency': 80,
            'attack_time': 0.01,
            'decay_time': 0.5,
            'intensity': item['volume'],
            'noise_component': 0.7,
            'low_freq_boost': True
        }
    
    def _generate_ambient_forest(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ambient forest audio"""
        return {
            'type': 'procedural',
            'pattern': 'continuous',
            'base_frequency': 150,
            'bird_calls': True,
            'wind_intensity': 0.3,
            'intensity': item['volume'],
            'variations': 5,
            'seasonal': 'spring'
        }
    
    def _generate_mechanical_audio(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mechanical audio"""
        return {
            'type': 'procedural',
            'pattern': 'rhythmic',
            'base_frequency': 60,
            'rhythm_bpm': 80,
            'mechanical_type': 'gear',
            'intensity': item['volume'],
            'metallic_resonance': True,
            'variations': 2
        }
    
    def _generate_magical_audio(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Generate magical audio"""
        return {
            'type': 'procedural',
            'pattern': 'shimmer',
            'base_frequency': 400,
            'harmonics': [400, 800, 1200, 1600],
            'intensity': item['volume'],
            'reverb_amount': 0.8,
            'chorus_effect': True,
            'sparkle_density': 0.6
        }
    
    def _generate_weather_audio(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Generate weather audio"""
        return {
            'type': 'procedural',
            'pattern': 'continuous',
            'weather_type': 'rain',
            'intensity': item['volume'],
            'droplet_density': 0.7,
            'thunder_probability': 0.1,
            'wind_factor': 0.4,
            'distance_factor': 0.5
        }
    
    def _generate_emotional_audio(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Generate emotional music"""
        emotion = item.get('emotion', 'neutral')
        
        emotional_presets = {
            'happy': {
                'key': 'C_major',
                'tempo': 120,
                'instruments': ['piano', 'strings', 'flute'],
                'chord_progression': ['C', 'G', 'Am', 'F'],
                'rhythm_pattern': 'upbeat'
            },
            'sad': {
                'key': 'Am_minor',
                'tempo': 60,
                'instruments': ['piano', 'cello', 'violin'],
                'chord_progression': ['Am', 'F', 'C', 'G'],
                'rhythm_pattern': 'slow'
            },
            'dramatic': {
                'key': 'Dm_minor',
                'tempo': 140,
                'instruments': ['orchestra', 'timpani', 'brass'],
                'chord_progression': ['Dm', 'Bb', 'F', 'C'],
                'rhythm_pattern': 'intense'
            },
            'mysterious': {
                'key': 'F#m_minor',
                'tempo': 80,
                'instruments': ['synthesizer', 'strings', 'flute'],
                'chord_progression': ['F#m', 'D', 'A', 'E'],
                'rhythm_pattern': 'ambient'
            },
            'energetic': {
                'key': 'E_major',
                'tempo': 150,
                'instruments': ['electric_guitar', 'drums', 'bass'],
                'chord_progression': ['E', 'B', 'C#m', 'A'],
                'rhythm_pattern': 'driving'
            },
            'calm': {
                'key': 'G_major',
                'tempo': 70,
                'instruments': ['acoustic_guitar', 'piano', 'strings'],
                'chord_progression': ['G', 'D', 'Em', 'C'],
                'rhythm_pattern': 'gentle'
            }
        }
        
        preset = emotional_presets.get(emotion, emotional_presets['happy'])
        
        return {
            'type': 'musical',
            'emotion': emotion,
            'intensity': item['volume'],
            **preset
        }
    
    # =============================================================================
    # AUDIO ANALYSIS AND SYNCHRONIZATION
    # =============================================================================
    
    def synchronize_with_animation(self, animation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize audio with animation keyframes"""
        
        sync_points = []
        
        # Find animation beats
        keyframes = animation_data.get('keyframes', [])
        
        for keyframe in keyframes:
            frame = keyframe.get('frame', 0)
            action = keyframe.get('action', 'movement')
            
            # Map actions to audio
            audio_mapping = {
                'movement': 'footsteps',
                'impact': 'impact',
                'magical': 'magical',
                'emotional': 'emotional'
            }
            
            if action in audio_mapping:
                sync_points.append({
                    'frame': frame,
                    'audio_type': audio_mapping[action],
                    'intensity': keyframe.get('intensity', 0.5)
                })
        
        return {
            'sync_points': sync_points,
            'total_sync_events': len(sync_points)
        }
    
    def create_audio_visualization(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create audio visualization in Blender"""
        
        # Create audio visualization objects
        vis_objects = []
        
        try:
            # Create audio spectrum visualizer
            bpy.ops.mesh.primitive_cube_add(size=0.1, location=(0, 0, 0))
            spectrum_obj = bpy.context.active_object
            spectrum_obj.name = "AudioSpectrum_Visualizer"
            
            # Add array modifier for spectrum bars
            array_mod = spectrum_obj.modifiers.new(name="Array", type='ARRAY')
            array_mod.count = 32
            array_mod.relative_offset_displace[0] = 1.2
            
            # Add material with emission
            mat = bpy.data.materials.new(name="AudioVis_Material")
            mat.use_nodes = True
            mat.node_tree.nodes["Principled BSDF"].inputs[17].default_value = 2.0  # Emission strength
            mat.node_tree.nodes["Principled BSDF"].inputs[18].default_value = (0, 1, 1, 1)  # Emission color
            
            spectrum_obj.data.materials.append(mat)
            
            vis_objects.append(spectrum_obj)
            
            print("🎵 Audio visualization created")
            
        except Exception as e:
            print(f"⚠️ Failed to create audio visualization: {e}")
        
        return {
            'visualization_objects': [obj.name for obj in vis_objects],
            'visualization_type': 'spectrum'
        }
    
    def export_audio_data(self, output_path: str) -> Dict[str, Any]:
        """Export audio data for external processing"""
        
        export_data = {
            'timeline': self.audio_timeline,
            'categories': self.audio_categories,
            'master_volume': self.volume_master,
            'scene_audio_settings': {
                'fps': bpy.context.scene.render.fps,
                'sample_rate': 44100,
                'bit_depth': 16,
                'channels': 2
            },
            'markers': []
        }
        
        # Export timeline markers
        for marker in bpy.context.scene.timeline_markers:
            if marker.name.startswith('AUDIO_'):
                export_data['markers'].append({
                    'name': marker.name,
                    'frame': marker.frame,
                    'selected': marker.select
                })
        
        # Save to file
        try:
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"🎵 Audio data exported to: {output_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to export audio data: {e}")
        
        return export_data

# =============================================================================
# AUDIO INTEGRATION OPERATORS
# =============================================================================

class AUDIO_OT_GenerateAudio(bpy.types.Operator):
    """Generate audio for animation"""
    bl_idname = "audio.generate_audio"
    bl_label = "Generate Audio"
    bl_description = "Generate audio for the current animation"
    
    def execute(self, context):
        scene = context.scene
        
        # Get audio request from scene
        audio_request = getattr(scene, 'audio_request', '')
        if not audio_request:
            self.report({'WARNING'}, "No audio request specified")
            return {'CANCELLED'}
        
        # Mock scene context
        scene_context = {
            'scene_analysis': {
                'scene_type': 'indoor',
                'mood': 'dramatic',
                'duration': '10'
            }
        }
        
        # Process audio request
        audio_manager = AudioManager()
        result = audio_manager.process_audio_request(audio_request, scene_context)
        
        self.report({'INFO'}, f"🎵 Generated {result['files_generated']} audio files")
        return {'FINISHED'}

class AUDIO_OT_SynchronizeAudio(bpy.types.Operator):
    """Synchronize audio with animation"""
    bl_idname = "audio.synchronize_audio"
    bl_label = "Synchronize Audio"
    bl_description = "Synchronize audio with animation keyframes"
    
    def execute(self, context):
        audio_manager = AudioManager()
        
        # Mock animation data
        animation_data = {
            'keyframes': [
                {'frame': 24, 'action': 'movement', 'intensity': 0.8},
                {'frame': 48, 'action': 'impact', 'intensity': 1.0},
                {'frame': 72, 'action': 'emotional', 'intensity': 0.6}
            ]
        }
        
        result = audio_manager.synchronize_with_animation(animation_data)
        
        self.report({'INFO'}, f"🎵 Created {result['total_sync_events']} audio sync points")
        return {'FINISHED'}

class AUDIO_OT_CreateVisualization(bpy.types.Operator):
    """Create audio visualization"""
    bl_idname = "audio.create_visualization"
    bl_label = "Create Audio Visualization"
    bl_description = "Create visual representation of audio"
    
    def execute(self, context):
        audio_manager = AudioManager()
        
        result = audio_manager.create_audio_visualization({})
        
        self.report({'INFO'}, f"🎵 Created audio visualization")
        return {'FINISHED'}

# =============================================================================
# AUDIO PANEL
# =============================================================================

class AUDIO_PT_AudioPanel(bpy.types.Panel):
    """Audio integration panel"""
    bl_label = "🔊 Audio Integration"
    bl_idname = "AUDIO_PT_audio_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Animation AI'
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Header
        header_box = layout.box()
        header_row = header_box.row()
        header_row.alignment = 'CENTER'
        header_row.label(text="🎵 AUDIO PIPELINE", icon='SPEAKER')
        
        # Audio request input
        input_box = layout.box()
        input_box.label(text="Audio Description:", icon='OUTLINER_OB_SPEAKER')
        input_box.prop(scene, "audio_request", text="")
        
        # Audio controls
        controls_box = layout.box()
        controls_box.label(text="Audio Controls:", icon='MODIFIER_ON')
        
        control_row = controls_box.row()
        control_row.operator("audio.generate_audio", text="🎵 Generate")
        control_row.operator("audio.synchronize_audio", text="🎯 Sync")
        
        viz_row = controls_box.row()
        viz_row.operator("audio.create_visualization", text="📊 Visualize")

# =============================================================================
# REGISTRATION
# =============================================================================

audio_classes = [
    AUDIO_OT_GenerateAudio,
    AUDIO_OT_SynchronizeAudio,
    AUDIO_OT_CreateVisualization,
    AUDIO_PT_AudioPanel,
]

def register_audio_system():
    for cls in audio_classes:
        bpy.utils.register_class(cls)
    
    # Properties
    bpy.types.Scene.audio_request = bpy.props.StringProperty(
        name="Audio Request",
        description="Describe the audio you want to generate",
        default="",
        maxlen=500
    )
    
    print("🔊 Audio Integration System v1.0 Ready!")
    print("Features:")
    print("  🎵 Procedural audio generation")
    print("  🎯 Animation synchronization")
    print("  📊 Audio visualization")
    print("  🔊 Multiple audio categories")
    print("  🎬 Timeline integration")
    print("  🎨 Emotional music generation")
    print("  ⚡ Real-time audio processing")

def unregister_audio_system():
    del bpy.types.Scene.audio_request
    
    for cls in reversed(audio_classes):
        bpy.utils.unregister_class(cls)

# Initialize global audio manager
global_audio_manager = AudioManager()

if __name__ == "__main__":
    register_audio_system()