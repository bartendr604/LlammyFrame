# dual_ai_creative_director.py - Complete Dual AI Integration for Llammy Framework v8.5
# Seamlessly integrates with your existing pipeline for end-to-end animation

import bpy
import json
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

@dataclass
class CreativeAnalysis:
    """Output from Creative Director AI - structured for Technical AI consumption"""
    characters: List[Dict[str, any]]
    narrative_purpose: str
    emotional_tone: str
    visual_requirements: List[str]
    key_actions: List[Dict[str, any]]
    atmosphere: Dict[str, any]
    story_beats: List[str]
    technical_constraints: Dict[str, any]
    success_probability: float
    vision_context: Optional[str] = None  # From your vision system

class CreativeDirectorAI:
    """Frontend Creative Director AI - Your story brain that understands narrative"""
    
    def __init__(self, model_manager=None, vision_service=None):
        self.model_manager = model_manager
        self.vision_service = vision_service
        self.character_knowledge = {}
        self.story_templates = {}
        self.creative_history = []
        self.active_story = "elephant_story"  # Your current story
        
        # Story system integration
        self.story_configs = {
            "elephant_story": {
                "characters": {
                    "Tien": {
                        "description": "intricately carved Jade elephant, male, wooden harmonica player",
                        "personality": "energetic, curious, clumsy",
                        "communication": "whimsical elephant-like trumpeting",
                        "colors": ["jade green", "gold accents"],
                        "props": ["wooden harmonica", "musical notes"]
                    },
                    "Nishang": {
                        "description": "female Glass-like purple jade elephant, shy, demure", 
                        "personality": "shy, emotional, artistic",
                        "communication": "dims or brightens emotional lighting",
                        "colors": ["clear glass", "purple jade", "rainbow refractions"],
                        "props": ["tattoo-like flowers on sides and ears", "emotional lighting"]
                    },
                    "Xiaohan": {
                        "description": "Wise ancient four claw Chinese dragon, narrator, mentor",
                        "personality": "wise, patient, powerful, philosophical",
                        "communication": "speaks in metaphor and philosophy",
                        "colors": ["deep purple", "silver", "cosmic blue"],
                        "props": ["ancient symbols", "floating orbs", "cosmic aura"]
                    }
                },
                "themes": ["friendship", "music", "emotion", "wisdom", "growth"],
                "visual_style": "magical realism with Asian mythology influences"
            }
        }
        
        print("🎨 Creative Director AI initialized - Your story brain is ready!")
    
    def analyze_creative_request(self, user_prompt: str, scene_context: Dict = None, 
                               reference_images: List[str] = None) -> CreativeAnalysis:
        """Analyze user request and break it down into structured creative components"""
        
        try:
            # Get vision context if reference images provided
            vision_context = self._get_vision_context(reference_images) if reference_images else None
            
            # Build comprehensive creative prompt
            creative_prompt = self._build_creative_director_prompt(user_prompt, scene_context, vision_context)
            
            # Get creative analysis from your best creative model
            creative_model = self._select_creative_model()
            response = self._call_creative_model(creative_prompt, creative_model)
            
            # Parse and structure the response
            analysis = self._parse_creative_response(response, user_prompt, vision_context)
            
            # Update character knowledge
            self._update_creative_memory(analysis, user_prompt)
            
            return analysis
            
        except Exception as e:
            print(f"❌ Creative analysis failed: {e}")
            return self._fallback_creative_analysis(user_prompt)
    
    def _get_vision_context(self, reference_images: List[str]) -> str:
        """Get vision analysis context from reference images"""
        if not self.vision_service or not reference_images:
            return None
        
        vision_analyses = []
        for image_path in reference_images:
            try:
                analysis = self.vision_service.analyze_reference_image(image_path)
                vision_analyses.append(f"Image: {image_path}\n{analysis}")
            except Exception as e:
                print(f"⚠️ Vision analysis failed for {image_path}: {e}")
        
        return "\n\n".join(vision_analyses) if vision_analyses else None
    
    def _build_creative_director_prompt(self, user_prompt: str, scene_context: Dict, vision_context: str) -> str:
        """Build comprehensive creative director prompt"""
        
        active_story = self.story_configs.get(self.active_story, {})
        characters = active_story.get("characters", {})
        themes = active_story.get("themes", [])
        visual_style = active_story.get("visual_style", "realistic")
        
        prompt = f"""You are a Creative Director AI specializing in {self.active_story} storytelling.

ACTIVE STORY UNIVERSE: {self.active_story}
Visual Style: {visual_style}
Core Themes: {', '.join(themes)}

AVAILABLE CHARACTERS:
"""
        
        for char_name, char_info in characters.items():
            prompt += f"""
{char_name}:
- Description: {char_info['description']}
- Personality: {char_info['personality']}
- Communication: {char_info['communication']}
- Colors: {', '.join(char_info['colors'])}
- Props: {', '.join(char_info['props'])}
"""
        
        if vision_context:
            prompt += f"""
VISUAL REFERENCE CONTEXT:
{vision_context}
"""
        
        if scene_context:
            prompt += f"""
CURRENT SCENE CONTEXT:
{json.dumps(scene_context, indent=2)}
"""
        
        prompt += f"""
USER REQUEST: "{user_prompt}"

Analyze this request and provide a structured creative breakdown in JSON format:

{{
    "characters": [
        {{
            "name": "character_name",
            "role_in_scene": "protagonist/supporting/background",
            "personality_traits": ["trait1", "trait2"],
            "emotional_state": "current_emotion",
            "key_actions": ["action1", "action2"],
            "visual_description": "specific_appearance_for_this_scene",
            "animation_notes": "movement_and_behavior_notes"
        }}
    ],
    "narrative_purpose": "what_story_function_this_scene_serves",
    "emotional_tone": "primary_emotion_to_convey",
    "secondary_emotions": ["emotion1", "emotion2"],
    "visual_requirements": {{
        "lighting": "mood_and_setup",
        "color_palette": ["primary_color", "secondary_color"],
        "composition": "camera_and_framing_notes",
        "atmosphere": "environmental_mood",
        "effects": ["particle_effects", "post_processing"]
    }},
    "key_actions": [
        {{
            "character": "character_name",
            "action": "specific_action_description",
            "timing": "start_frame_or_sequence",
            "emotional_weight": 0.8,
            "technical_notes": "animation_requirements"
        }}
    ],
    "story_beats": ["setup", "development", "climax", "resolution"],
    "technical_constraints": {{
        "complexity": "low/medium/high",
        "performance_target": "realtime/quality",
        "duration_estimate": "seconds",
        "resource_requirements": "memory_and_processing_notes"
    }},
    "audio_requirements": {{
        "music_mood": "emotional_music_style",
        "sound_effects": ["effect1", "effect2"],
        "character_sounds": "vocalizations_or_audio_cues"
    }},
    "success_probability": 0.85
}}

Focus on creating emotionally engaging, character-driven scenes that utilize the strengths of each character."""
        
        return prompt
    
    def _select_creative_model(self) -> str:
        """Select best available creative model"""
        if self.model_manager:
            # Try to get creative models from your model manager
            creative_models = self.model_manager.get_available_models(category="creative")
            if creative_models:
                return creative_models[0]["id"]  # Best creative model
        
        # Fallback to your known good models
        fallback_models = ["claude-3-5-sonnet", "llama3", "mistral"]
        return fallback_models[0]
    
    def _call_creative_model(self, prompt: str, model: str) -> str:
        """Call creative model with higher temperature for creativity"""
        if self.model_manager:
            try:
                # Use your existing model manager
                return self.model_manager.generate_response(
                    model=model,
                    prompt=prompt,
                    temperature=0.8,  # Higher for creativity
                    max_tokens=1500
                )
            except Exception as e:
                print(f"⚠️ Model manager call failed: {e}")
        
        # Fallback to your existing pipeline logic
        return self._fallback_model_call(prompt)
    
    def _fallback_model_call(self, prompt: str) -> str:
        """Fallback to your existing API calling logic"""
        # This integrates with your existing API calling system
        try:
            # Use your universal_api_call function if available
            import llammy_core
            if hasattr(llammy_core, 'universal_api_call'):
                return llammy_core.universal_api_call("llama3", prompt, "ollama")
        except:
            pass
        
        return "Creative analysis placeholder - check model connection"
    
    def _parse_creative_response(self, response: str, user_prompt: str, vision_context: str) -> CreativeAnalysis:
        """Parse AI response into structured CreativeAnalysis"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                return CreativeAnalysis(
                    characters=data.get('characters', []),
                    narrative_purpose=data.get('narrative_purpose', 'scene_creation'),
                    emotional_tone=data.get('emotional_tone', 'neutral'),
                    visual_requirements=data.get('visual_requirements', {}),
                    key_actions=data.get('key_actions', []),
                    atmosphere=data.get('visual_requirements', {}).get('atmosphere', {}),
                    story_beats=data.get('story_beats', ['setup', 'action', 'resolution']),
                    technical_constraints=data.get('technical_constraints', {}),
                    success_probability=data.get('success_probability', 0.7),
                    vision_context=vision_context
                )
            else:
                return self._parse_text_response(response, user_prompt, vision_context)
                
        except json.JSONDecodeError:
            return self._parse_text_response(response, user_prompt, vision_context)
    
    def _parse_text_response(self, response: str, user_prompt: str, vision_context: str) -> CreativeAnalysis:
        """Fallback text parsing when JSON fails"""
        # Extract key information from text response
        characters = self._extract_characters_from_text(response)
        emotional_tone = self._extract_emotion_from_text(response)
        
        return CreativeAnalysis(
            characters=characters,
            narrative_purpose="scene_creation",
            emotional_tone=emotional_tone,
            visual_requirements=["basic_lighting", "standard_materials"],
            key_actions=[{"character": "main", "action": user_prompt, "timing": "main"}],
            atmosphere={"mood": emotional_tone},
            story_beats=["setup", "action", "resolution"],
            technical_constraints={"complexity": "medium"},
            success_probability=0.6,
            vision_context=vision_context
        )
    
    def _extract_characters_from_text(self, text: str) -> List[Dict]:
        """Extract character information from text response"""
        active_story = self.story_configs.get(self.active_story, {})
        story_characters = active_story.get("characters", {})
        
        found_characters = []
        text_lower = text.lower()
        
        for char_name, char_info in story_characters.items():
            if char_name.lower() in text_lower:
                found_characters.append({
                    "name": char_name,
                    "role_in_scene": "main",
                    "personality_traits": char_info["personality"].split(", "),
                    "emotional_state": "neutral",
                    "key_actions": ["present"],
                    "visual_description": char_info["description"],
                    "animation_notes": "standard_animation"
                })
        
        if not found_characters:
            # Add default character
            found_characters.append({
                "name": "Tien",  # Default to Tien
                "role_in_scene": "main",
                "personality_traits": ["energetic", "curious"],
                "emotional_state": "neutral",
                "key_actions": ["present"],
                "visual_description": "jade elephant",
                "animation_notes": "standard_animation"
            })
        
        return found_characters
    
    def _extract_emotion_from_text(self, text: str) -> str:
        """Extract primary emotion from text response"""
        emotions = ["joyful", "sad", "excited", "peaceful", "dramatic", "mysterious", "energetic", "playful"]
        text_lower = text.lower()
        
        for emotion in emotions:
            if emotion in text_lower:
                return emotion
        
        return "neutral"
    
    def _update_creative_memory(self, analysis: CreativeAnalysis, user_prompt: str):
        """Update creative memory and character knowledge"""
        # Update character database
        for char in analysis.characters:
            char_name = char.get('name', 'unknown')
            if char_name != 'unknown':
                if char_name not in self.character_knowledge:
                    self.character_knowledge[char_name] = {
                        'appearances': 0,
                        'personality_traits': [],
                        'typical_actions': [],
                        'visual_consistency': {},
                        'emotional_states': []
                    }
                
                # Update character info
                self.character_knowledge[char_name]['appearances'] += 1
                
                # Add new traits
                traits = char.get('personality_traits', [])
                for trait in traits:
                    if trait not in self.character_knowledge[char_name]['personality_traits']:
                        self.character_knowledge[char_name]['personality_traits'].append(trait)
                
                # Add emotional state
                emotional_state = char.get('emotional_state', 'neutral')
                if emotional_state not in self.character_knowledge[char_name]['emotional_states']:
                    self.character_knowledge[char_name]['emotional_states'].append(emotional_state)
        
        # Store creative decision
        self.creative_history.append({
            'timestamp': time.time(),
            'user_prompt': user_prompt,
            'analysis': analysis.__dict__,
            'characters_involved': [c.get('name') for c in analysis.characters]
        })
        
        # Keep only last 50 creative decisions for performance
        if len(self.creative_history) > 50:
            self.creative_history.pop(0)
    
    def _fallback_creative_analysis(self, user_prompt: str) -> CreativeAnalysis:
        """Fallback analysis when AI fails"""
        return CreativeAnalysis(
            characters=[{
                "name": "Tien",
                "role_in_scene": "main",
                "personality_traits": ["energetic", "curious"],
                "emotional_state": "neutral",
                "key_actions": ["present"],
                "visual_description": "jade elephant with harmonica",
                "animation_notes": "standard"
            }],
            narrative_purpose="basic_scene",
            emotional_tone="neutral",
            visual_requirements={"lighting": "basic", "color_palette": ["jade", "gold"]},
            key_actions=[{"character": "Tien", "action": user_prompt, "timing": "main"}],
            atmosphere={"mood": "default"},
            story_beats=["create", "display"],
            technical_constraints={"complexity": "low"},
            success_probability=0.5
        )

class TechnicalAI:
    """Backend Technical AI - Your code generation brain"""
    
    def __init__(self, model_manager=None, correction_system=None, debug_system=None):
        self.model_manager = model_manager
        self.correction_system = correction_system
        self.debug_system = debug_system
        self.blender_patterns = {}
        self.optimization_cache = {}
        
        print("⚙️ Technical AI initialized - Your code brain is ready!")
    
    def generate_blender_script(self, creative_analysis: CreativeAnalysis, user_prompt: str) -> Tuple[str, Dict]:
        """Generate Blender Python script based on Creative Director's analysis"""
        
        try:
            # Build technical prompt with creative analysis
            technical_prompt = self._build_technical_prompt(creative_analysis, user_prompt)
            
            # Get technical response from your best technical model
            technical_model = self._select_technical_model()
            script = self._call_technical_model(technical_prompt, technical_model)
            
            # Clean and optimize script
            cleaned_script = self._clean_and_optimize_script(script, creative_analysis)
            
            # Apply your existing corrections
            if self.correction_system:
                corrected_script, corrections = self.correction_system.apply_blender_corrections(cleaned_script)
                cleaned_script = corrected_script
            
            # Update technical patterns
            self._update_technical_patterns(cleaned_script, creative_analysis)
            
            # Generate metadata
            metadata = {
                "characters_created": [c.get('name') for c in creative_analysis.characters],
                "emotional_tone": creative_analysis.emotional_tone,
                "complexity": creative_analysis.technical_constraints.get('complexity', 'medium'),
                "estimated_duration": creative_analysis.technical_constraints.get('duration_estimate', '10'),
                "features_used": ["Dual AI", "Character System", "Story Integration"]
            }
            
            return cleaned_script, metadata
            
        except Exception as e:
            print(f"❌ Technical script generation failed: {e}")
            return self._fallback_technical_script(creative_analysis, user_prompt)
    
    def _build_technical_prompt(self, analysis: CreativeAnalysis, user_prompt: str) -> str:
        """Build specialized prompt for Technical AI"""
        
        prompt = f"""You are a Blender Python Expert specializing in character animation and storytelling.

CREATIVE DIRECTOR ANALYSIS:
{json.dumps(analysis.__dict__, indent=2)}

GENERATE COMPREHENSIVE BLENDER PYTHON SCRIPT:

1. SCENE SETUP:
   - Clear scene with error handling
   - Set appropriate units and frame range
   - Create proper camera positioning for: {analysis.visual_requirements.get('composition', 'medium shot')}

2. CHARACTER CREATION:
"""
        
        for char in analysis.characters:
            char_name = char.get('name', 'character')
            char_desc = char.get('visual_description', 'basic character')
            char_actions = char.get('key_actions', [])
            
            prompt += f"""
   - {char_name}: {char_desc}
     * Role: {char.get('role_in_scene', 'supporting')}
     * Actions: {', '.join(char_actions)}
     * Animation: {char.get('animation_notes', 'standard')}
"""
        
        prompt += f"""
3. LIGHTING & ATMOSPHERE:
   - Mood: {analysis.emotional_tone}
   - Lighting: {analysis.visual_requirements.get('lighting', 'three-point')}
   - Atmosphere: {analysis.atmosphere.get('mood', 'neutral')}

4. MATERIALS & COLORS:
   - Color Palette: {analysis.visual_requirements.get('color_palette', ['default'])}
   - Material Style: {analysis.visual_requirements.get('effects', ['standard'])}

5. ANIMATION KEYFRAMES:
   - Story Beats: {', '.join(analysis.story_beats)}
   - Key Actions: {len(analysis.key_actions)} major actions
   - Duration: {analysis.technical_constraints.get('duration_estimate', '10')} seconds

6. AUDIO INTEGRATION:
   - Add timeline markers for audio sync
   - Include comments for: {analysis.__dict__.get('audio_requirements', {}).get('sound_effects', [])}

REQUIREMENTS:
- Use proper Blender 4.5+ API
- Include comprehensive error handling
- Add detailed comments for each section
- Optimize for performance: {analysis.technical_constraints.get('performance_target', 'balanced')}
- Create working, executable code

Original user request: {user_prompt}

Generate complete, professional Blender Python script:"""
        
        return prompt
    
    def _select_technical_model(self) -> str:
        """Select best available technical model"""
        if self.model_manager:
            technical_models = self.model_manager.get_available_models(category="technical")
            if technical_models:
                return technical_models[0]["id"]  # Best technical model
        
        # Fallback to your known good technical models
        fallback_models = ["qwen2.5:7b", "qwen2.5-coder:7b", "codellama:7b"]
        return fallback_models[0]
    
    def _call_technical_model(self, prompt: str, model: str) -> str:
        """Call technical model with lower temperature for precision"""
        if self.model_manager:
            try:
                return self.model_manager.generate_response(
                    model=model,
                    prompt=prompt,
                    temperature=0.3,  # Lower for technical precision
                    max_tokens=2000
                )
            except Exception as e:
                print(f"⚠️ Technical model call failed: {e}")
        
        # Fallback to your existing pipeline
        return self._fallback_technical_call(prompt)
    
    def _fallback_technical_call(self, prompt: str) -> str:
        """Fallback technical model call"""
        try:
            import llammy_core
            if hasattr(llammy_core, 'universal_api_call'):
                return llammy_core.universal_api_call("qwen2.5:7b", prompt, "ollama")
        except:
            pass
        
        return self._generate_basic_script()
    
    def _clean_and_optimize_script(self, script: str, analysis: CreativeAnalysis) -> str:
        """Clean and optimize generated script"""
        # Remove code blocks if present
        if "```python" in script:
            script = script.split("```python")[1].split("```")[0]
        elif "```" in script:
            script = script.split("```")[1].split("```")[0]
        
        # Ensure proper imports
        if not script.strip().startswith("import bpy"):
            script = "import bpy\nimport bmesh\nfrom mathutils import Vector\n\n" + script
        
        # Add complexity optimizations
        complexity = analysis.technical_constraints.get('complexity', 'medium')
        if complexity == 'low':
            script = self._add_performance_optimizations(script)
        
        return script.strip()
    
    def _add_performance_optimizations(self, script: str) -> str:
        """Add performance optimizations for low complexity scenes"""
        optimizations = [
            "# Performance optimizations for smooth playback",
            "bpy.context.scene.render.use_simplify = True",
            "bpy.context.scene.render.simplify_subdivision = 2",
            "bpy.context.scene.render.use_simplify_triangulate = True",
            ""
        ]
        
        return "\n".join(optimizations) + "\n" + script
    
    def _update_technical_patterns(self, script: str, analysis: CreativeAnalysis):
        """Update technical patterns for reuse"""
        pattern_key = f"{analysis.emotional_tone}_{analysis.technical_constraints.get('complexity')}"
        
        self.blender_patterns[pattern_key] = {
            'script_template': script[:500],  # Store first 500 chars as template
            'success_probability': analysis.success_probability,
            'usage_count': self.blender_patterns.get(pattern_key, {}).get('usage_count', 0) + 1,
            'characters_used': [c.get('name') for c in analysis.characters]
        }
    
    def _fallback_technical_script(self, analysis: CreativeAnalysis, user_prompt: str) -> Tuple[str, Dict]:
        """Generate fallback script when AI fails"""
        script = self._generate_basic_script(analysis, user_prompt)
        metadata = {
            "characters_created": [c.get('name') for c in analysis.characters],
            "emotional_tone": analysis.emotional_tone,
            "complexity": "basic",
            "features_used": ["Fallback System"]
        }
        return script, metadata
    
    def _generate_basic_script(self, analysis: CreativeAnalysis = None, user_prompt: str = "") -> str:
        """Generate basic script template"""
        characters = analysis.characters if analysis else [{"name": "Tien"}]
        
        script = f"""import bpy
import bmesh
from mathutils import Vector

# Generated by Llammy Framework v8.5 - Dual AI System
# Request: {user_prompt}
# Characters: {', '.join([c.get('name', 'Unknown') for c in characters])}

def create_scene():
    \"\"\"Create animated scene with characters\"\"\"
    
    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Create characters
"""
        
        for i, char in enumerate(characters):
            char_name = char.get('name', f'Character_{i}')
            script += f"""
    # Create {char_name}
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.5, location=({i*3}, 0, 0))
    {char_name.lower()}_obj = bpy.context.active_object
    {char_name.lower()}_obj.name = "{char_name}"
    
    # Add material for {char_name}
    mat = bpy.data.materials.new(name="{char_name}_material")
    mat.use_nodes = True
    {char_name.lower()}_obj.data.materials.append(mat)
"""
        
        script += """
    # Add lighting
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.active_object
    sun.data.energy = 3.0
    
    # Add camera
    bpy.ops.object.camera_add(location=(7, -7, 5))
    camera = bpy.context.active_object
    camera.rotation_euler = (1.1, 0, 0.785)
    
    # Set frame range
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 240
    
    print("✅ Scene created successfully!")

if __name__ == "__main__":
    create_scene()
"""
        
        return script

class DualAICoordinator:
    """Coordinator that manages the dual AI workflow"""
    
    def __init__(self, model_manager=None, vision_service=None, correction_system=None, 
                 debug_system=None, audio_system=None):
        self.creative_director = CreativeDirectorAI(model_manager, vision_service)
        self.technical_ai = TechnicalAI(model_manager, correction_system, debug_system)
        self.audio_system = audio_system
        self.collaboration_history = []
        self.performance_metrics = {}
        
        print("🎭 Dual AI Coordinator initialized - Creative + Technical minds united!")
    
    def execute_dual_ai_pipeline(self, user_input: str, context_info: str = "", 
                                reference_images: List[str] = None) -> Dict[str, Any]:
        """Execute the complete dual AI pipeline"""
        
        start_time = time.time()
        
        try:
            print("🎨 Phase 1: Creative Director analyzing request...")
            
            # PHASE 1: Creative Director analyzes request
            scene_context = self._build_scene_context(context_info)
            creative_analysis = self.creative_director.analyze_creative_request(
                user_input, scene_context, reference_images
            )
            
            print(f"✅ Creative analysis complete - {len(creative_analysis.characters)} characters, {creative_analysis.emotional_tone} tone")
            
            print("⚙️ Phase 2: Technical AI generating Blender script...")
            
            # PHASE 2: Technical AI generates Blender script
            script, metadata = self.technical_ai.generate_blender_script(
                creative_analysis, user_input
            )
            
            print("🔊 Phase 3: Audio integration...")
            
            # PHASE 3: Audio integration (if available)
            audio_info = self._integrate_audio(creative_analysis, script)
            
            # PHASE 4: Auto-debug if available
            if self.technical_ai.debug_system:
                print("🤖 Phase 4: Auto-debug validation...")
                script = self._auto_debug_validation(script, user_input, creative_analysis)
            
            # Calculate execution metrics
            execution_time = time.time() - start_time
            quality_score = self._calculate_quality_score(creative_analysis, script, metadata)
            
            # Record collaboration
            self._record_collaboration(user_input, creative_analysis, script, metadata, execution_time)
            
            # Build comprehensive result
            result = {
                'success': True,
                'creative_response': self._format_creative_response(creative_analysis),
                'code': self._format_final_script(script, metadata, creative_analysis),
                'quality_score': quality_score,
                'features_used': metadata.get('features_used', []) + ['Dual AI Pipeline'],
                'execution_time': execution_time,
                'characters_created': metadata.get('characters_created', []),
                'emotional_tone': creative_analysis.emotional_tone,
                'audio_info': audio_info,
                'vision_context': creative_analysis.vision_context is not None
            }
            
            print(f"🎉 Dual AI pipeline complete! Quality: {quality_score}%, Time: {execution_time:.1f}s")
            return result
            
        except Exception as e:
            error_msg = f"Dual AI pipeline failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            return {
                'success': False,
                'error': error_