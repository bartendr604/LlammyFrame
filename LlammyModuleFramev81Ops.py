# ui/llammy_operators.py - All Blender Operators
# Llammy Framework v8.5 - Modular Edition

import bpy
import time
from datetime import datetime
from ..llammy_core import (
    get_framework, get_rag_system, get_model_manager, 
    get_debug_system, get_pipeline, get_learning_system,
    get_correction_system, get_character_system, get_metrics
)

class LLAMMY_OT_RunPipeline(bpy.types.Operator):
    bl_idname = "llammy.run_pipeline"
    bl_label = "Run Pipeline"
    bl_description = "Execute the complete AI pipeline with RAG enhancement and auto-debugging"
    
    def execute(self, context):
        scene = context.scene
        user_input = scene.llammy_user_input.strip()
        context_info = getattr(scene, 'llammy_context', '').strip()
        backend = getattr(scene, 'llammy_backend', 'ollama')
        api_key = getattr(scene, 'llammy_api_key', '')
        
        if not user_input:
            self.report({'WARNING'}, "Please enter a request")
            return {'CANCELLED'}
        
        # Get systems
        pipeline = get_pipeline()
        model_manager = get_model_manager()
        metrics = get_metrics()
        framework = get_framework()
        
        if not pipeline or not framework.initialized:
            self.report({'ERROR'}, "Framework not initialized")
            return {'CANCELLED'}
        
        # Validate models
        creative_model = scene.llammy_creative_model
        technical_model = scene.llammy_technical_model
        
        if not model_manager.validate_model(creative_model, backend):
            self.report({'ERROR'}, "Invalid creative model selected")
            return {'CANCELLED'}
        
        if backend == "claude" and not api_key:
            self.report({'ERROR'}, "Claude backend requires API key")
            return {'CANCELLED'}
        
        start_time = time.time()
        
        try:
            # Execute pipeline
            result = pipeline.execute_full_pipeline(
                user_input=user_input,
                context_info=context_info,
                creative_model=creative_model,
                technical_model=technical_model,
                backend=backend,
                api_key=api_key
            )
            
            if result['success']:
                # Update scene with results
                scene.llammy_director_response = result.get('creative_response', '')[:800]
                scene.llammy_technical_response = result.get('code', '')
                scene.llammy_debug_info = result.get('debug_info', '')
                
                # Report success
                score = result.get('quality_score', 0)
                fixes = result.get('fixes_applied', 0)
                features = result.get('features_used', [])
                
                feature_text = f" ({' + '.join(features)})" if features else ""
                
                if score >= 95:
                    self.report({'INFO'}, f"🏆 EXCELLENT! Score: {score}% | Fixes: {fixes}{feature_text}")
                elif score >= 80:
                    self.report({'INFO'}, f"✅ GOOD! Score: {score}% | Fixes: {fixes}{feature_text}")
                else:
                    self.report({'WARNING'}, f"⚠️ Needs work. Score: {score}% | Fixes: {fixes}{feature_text}")
            else:
                error_msg = result.get('error', 'Unknown error')
                self.report({'ERROR'}, f"Pipeline failed: {error_msg}")
                
                if hasattr(scene, 'llammy_debug_info'):
                    scene.llammy_debug_info = f"ERROR: {error_msg}"
        
        except Exception as e:
            self.report({'ERROR'}, f"Pipeline execution failed: {str(e)}")
            
        return {'FINISHED'}

class LLAMMY_OT_ExecuteCode(bpy.types.Operator):
    bl_idname = "llammy.execute_code"
    bl_label = "Smart Execute with Auto-Debug"
    bl_description = "Execute the generated code with automatic error fixing"
    
    def execute(self, context):
        scene = context.scene
        code = getattr(scene, 'llammy_technical_response', '').strip()
        
        if not code:
            self.report({'WARNING'}, "No code to execute")
            return {'CANCELLED'}
        
        # Get debug system
        debug_system = get_debug_system()
        if not debug_system:
            self.report({'ERROR'}, "Debug system not available")
            return {'CANCELLED'}
        
        # Clean code (remove header if present)
        clean_code = self._extract_code(code)
        
        # Use autonomous error handling
        user_input = getattr(scene, 'llammy_user_input', '')
        context_info = getattr(scene, 'llammy_context', '')
        
        success, message, final_code = debug_system.safe_execute_with_auto_fix(
            clean_code, user_input, context_info
        )
        
        if success:
            self.report({'INFO'}, f"✅ {message}")
            
            # Update code if it was fixed
            if final_code != clean_code:
                scene.llammy_technical_response = self._update_code_with_header(
                    scene.llammy_technical_response, final_code
                )
                self.report({'INFO'}, "🔧 Code was auto-fixed during execution!")
        else:
            self.report({'ERROR'}, f"❌ {message}")
        
        return {'FINISHED'}
    
    def _extract_code(self, code):
        """Extract actual code, removing headers"""
        if "# Generated by Llammy Framework" in code:
            lines = code.split('\n')
            start_idx = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith('#'):
                    start_idx = i
                    break
            return '\n'.join(lines[start_idx:])
        return code
    
    def _update_code_with_header(self, original_code, new_code):
        """Update code while preserving header"""
        if "# Generated by Llammy Framework" in original_code:
            header_lines = []
            for line in original_code.split('\n'):
                if line.strip().startswith('#') or not line.strip():
                    header_lines.append(line)
                else:
                    break
            return '\n'.join(header_lines) + '\n' + new_code
        return new_code

class LLAMMY_OT_ViewCode(bpy.types.Operator):
    bl_idname = "llammy.view_code"
    bl_label = "View Code"
    bl_description = "View code in Scripting workspace"
    
    def execute(self, context):
        scene = context.scene
        code = getattr(scene, 'llammy_technical_response', '').strip()
        
        if not code:
            self.report({'WARNING'}, "No code to view")
            return {'CANCELLED'}
        
        # Switch to Scripting workspace
        bpy.context.window.workspace = bpy.data.workspaces['Scripting']
        
        # Create new text block
        text_name = f"Llammy_Code_v85_{datetime.now().strftime('%H%M')}"
        text_block = bpy.data.texts.new(name=text_name)
        text_block.from_string(code)
        
        # Set as active in text editor
        def set_active():
            for area in bpy.context.screen.areas:
                if area.type == 'TEXT_EDITOR':
                    area.spaces.active.text = text_block
        
        bpy.app.timers.register(set_active, first_interval=0.1)
        
        self.report({'INFO'}, f"Code opened: {text_name}")
        return {'FINISHED'}

class LLAMMY_OT_ValidateCode(bpy.types.Operator):
    bl_idname = "llammy.validate_code"
    bl_label = "Validate Code"
    bl_description = "Validate and fix code"
    
    def execute(self, context):
        scene = context.scene
        code = getattr(scene, 'llammy_technical_response', '').strip()
        
        if not code:
            self.report({'WARNING'}, "No code to validate")
            return {'CANCELLED'}
        
        # Get correction system
        correction_system = get_correction_system()
        if not correction_system:
            self.report({'ERROR'}, "Correction system not available")
            return {'CANCELLED'}
        
        # Extract and validate code
        clean_code = self._extract_clean_code(code)
        
        # Apply corrections
        result = correction_system.validate_and_fix(clean_code)
        
        # Update scene
        if result['fixed_code'] != clean_code:
            scene.llammy_technical_response = self._update_with_header(
                scene.llammy_technical_response, result['fixed_code']
            )
        
        # Report results
        score = result.get('quality_score', 0)
        fixes = result.get('total_fixes', 0)
        
        if score >= 95:
            self.report({'INFO'}, f"🏆 PERFECT! Score: {score}% | Fixed: {fixes}")
        elif score >= 80:
            self.report({'INFO'}, f"✅ GOOD! Score: {score}% | Fixed: {fixes}")
        else:
            self.report({'WARNING'}, f"⚠️ Needs work. Score: {score}% | Fixed: {fixes}")
        
        return {'FINISHED'}
    
    def _extract_clean_code(self, code):
        """Extract clean code without headers"""
        if "# Generated by Llammy Framework" in code:
            lines = code.split('\n')
            start_idx = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith('#'):
                    start_idx = i
                    break
            return '\n'.join(lines[start_idx:])
        return code
    
    def _update_with_header(self, original_code, new_code):
        """Update code preserving headers"""
        if "# Generated by Llammy Framework" in original_code:
            header_lines = []
            for line in original_code.split('\n'):
                if line.strip().startswith('#') or not line.strip():
                    header_lines.append(line)
                else:
                    break
            return '\n'.join(header_lines) + '\n' + new_code
        return new_code

class LLAMMY_OT_RefreshModels(bpy.types.Operator):
    bl_idname = "llammy.refresh_models"
    bl_label = "Refresh Models"
    bl_description = "Refresh model list"
    
    def execute(self, context):
        model_manager = get_model_manager()
        if not model_manager:
            self.report({'ERROR'}, "Model manager not available")
            return {'CANCELLED'}
        
        backend = getattr(context.scene, 'llammy_backend', 'ollama')
        api_key = getattr(context.scene, 'llammy_api_key', '')
        
        success, message = model_manager.refresh_models(backend, api_key)
        
        if success:
            self.report({'INFO'}, f"✅ {message}")
        else:
            self.report({'ERROR'}, f"❌ {message}")
        
        return {'FINISHED'}

class LLAMMY_OT_SwitchStory(bpy.types.Operator):
    bl_idname = "llammy.switch_story"
    bl_label = "Switch Story"
    bl_description = "Switch to a different story/character set"
    
    def execute(self, context):
        scene = context.scene
        story_name = scene.llammy_story
        
        framework = get_framework()
        if not framework:
            self.report({'ERROR'}, "Framework not available")
            return {'CANCELLED'}
        
        success = framework.switch_story(story_name)
        
        if success:
            story_info = framework.get_active_story_info()
            self.report({'INFO'}, f"✅ Switched to: {story_info.get('name', story_name)}")
            
            # Force refresh UI
            for area in bpy.context.screen.areas:
                area.tag_redraw()
        else:
            self.report({'ERROR'}, f"❌ Failed to switch to story: {story_name}")
        
        return {'FINISHED'}

class LLAMMY_OT_GenerateCharacter(bpy.types.Operator):
    bl_idname = "llammy.generate_character"
    bl_label = "Generate Character"
    bl_description = "Generate character rigging code"
    
    def execute(self, context):
        scene = context.scene
        character_name = scene.llammy_character
        
        character_system = get_character_system()
        if not character_system:
            self.report({'ERROR'}, "Character system not available")
            return {'CANCELLED'}
        
        try:
            rigging_code = character_system.generate_character_code(character_name)
            scene.llammy_technical_response = rigging_code
            
            if hasattr(scene, 'llammy_debug_info'):
                story_info = get_framework().get_active_story_info()
                scene.llammy_debug_info = f"Character: {character_name} from {story_info.get('name', 'Unknown')} story"
            
            self.report({'INFO'}, f"✅ {character_name} character generated!")
            
        except Exception as e:
            self.report({'ERROR'}, f"Character generation failed: {str(e)}")
        
        return {'FINISHED'}

class LLAMMY_OT_InitializeRAG(bpy.types.Operator):
    bl_idname = "llammy.initialize_rag"
    bl_label = "Initialize RAG"
    bl_description = "Initialize the RAG system for enhanced context awareness"
    
    def execute(self, context):
        rag_system = get_rag_system()
        if not rag_system:
            self.report({'ERROR'}, "RAG system not available")
            return {'CANCELLED'}
        
        success, message = rag_system.initialize_enhanced()
        
        if success:
            self.report({'INFO'}, f"✅ {message}")
        else:
            self.report({'ERROR'}, f"❌ {message}")
        
        return {'FINISHED'}

class LLAMMY_OT_ViewMetrics(bpy.types.Operator):
    bl_idname = "llammy.view_metrics"
    bl_label = "View Metrics"
    bl_description = "View accumulated performance metrics"
    
    def execute(self, context):
        metrics = get_metrics()
        if not metrics:
            self.report({'ERROR'}, "Metrics system not available")
            return {'CANCELLED'}
        
        try:
            report_content = metrics.generate_full_report()
            
            # Create text block
            bpy.context.window.workspace = bpy.data.workspaces['Scripting']
            text_name = f"Llammy_Metrics_v85_{datetime.now().strftime('%H%M')}"
            text_block = bpy.data.texts.new(name=text_name)
            text_block.from_string(report_content)
            
            # Set as active
            def set_active():
                for area in bpy.context.screen.areas:
                    if area.type == 'TEXT_EDITOR':
                        area.spaces.active.text = text_block
            
            bpy.app.timers.register(set_active, first_interval=0.1)
            
            self.report({'INFO'}, f"Metrics report opened: {text_name}")
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to generate metrics: {str(e)}")
        
        return {'FINISHED'}

class LLAMMY_OT_ViewDebugStats(bpy.types.Operator):
    bl_idname = "llammy.view_debug_stats"
    bl_label = "View Debug Stats"
    bl_description = "View detailed auto-debugging statistics"
    
    def execute(self, context):
        debug_system = get_debug_system()
        if not debug_system:
            self.report({'ERROR'}, "Debug system not available")
            return {'CANCELLED'}
        
        try:
            debug_content = debug_system.generate_debug_report()
            
            # Create text block
            bpy.context.window.workspace = bpy.data.workspaces['Scripting']
            text_name = f"Llammy_DebugStats_v85_{datetime.now().strftime('%H%M')}"
            text_block = bpy.data.texts.new(name=text_name)
            text_block.from_string(debug_content)
            
            # Set as active
            def set_active():
                for area in bpy.context.screen.areas:
                    if area.type == 'TEXT_EDITOR':
                        area.spaces.active.text = text_block
            
            bpy.app.timers.register(set_active, first_interval=0.1)
            
            self.report({'INFO'}, f"Debug statistics opened: {text_name}")
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to generate debug stats: {str(e)}")
        
        return {'FINISHED'}

class LLAMMY_OT_Diagnose(bpy.types.Operator):
    bl_idname = "llammy.diagnose"
    bl_label = "Diagnose"
    bl_description = "System diagnostics including all modules"
    
    def execute(self, context):
        framework = get_framework()
        if not framework:
            self.report({'ERROR'}, "Framework not available")
            return {'CANCELLED'}
        
        try:
            # Generate comprehensive diagnostics
            diagnostic_content = self._generate_diagnostics(context)
            
            # Create text block
            bpy.context.window.workspace = bpy.data.workspaces['Scripting']
            text_name = f"Llammy_Diagnostics_v85_{datetime.now().strftime('%H%M')}"
            text_block = bpy.data.texts.new(name=text_name)
            text_block.from_string(diagnostic_content)
            
            # Set as active
            def set_active():
                for area in bpy.context.screen.areas:
                    if area.type == 'TEXT_EDITOR':
                        area.spaces.active.text = text_block
            
            bpy.app.timers.register(set_active, first_interval=0.1)
            
            self.report({'INFO'}, "Diagnostics complete - Modular systems active!")
            
        except Exception as e:
            self.report({'ERROR'}, f"Diagnostics failed: {str(e)}")
        
        return {'FINISHED'}
    
    def _generate_diagnostics(self, context):
        """Generate comprehensive diagnostics report"""
        framework = get_framework()
        
        content = f"""# Llammy Framework v8.5 - Modular Edition Diagnostics
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🏗️ FRAMEWORK STATUS:
Framework Initialized: {'✅ YES' if framework.initialized else '❌ NO'}
Active Story: {framework.get_active_story_info().get('name', 'Unknown')}
Available Stories: {', '.join(framework.get_available_stories())}

## 📦 MODULE STATUS:
"""
        
        # Check each module
        modules = {
            'RAG System': get_rag_system(),
            'Model Manager': get_model_manager(),
            'Debug System': get_debug_system(),
            'Pipeline': get_pipeline(),
            'Learning System': get_learning_system(),
            'Correction System': get_correction_system(),
            'Character System': get_character_system(),
            'Metrics': get_metrics()
        }
        
        for name, module in modules.items():
            status = "✅ LOADED" if module else "❌ NOT LOADED"
            content += f"{name}: {status}\n"
        
        content += f"""

## 🎭 CHARACTER SYSTEM:
Active Story: {framework.active_story}
Story Config: {framework.get_active_story_info()}

## 🚀 MODULAR BENEFITS:
✅ Manageable file sizes (200-400 lines per module)
✅ Independent module development
✅ Swappable story systems
✅ Easy context management for AI assistance
✅ Parallel development capability
✅ Module-specific testing

## 💡 DEVELOPMENT STATUS:
Core Structure: ✅ COMPLETE
Operators Module: ✅ COMPLETE  
Character Module: 🔄 IN PROGRESS
RAG Module: 🔄 IN PROGRESS
Debug Module: 🔄 IN PROGRESS

🎉 MODULAR ARCHITECTURE SUCCESSFULLY DEPLOYED!
No more 2800 KB monolithic files!
Each module is focused, manageable, and independent.
"""
        
        return content

class LLAMMY_OT_ForceStop(bpy.types.Operator):
    bl_idname = "llammy.force_stop"
    bl_label = "Force Stop"
    bl_description = "Emergency stop all operations"
    
    def execute(self, context):
        framework = get_framework()
        if framework:
            # Stop all systems
            for system in [get_pipeline(), get_debug_system(), get_rag_system()]:
                if system and hasattr(system, 'force_stop'):
                    system.force_stop()
        
        self.report({'INFO'}, "All operations stopped")
        return {'FINISHED'}

class LLAMMY_OT_ResetAll(bpy.types.Operator):
    bl_idname = "llammy.reset_all"
    bl_label = "Reset All"
    bl_description = "Reset all fields and status"
    
    def execute(self, context):
        scene = context.scene
        
        # Clear all scene properties
        scene.llammy_user_input = ""
        scene.llammy_context = ""
        if hasattr(scene, 'llammy_director_response'):
            scene.llammy_director_response = ""
        if hasattr(scene, 'llammy_technical_response'):
            scene.llammy_technical_response = ""
        if hasattr(scene, 'llammy_debug_info'):
            scene.llammy_debug_info = ""
        
        # Reset systems
        framework = get_framework()
        if framework:
            for system in [get_pipeline(), get_debug_system(), get_metrics()]:
                if system and hasattr(system, 'reset'):
                    system.reset()
        
        self.report({'INFO'}, "All fields and systems reset")
        return {'FINISHED'}

# List of all operator classes for registration
OPERATOR_CLASSES = [
    LLAMMY_OT_RunPipeline,
    LLAMMY_OT_ExecuteCode,
    LLAMMY_OT_ViewCode,
    LLAMMY_OT_ValidateCode,
    LLAMMY_OT_RefreshModels,
    LLAMMY_OT_SwitchStory,
    LLAMMY_OT_GenerateCharacter,
    LLAMMY_OT_InitializeRAG,
    LLAMMY_OT_ViewMetrics,
    LLAMMY_OT_ViewDebugStats,
    LLAMMY_OT_Diagnose,
    LLAMMY_OT_ForceStop,
    LLAMMY_OT_ResetAll,
]

def register_operators():
    """Register all operator classes"""
    for cls in OPERATOR_CLASSES:
        bpy.utils.register_class(cls)
    print("✅ Operators module registered")

def unregister_operators():
    """Unregister all operator classes"""
    for cls in reversed(OPERATOR_CLASSES):
        bpy.utils.unregister_class(cls)
    print("🔄 Operators module unregistered")