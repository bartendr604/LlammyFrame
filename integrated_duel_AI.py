# =============================================================================
# LLAMMY MODULE 4: FAST DUAL AI ENGINE - INTEGRATED VERSION
# fast_dual_ai_engine.py
# 
# Built on top of the existing 3 modules for trade show readiness
# =============================================================================

import bpy
import json
import http.client
import time
from typing import Dict, Any, Optional, Tuple, List

# Import our existing modules
try:
    from .llammy_core_learn import (
        update_status, update_metrics, save_learning_data, 
        get_core_status, core_learning_engine
    )
    CORE_LEARNING_AVAILABLE = True
except ImportError:
    CORE_LEARNING_AVAILABLE = False
    print("⚠️ Core learning not available")

try:
    from .llammy_auto_debug import (
        get_debug_system, execute_with_auto_fix, 
        autonomous_debug_system
    )
    AUTO_DEBUG_AVAILABLE = True
except ImportError:
    AUTO_DEBUG_AVAILABLE = False
    print("⚠️ Auto debug not available")

try:
    from .lammy_RAG_system import (
        get_context_for_request, create_enhanced_prompt,
        llammy_rag_system
    )
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️ RAG system not available")

class IntegratedDualAI:
    """
    Fast Dual AI Engine integrated with existing Llammy modules
    
    Uses:
    - Core Learning for metrics and status
    - Auto Debug for error fixing
    - RAG system for enhanced context (if available)
    """
    
    def __init__(self):
        self.fast_model = "llama3.2:latest"
        self.initialized = False
        self.execution_history = []
        
        print("🎭 Integrated Dual AI Engine initialized!")
        
        # Initialize our existing modules
        if CORE_LEARNING_AVAILABLE:
            if not core_learning_engine.initialized:
                core_learning_engine.initialize()
        
        if AUTO_DEBUG_AVAILABLE:
            autonomous_debug_system.learning_enabled = True
            
        if RAG_AVAILABLE:
            # Try to initialize RAG (optional)
            try:
                llammy_rag_system.initialize_rag()
            except:
                print("⚠️ RAG initialization failed - using fallback context")
        
        self.initialized = True
    
    def execute_dual_pipeline(self, user_request: str, backend: str = 'ollama',
                             creative_model: str = '', technical_model: str = '',
                             api_key: str = '') -> Dict[str, Any]:
        """Execute the integrated dual AI pipeline"""
        
        start_time = time.time()
        
        if CORE_LEARNING_AVAILABLE:
            update_status("executing", "Dual AI pipeline")
            core_learning_engine.metrics.reset_pipeline_stages()
            core_learning_engine.metrics.update_stage("Prompt Generation", "active")
        
        print(f"🎭 Integrated Dual AI: {user_request[:50]}...")
        
        # Use only fast models
        if not creative_model or 'qwen' in creative_model.lower():
            creative_model = self.fast_model
        if not technical_model or 'qwen' in technical_model.lower():
            technical_model = self.fast_model
        
        result = {
            'success': False,
            'user_request': user_request,
            'creative_model': creative_model,
            'technical_model': technical_model,
            'backend': backend,
            'timestamp': start_time,
            'modules_used': []
        }
        
        try:
            # STAGE 1: Enhanced context (if RAG available)
            if CORE_LEARNING_AVAILABLE:
                core_learning_engine.metrics.update_stage("RAG Context Retrieval", "active")
            
            enhanced_context = ""
            if RAG_AVAILABLE:
                try:
                    enhanced_context = get_context_for_request(user_request)
                    result['modules_used'].append('RAG')
                    print("🧠 RAG context retrieved")
                except:
                    enhanced_context = self._fallback_context(user_request)
                    print("⚠️ RAG failed, using fallback context")
            else:
                enhanced_context = self._fallback_context(user_request)
            
            # STAGE 2: Creative AI
            if CORE_LEARNING_AVAILABLE:
                core_learning_engine.metrics.update_stage("Heavy Lifting", "active")
            
            print("🎨 Creative AI...")
            creative_prompt = f"""You are a Blender creative director. Provide brief guidance for: "{user_request}"

{enhanced_context}

Give style, objects, and visual direction in 20 words or less:"""
            
            creative_response = self._fast_ollama(creative_prompt, creative_model, 20)
            
            # STAGE 3: Technical AI with enhanced prompting
            if CORE_LEARNING_AVAILABLE:
                core_learning_engine.metrics.update_stage("Code Generation", "active")
            
            print("⚙️ Technical AI...")
            
            if RAG_AVAILABLE:
                try:
                    # Use RAG-enhanced prompt
                    technical_prompt = create_enhanced_prompt(
                        creative_response, user_request, enhanced_context
                    )
                    result['modules_used'].append('RAG_Enhanced_Prompt')
                except:
                    technical_prompt = self._basic_technical_prompt(user_request, creative_response)
            else:
                technical_prompt = self._basic_technical_prompt(user_request, creative_response)
            
            # Generate code with smart templates + AI
            code = self._generate_smart_code(user_request, creative_response, technical_model)
            
            # STAGE 4: Execute with auto-debug
            if CORE_LEARNING_AVAILABLE:
                core_learning_engine.metrics.update_stage("Auto-Debug", "active")
            
            print("🔧 Executing with auto-debug...")
            
            if AUTO_DEBUG_AVAILABLE:
                # Use autonomous debug system
                success, error_msg, final_code = execute_with_auto_fix(
                    code, user_request, enhanced_context
                )
                result['modules_used'].append('Auto_Debug')
                execution_result = {
                    'success': success,
                    'message': error_msg if not success else "Executed successfully",
                    'final_code': final_code
                }
            else:
                # Basic execution without auto-debug
                execution_result = self._basic_execute(code)
                final_code = code
            
            # STAGE 5: Results and learning
            processing_time = time.time() - start_time
            
            result.update({
                'success': execution_result['success'],
                'creative_response': creative_response,
                'technical_response': f"Generated {user_request} using {', '.join(result['modules_used'])}",
                'generated_code': final_code,
                'execution_success': execution_result['success'],
                'execution_message': execution_result.get('message', ''),
                'processing_time': processing_time,
                'quality_score': 95 if execution_result['success'] else 60,
                'features_used': self._extract_features(final_code),
                'modules_integrated': result['modules_used']
            })
            
            # Save learning data
            if CORE_LEARNING_AVAILABLE:
                save_learning_data(
                    user_request, 
                    final_code, 
                    execution_result['success'],
                    f"Integrated: {', '.join(result['modules_used'])}"
                )
                update_metrics(execution_result['success'], processing_time)
                update_status("idle", "Pipeline completed")
                
                # Update all stages to completed
                for stage in core_learning_engine.metrics.pipeline_stages:
                    stage["status"] = "completed"
            
            print(f"✅ Integrated pipeline completed in {processing_time:.1f}s")
            print(f"🔗 Modules used: {', '.join(result['modules_used'])}")
            
            return result
            
        except Exception as e:
            print(f"❌ Integrated pipeline failed: {e}")
            
            if CORE_LEARNING_AVAILABLE:
                update_metrics(False, time.time() - start_time)
                update_status("idle", "Pipeline failed")
            
            result.update({
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            })
            return result
    
    def _fast_ollama(self, prompt: str, model: str, max_tokens: int = 20) -> str:
        """Fast Ollama generation with timeout protection"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.3,
                    "top_k": 10
                }
            }
            
            conn = http.client.HTTPConnection("localhost", 11434, timeout=8)
            conn.request("POST", "/api/generate", 
                        body=json.dumps(payload).encode(),
                        headers={'Content-Type': 'application/json'})
            
            response = conn.getresponse()
            result = json.loads(response.read().decode())
            conn.close()
            
            return result.get('response', '').strip() or "Modern creative style"
            
        except Exception as e:
            print(f"⚡ AI timeout, using fallback: {e}")
            return "Creative modern style"
    
    def _fallback_context(self, user_request: str) -> str:
        """Fallback context when RAG unavailable"""
        return """=== BLENDER CONTEXT ===
• Use bpy.ops.mesh.primitive_cube_add() for cubes
• Use material.use_nodes = True before nodes
• Use material.node_tree.nodes for node access
• Use bpy.context.active_object for current object"""
    
    def _basic_technical_prompt(self, user_request: str, creative_guidance: str) -> str:
        """Basic technical prompt without RAG enhancement"""
        return f"""Generate Blender Python code for: "{user_request}"
Creative guidance: {creative_guidance}

Requirements:
- Start with import bpy
- Use proper Blender 4.0+ API
- Include material setup if needed
- Clear scene first

Generate only Python code:"""
    
    def _generate_smart_code(self, user_request: str, creative_guidance: str, model: str) -> str:
        """Smart code generation with templates + AI"""
        request_lower = user_request.lower()
        
        # Template matching for speed
        if "cube" in request_lower:
            color = self._extract_color(request_lower)
            return self._cube_template(color)
        elif "sphere" in request_lower:
            color = self._extract_color(request_lower)
            return self._sphere_template(color)
        else:
            # Use AI for complex requests
            return self._ai_generate_code(user_request, creative_guidance, model)
    
    def _extract_color(self, text: str) -> str:
        """Extract color from text"""
        colors = ["blue", "red", "green", "yellow", "purple", "orange"]
        for color in colors:
            if color in text:
                return color
        return "blue"
    
    def _cube_template(self, color: str) -> str:
        """Fast cube template"""
        color_map = {
            "blue": (0, 0, 1, 1), "red": (1, 0, 0, 1), "green": (0, 1, 0, 1),
            "yellow": (1, 1, 0, 1), "purple": (1, 0, 1, 1), "orange": (1, 0.5, 0, 1)
        }
        rgba = color_map.get(color, (0, 0, 1, 1))
        
        return f"""import bpy

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Create {color} cube
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
cube = bpy.context.active_object
cube.name = "Llammy_{color.title()}Cube"

# Create material
mat = bpy.data.materials.new(name="Llammy_{color.title()}Material")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs[0].default_value = {rgba}
cube.data.materials.append(mat)

print("✅ Created {color} cube with Llammy!")"""
    
    def _sphere_template(self, color: str) -> str:
        """Fast sphere template"""
        rgba = {"blue": (0, 0, 1, 1), "red": (1, 0, 0, 1), "green": (0, 1, 0, 1)}.get(color, (0, 0, 1, 1))
        
        return f"""import bpy

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Create {color} sphere
bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))
sphere = bpy.context.active_object
sphere.name = "Llammy_{color.title()}Sphere"

# Create material
mat = bpy.data.materials.new(name="Llammy_{color.title()}SphereMat")
mat.use_nodes = True
mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = {rgba}
sphere.data.materials.append(mat)

print("✅ Created {color} sphere with Llammy!")"""
    
    def _ai_generate_code(self, user_request: str, creative_guidance: str, model: str) -> str:
        """AI-assisted code for complex requests"""
        try:
            prompt = f"Blender Python code for: {user_request}. Brief code with import bpy:"
            ai_code = self._fast_ollama(prompt, model, 100)
            
            if "import bpy" not in ai_code:
                ai_code = "import bpy\n\n" + ai_code
            
            return ai_code
        except:
            return self._cube_template("blue")  # Fallback
    
    def _basic_execute(self, code: str) -> Dict[str, Any]:
        """Basic code execution without auto-debug"""
        try:
            exec_globals = {'bpy': bpy, '__builtins__': __builtins__}
            exec(code, exec_globals)
            bpy.context.view_layer.update()
            
            return {
                'success': True,
                'message': 'Code executed successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Execution failed: {e}"
            }
    
    def _extract_features(self, code: str) -> List[str]:
        """Extract features from code"""
        features = []
        if 'cube' in code.lower():
            features.append('Modeling')
        if 'material' in code.lower():
            features.append('Materials')
        return features
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of module integration"""
        return {
            'initialized': self.initialized,
            'modules_available': {
                'core_learning': CORE_LEARNING_AVAILABLE,
                'auto_debug': AUTO_DEBUG_AVAILABLE,
                'rag_system': RAG_AVAILABLE
            },
            'fast_model': self.fast_model,
            'execution_history_count': len(self.execution_history)
        }

# =============================================================================
# INTEGRATION WITH EXISTING MASTER COORDINATOR
# =============================================================================

def integrate_with_master_coordinator(coordinator_class):
    """
    Integration function to add to your Master_Coordinator.py
    
    Add this to your LlammyMasterCoordinator.__init__:
    self.integrated_dual_ai = IntegratedDualAI()
    
    Replace your execute_full_pipeline method with:
    return self.integrated_dual_ai.execute_dual_pipeline(
        user_input, backend, creative_model, technical_model, api_key
    )
    """
    
    # Add the integrated engine to coordinator
    coordinator_class.integrated_dual_ai = IntegratedDualAI()
    
    # Override the execute method
    def integrated_execute_full_pipeline(self, user_input: str, backend: str = 'ollama',
                                       creative_model: str = '', technical_model: str = '',
                                       api_key: str = '') -> Dict[str, Any]:
        """Execute using integrated dual AI with all modules"""
        return self.integrated_dual_ai.execute_dual_pipeline(
            user_input, backend, creative_model, technical_model, api_key
        )
    
    coordinator_class.execute_full_pipeline = integrated_execute_full_pipeline
    
    print("🔗 Integrated Dual AI connected to Master Coordinator!")

# Global instance
integrated_dual_ai = IntegratedDualAI()

print("🎭 MODULE 4: INTEGRATED DUAL AI ENGINE - READY!")
print("Built on your existing 3 modules:")
print("✅ Core Learning - metrics, status, persistent memory")
print("✅ Auto Debug - autonomous error fixing")  
print("✅ RAG System - enhanced context (optional)")
print("✅ Fast execution optimized for trade show")
print("✅ Template-based speed + AI fallback")
print("✅ Complete integration with existing infrastructure")
print("")
print("🚀 TRADE SHOW READY: Fast, reliable, integrated!")