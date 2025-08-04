# llammy_clean_scanner.py
# CLEAN MODEL SCANNER - Just YOUR models, no suggestions
# Perfect for building fine-tune datasets from your preferred models
# FIXED VERSION - No registration errors

import bpy
import os
import sys
import json
import time
from pathlib import Path

bl_info = {
    "name": "Llammy Clean Scanner v8.5",
    "author": "JJ McQuade", 
    "version": (8, 5, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Llammy",
    "description": "Clean dual AI pipeline - scans YOUR models only, builds datasets",
    "category": "Animation",
}

print("🔍 LLAMMY CLEAN SCANNER - LOADING...")
print("=" * 50)

# Global instances - Initialize early to prevent registration errors
_dual_ai_system = None

def get_dual_ai_system():
    """Get or create dual AI system"""
    global _dual_ai_system
    if _dual_ai_system is None:
        _dual_ai_system = CleanDualAI()
    return _dual_ai_system

# Simple HTTP request function
def simple_http_get(url, timeout=10):
    """Simple HTTP GET with fallback methods"""
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except:
        try:
            import requests
            response = requests.get(url, timeout=timeout)
            return response.json() if response.status_code == 200 else None
        except:
            return None

def simple_http_post(url, data, timeout=120):
    """Simple HTTP POST with fallback methods"""
    try:
        import urllib.request
        import urllib.parse
        
        json_data = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(url, data=json_data, 
                                   headers={'Content-Type': 'application/json'})
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except:
        try:
            import requests
            response = requests.post(url, json=data, timeout=timeout)
            return response.json() if response.status_code == 200 else None
        except:
            return None

# Your Model Scanner
class YourModelScanner:
    """Scans only YOUR installed models - no suggestions"""
    
    def __init__(self):
        self.your_models = []
        self.last_scan = 0
        self.generation_history = []  # For building fine-tune datasets
        
    def scan_your_models(self):
        """Scan YOUR actual Ollama models"""
        current_time = time.time()
        
        # Cache for 60 seconds
        if current_time - self.last_scan < 60 and self.your_models:
            return self.your_models
        
        print("🔍 Scanning YOUR Ollama models...")
        
        try:
            data = simple_http_get("http://localhost:11434/api/tags")
            
            if data and "models" in data:
                models = []
                raw_models = data["models"]
                
                print(f"📦 Found {len(raw_models)} of YOUR models:")
                
                for model in raw_models:
                    name = model.get("name", "")
                    display_name = name.replace(":latest", "")
                    size = model.get("size", 0)
                    size_gb = size / (1024**3) if size else 0
                    
                    # Just show what you have - no categorization
                    description = f"{display_name}"
                    if size_gb > 0:
                        description += f" ({size_gb:.1f}GB)"
                    
                    models.append((name, display_name, description))
                    print(f"  • {display_name} ({size_gb:.1f}GB)")
                
                self.your_models = models
                self.last_scan = current_time
                
                print(f"✅ {len(models)} models ready for use!")
                return models
            
            else:
                print("⚠️ No models found - check Ollama connection")
                return [("no_connection", "Ollama offline", "Start: ollama serve")]
                
        except Exception as e:
            print(f"❌ Scan error: {e}")
            return [("error", f"Error: {str(e)}", "Check Ollama")]
    
    def generate_with_model(self, prompt, model_name, temperature=0.7):
        """Generate with your specific model"""
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False
            }
            
            result = simple_http_post("http://localhost:11434/api/generate", payload)
            
            if result and "response" in result:
                response_text = result["response"]
                
                # Store for dataset building
                self.generation_history.append({
                    "timestamp": time.time(),
                    "model": model_name,
                    "prompt": prompt,
                    "response": response_text,
                    "temperature": temperature,
                    "success": True
                })
                
                return response_text
            else:
                return f"# Error: No response from {model_name}"
                
        except Exception as e:
            return f"# Error with {model_name}: {str(e)}"
    
    def get_dataset_entries(self):
        """Get all generation history for fine-tuning dataset"""
        return self.generation_history
    
    def export_dataset(self, format_type="jsonl"):
        """Export your generation history as training dataset"""
        if not self.generation_history:
            return "No data to export"
        
        if format_type == "jsonl":
            lines = []
            for entry in self.generation_history:
                if entry["success"]:
                    # Standard fine-tuning format
                    jsonl_entry = {
                        "messages": [
                            {"role": "user", "content": entry["prompt"]},
                            {"role": "assistant", "content": entry["response"]}
                        ],
                        "model_used": entry["model"],
                        "timestamp": entry["timestamp"],
                        "temperature": entry["temperature"]
                    }
                    lines.append(json.dumps(jsonl_entry))
            
            return "\n".join(lines)
        
        return json.dumps(self.generation_history, indent=2)

# Clean Dual AI System
class CleanDualAI:
    """Clean dual AI with YOUR swappable models"""
    
    def __init__(self):
        self.scanner = YourModelScanner()
        
    def get_your_models(self):
        """Get YOUR models for both creative and technical dropdowns"""
        return self.scanner.scan_your_models()
    
    def generate_animation(self, user_input, creative_model, technical_model):
        """Generate animation with YOUR selected models"""
        start_time = time.time()
        
        try:
            # Step 1: Creative analysis with YOUR creative model
            creative_prompt = f'''You are a Creative Director for animation. Analyze this request and provide creative guidance in JSON format.

USER REQUEST: "{user_input}"

Respond with JSON:
{{
    "characters": ["character1", "character2"],
    "mood": "emotional tone",
    "visual_style": "visual approach",
    "story_beats": ["beat1", "beat2", "beat3"],
    "technical_requirements": ["requirement1", "requirement2"]
}}

Focus on creative and artistic aspects that make compelling animation.'''

            print(f"🎭 Creative analysis with YOUR model: {creative_model}")
            creative_response = self.scanner.generate_with_model(creative_prompt, creative_model, 0.8)
            
            # Step 2: Technical generation with YOUR technical model
            technical_prompt = f'''You are a Blender Python expert. Generate working code for this request.

ORIGINAL REQUEST: "{user_input}"
CREATIVE ANALYSIS: {creative_response}

Generate complete Blender Python code that:
1. Starts with 'import bpy'
2. Clears the scene safely
3. Creates the requested animation
4. Includes proper lighting and camera
5. Uses Blender 4.5 API correctly
6. Has error handling

Generate professional, working Blender Python code:'''

            print(f"⚙️ Code generation with YOUR model: {technical_model}")
            technical_response = self.scanner.generate_with_model(technical_prompt, technical_model, 0.3)
            
            # Step 3: Execute the code
            execution_result = self._execute_code(technical_response)
            
            generation_time = time.time() - start_time
            
            return {
                "success": execution_result["success"],
                "user_input": user_input,
                "creative_analysis": creative_response,
                "generated_code": technical_response,
                "execution_result": execution_result,
                "models_used": {
                    "creative": creative_model,
                    "technical": technical_model
                },
                "generation_time": generation_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "user_input": user_input,
                "generation_time": time.time() - start_time
            }
    
    def _execute_code(self, code):
        """Execute Blender code safely"""
        try:
            # Clean the code
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            
            if not code.strip().startswith("import bpy"):
                code = "import bpy\nimport mathutils\n\n" + code
            
            # Execute
            namespace = {'bpy': bpy, 'mathutils': __import__('mathutils')}
            exec(code, namespace)
            
            return {
                "success": True,
                "message": "Code executed successfully",
                "objects_created": len([obj for obj in bpy.context.scene.objects if obj.select_get()])
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Execution failed: {str(e)}"
            }

# Property functions for model dropdowns - Fixed to prevent errors
def get_your_creative_models(self, context):
    """Get YOUR models for creative dropdown"""
    try:
        dual_ai = get_dual_ai_system()
        models = dual_ai.get_your_models()
        
        if not models or models[0][0] in ['no_connection', 'error']:
            return [("none", "No models found", "Check Ollama connection")]
        
        # Add indicators for better UX but don't filter
        enhanced_models = []
        for model_id, display_name, description in models:
            if any(keyword in model_id.lower() for keyword in ['mistral', 'creative', 'story']):
                enhanced_models.append((model_id, f"🎭 {display_name}", f"{description} - Good for creative"))
            elif any(keyword in model_id.lower() for keyword in ['phi3', 'mini', '1b', '3b']):
                enhanced_models.append((model_id, f"⚡ {display_name}", f"{description} - Fast creative"))
            else:
                enhanced_models.append((model_id, display_name, description))
        
        return enhanced_models if enhanced_models else [("none", "No models", "Check Ollama")]
    except Exception as e:
        print(f"Error getting creative models: {e}")
        return [("error", "Error loading models", str(e))]

def get_your_technical_models(self, context):
    """Get YOUR models for technical dropdown"""
    try:
        dual_ai = get_dual_ai_system()
        models = dual_ai.get_your_models()
        
        if not models or models[0][0] in ['no_connection', 'error']:
            return [("none", "No models found", "Check Ollama connection")]
        
        # Add indicators for better UX but don't filter  
        enhanced_models = []
        for model_id, display_name, description in models:
            if any(keyword in model_id.lower() for keyword in ['coder', 'code', 'deepseek']):
                enhanced_models.append((model_id, f"🔧 {display_name}", f"{description} - Perfect for code"))
            elif any(keyword in model_id.lower() for keyword in ['qwen', 'technical']):
                enhanced_models.append((model_id, f"⚙️ {display_name}", f"{description} - Good for technical"))
            else:
                enhanced_models.append((model_id, display_name, description))
        
        return enhanced_models if enhanced_models else [("none", "No models", "Check Ollama")]
    except Exception as e:
        print(f"Error getting technical models: {e}")
        return [("error", "Error loading models", str(e))]

# Clean UI Panel
class LLAMMY_PT_CleanPanel(bpy.types.Panel):
    """Clean panel showing YOUR models only"""
    bl_label = "Llammy Clean Scanner v8.5"
    bl_idname = "LLAMMY_PT_clean_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Llammy'
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Header
        header_box = layout.box()
        header_box.label(text="🔍 YOUR MODELS SYSTEM", icon='BLENDER')
        
        # Model scan section
        scan_box = layout.box()
        scan_box.label(text="📦 YOUR OLLAMA MODELS", icon='SETTINGS')
        
        scan_row = scan_box.row()
        scan_row.operator("llammy.scan_your_models", text="🔄 Scan Your Models", icon='FILE_REFRESH')
        
        # Show model count
        try:
            dual_ai = get_dual_ai_system()
            your_models = dual_ai.get_your_models()
            if your_models and your_models[0][0] not in ['no_connection', 'error']:
                scan_box.label(text=f"✅ {len(your_models)} of YOUR models available")
            else:
                scan_box.alert = True
                scan_box.label(text="⚠️ No models found", icon='ERROR')
        except:
            scan_box.alert = True
            scan_box.label(text="⚠️ Connection error", icon='ERROR')
        
        # Input section
        input_box = layout.box()
        input_box.label(text="🎬 CREATE ANIMATION", icon='SEQUENCE')
        input_box.prop(scene, "llammy_user_input", text="")
        
        # Model selection - YOUR models only
        models_box = layout.box()
        models_box.label(text="🎭 SELECT YOUR MODELS", icon='NODETREE')
        
        # Creative model selection
        creative_col = models_box.column()
        creative_col.label(text="🎭 Creative Director Model:")
        creative_col.prop(scene, "llammy_your_creative_model", text="")
        
        # Technical model selection  
        technical_col = models_box.column()
        technical_col.label(text="⚙️ Technical AI Model:")
        technical_col.prop(scene, "llammy_your_technical_model", text="")
        
        # Generation
        generate_box = layout.box()
        generate_row = generate_box.row()
        generate_row.scale_y = 2.0
        generate_row.operator("llammy.generate_clean", text="🚀 GENERATE WITH YOUR MODELS", icon='PLAY')
        
        # Dataset building section
        dataset_box = layout.box()
        dataset_box.label(text="📊 DATASET BUILDING", icon='GRAPH')
        
        try:
            dual_ai = get_dual_ai_system()
            history_count = len(dual_ai.scanner.get_dataset_entries())
            dataset_box.label(text=f"📈 {history_count} generations captured for fine-tuning")
        except:
            dataset_box.label(text="📈 0 generations captured")
        
        dataset_row = dataset_box.row()
        dataset_row.operator("llammy.export_dataset", text="📤 Export Dataset", icon='EXPORT')
        dataset_row.operator("llammy.clear_dataset", text="🗑️ Clear Data", icon='TRASH')
        
        # Results section
        if hasattr(scene, 'llammy_last_result') and scene.llammy_last_result:
            results_box = layout.box()
            results_box.label(text="📊 LAST GENERATION", icon='INFO')
            
            try:
                result = json.loads(scene.llammy_last_result)
                
                if result.get('success'):
                    results_box.label(text="✅ Success!", icon='CHECKMARK')
                    
                    models_used = result.get('models_used', {})
                    results_box.label(text=f"🎭 Creative: {models_used.get('creative', 'unknown')}")
                    results_box.label(text=f"⚙️ Technical: {models_used.get('technical', 'unknown')}")
                    
                    gen_time = result.get('generation_time', 0)
                    results_box.label(text=f"⏱️ Time: {gen_time:.1f}s")
                else:
                    results_box.alert = True
                    results_box.label(text="❌ Generation failed", icon='ERROR')
                    
            except:
                results_box.label(text="⚠️ Invalid result")

# Operators
class LLAMMY_OT_ScanYourModels(bpy.types.Operator):
    """Scan your actual Ollama models"""
    bl_idname = "llammy.scan_your_models"
    bl_label = "Scan Your Models"
    bl_description = "Scan YOUR installed Ollama models"
    
    def execute(self, context):
        try:
            dual_ai = get_dual_ai_system()
            models = dual_ai.scanner.scan_your_models()
            
            if models and models[0][0] not in ['no_connection', 'error']:
                self.report({'INFO'}, f"✅ Found {len(models)} of YOUR models!")
            else:
                self.report({'WARNING'}, "⚠️ No models found - check Ollama connection")
                
        except Exception as e:
            self.report({'ERROR'}, f"❌ Scan failed: {str(e)}")
            
        return {'FINISHED'}

class LLAMMY_OT_GenerateClean(bpy.types.Operator):
    """Generate animation with your selected models"""
    bl_idname = "llammy.generate_clean"
    bl_label = "Generate with Your Models"
    bl_description = "Generate animation using your selected models"
    
    def execute(self, context):
        scene = context.scene
        
        user_input = getattr(scene, 'llammy_user_input', '')
        if not user_input.strip():
            self.report({'ERROR'}, "Enter animation description")
            return {'CANCELLED'}
        
        creative_model = getattr(scene, 'llammy_your_creative_model', '')
        technical_model = getattr(scene, 'llammy_your_technical_model', '')
        
        if not creative_model or not technical_model:
            self.report({'ERROR'}, "Select both creative and technical models")
            return {'CANCELLED'}
        
        self.report({'INFO'}, f"🎬 Generating with YOUR models...")
        
        try:
            dual_ai = get_dual_ai_system()
            result = dual_ai.generate_animation(user_input, creative_model, technical_model)
            
            scene.llammy_last_result = json.dumps(result)
            
            if result['success']:
                objects = result.get('execution_result', {}).get('objects_created', 0)
                time_taken = result.get('generation_time', 0)
                self.report({'INFO'}, f"✅ Success! {objects} objects created in {time_taken:.1f}s")
            else:
                error = result.get('error', 'Unknown error')
                self.report({'ERROR'}, f"❌ Failed: {error}")
                
        except Exception as e:
            self.report({'ERROR'}, f"System error: {str(e)}")
        
        return {'FINISHED'}

class LLAMMY_OT_ExportDataset(bpy.types.Operator):
    """Export generation history as fine-tuning dataset"""
    bl_idname = "llammy.export_dataset"
    bl_label = "Export Dataset"
    bl_description = "Export your generation history for model fine-tuning"
    
    def execute(self, context):
        try:
            dual_ai = get_dual_ai_system()
            dataset = dual_ai.scanner.export_dataset("jsonl")
            
            if dataset and dataset != "No data to export":
                # Save to file
                import tempfile
                temp_dir = tempfile.gettempdir()
                timestamp = int(time.time())
                file_path = os.path.join(temp_dir, f"llammy_dataset_{timestamp}.jsonl")
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(dataset)
                
                count = len(dual_ai.scanner.get_dataset_entries())
                self.report({'INFO'}, f"📤 Exported {count} entries to: {file_path}")
            else:
                self.report({'WARNING'}, "No data to export - generate some animations first")
                
        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {str(e)}")
            
        return {'FINISHED'}

class LLAMMY_OT_ClearDataset(bpy.types.Operator):
    """Clear generation history"""
    bl_idname = "llammy.clear_dataset"
    bl_label = "Clear Dataset"
    bl_description = "Clear captured generation history"
    
    def execute(self, context):
        try:
            dual_ai = get_dual_ai_system()
            count = len(dual_ai.scanner.generation_history)
            dual_ai.scanner.generation_history.clear()
            
            self.report({'INFO'}, f"🗑️ Cleared {count} dataset entries")
            
        except Exception as e:
            self.report({'ERROR'}, f"Clear failed: {str(e)}")
            
        return {'FINISHED'}

# Properties
def init_clean_properties():
    """Initialize properties for clean system"""
    
    bpy.types.Scene.llammy_user_input = bpy.props.StringProperty(
        name="Animation Description",
        description="Describe the animation you want to create",
        default="A character waves hello",
        maxlen=1000
    )
    
    bpy.types.Scene.llammy_your_creative_model = bpy.props.EnumProperty(
        items=get_your_creative_models,
        name="Your Creative Model"
    )
    
    bpy.types.Scene.llammy_your_technical_model = bpy.props.EnumProperty(
        items=get_your_technical_models,
        name="Your Technical Model"
    )
    
    bpy.types.Scene.llammy_last_result = bpy.props.StringProperty(
        name="Last Result",
        default=""
    )

def clear_clean_properties():
    """Clear properties"""
    props = ['llammy_user_input', 'llammy_your_creative_model', 'llammy_your_technical_model', 'llammy_last_result']
    for prop in props:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

# Global instance - moved here to prevent import errors
_dual_ai_system = None

# Registration - FIXED
classes = [
    LLAMMY_PT_CleanPanel,
    LLAMMY_OT_ScanYourModels,
    LLAMMY_OT_GenerateClean,
    LLAMMY_OT_ExportDataset,
    LLAMMY_OT_ClearDataset,
]

def register():
    """Register clean system - FIXED VERSION"""
    global _dual_ai_system
    
    print("🔄 Starting Llammy Clean Scanner registration...")
    
    try:
        # Register all classes first
        for cls in classes:
            print(f"  Registering {cls.__name__}...")
            bpy.utils.register_class(cls)
        
        print("✅ All classes registered successfully")
        
        # Initialize properties
        init_clean_properties()
        print("✅ Properties initialized")
        
        # Initialize system
        _dual_ai_system = CleanDualAI()
        print("✅ Dual AI system created")
        
        # Test model scanning
        models = _dual_ai_system.get_your_models()
        
        if models and models[0][0] not in ['no_connection', 'error']:
            print(f"✅ Found {len(models)} of YOUR models:")
            for model_id, display_name, description in models[:3]:
                print(f"  • {display_name}")
            if len(models) > 3:
                print(f"  • ... and {len(models) - 3} more")
        else:
            print("⚠️ No models found - check Ollama connection")
            print("   To fix: Run 'ollama serve' in terminal")
        
        print("\n🎉 LLAMMY CLEAN SCANNER READY!")
        print("=" * 50)
        print("🔍 Scans YOUR models only - no suggestions")
        print("🔄 Swappable model selection")
        print("📊 Builds fine-tuning datasets from your generations")
        print("🎬 Check the 'Llammy' tab in Blender sidebar!")
        
    except Exception as e:
        print(f"❌ Registration error: {e}")
        print("🔧 This usually means a previous version is still loaded")
        print("🔄 Try: Restart Blender and install again")
        raise e

def unregister():
    """Unregister clean system"""
    global _dual_ai_system
    
    print("🔄 Unregistering Llammy Clean Scanner...")
    
    # Clear properties first
    clear_clean_properties()
    
    # Unregister classes in reverse order
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
            print(f"  Unregistered {cls.__name__}")
        except Exception as e:
            print(f"  Warning: Could not unregister {cls.__name__}: {e}")
    
    # Clear global instance
    _dual_ai_system = None
    
    print("✅ Llammy Clean Scanner unregistered")

# Test registration when run directly
if __name__ == "__main__":
    print("🧪 Testing registration...")
    try:
        register()
        print("✅ Test registration successful!")
    except Exception as e:
        print(f"❌ Test registration failed: {e}")
        import traceback
        traceback.print_exc()