import os
import csv
import json
import bpy
from pathlib import Path
import time
import traceback
from datetime import datetime
from llammy.rag import LlammyRAG
from llammy.debug import SelfDebuggingSystem
from llammy.api import universal_api_call, BLENDER_API_CORRECTIONS
from llammy.metrics import MetricsTracker

# Import the provided execution engine
class ExecutionResult:
    """Container for execution results and metadata"""
    def __init__(self):
        self.success = False
        self.error = None
        self.objects_created = 0
        self.warnings = []
        self.execution_time = 0
        self.output_log = []

class AIExecutionEngine:
    """Enhanced engine for safely executing AI-generated Blender scripts"""
    def __init__(self):
        self.result = ExecutionResult()
        self.original_stdout = None
        self.captured_output = StringIO()
        
    def execute_generated_script(self, script_code: str, preserve_existing: bool = False) -> Dict[str, Any]:
        import time
        start_time = time.time()
        try:
            self._setup_execution_environment()
            initial_object_count = len(bpy.data.objects)
            if preserve_existing:
                script_code = self._modify_script_for_preservation(script_code)
            self._validate_script_safety(script_code)
            self._execute_with_monitoring(script_code)
            final_object_count = len(bpy.data.objects)
            self.result.objects_created = final_object_count - (initial_object_count if preserve_existing else 0)
            self.result.success = True
            self._refresh_viewport()
        except Exception as e:
            self.result.success = False
            self.result.error = str(e)
            print(f"❌ Script execution failed: {e}")
            traceback.print_exc()
        finally:
            self.result.execution_time = time.time() - start_time
            self._cleanup_execution_environment()
        return {
            'success': self.result.success,
            'error': self.result.error,
            'objects_created': self.result.objects_created,
            'warnings': self.result.warnings,
            'execution_time': self.result.execution_time,
            'output_log': self.result.output_log
        }
    
    def _setup_execution_environment(self):
        self.original_stdout = sys.stdout
        sys.stdout = self.captured_output
        if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        if bpy.context.area and bpy.context.area.type != 'VIEW_3D':
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    bpy.context.area = area
                    break
    
    def _cleanup_execution_environment(self):
        if self.original_stdout:
            sys.stdout = self.original_stdout
        output = self.captured_output.getvalue()
        if output:
            self.result.output_log = output.split('\n')
    
    def _modify_script_for_preservation(self, script_code: str) -> str:
        preservation_code = """
import bpy
objects_to_remove = []
for obj in bpy.data.objects:
    if (obj.name.startswith(('Cube', 'Light', 'Camera')) or 
        obj.name.startswith(('AI_', 'Generated_', 'Elephant_'))):
        objects_to_remove.append(obj)
for obj in objects_to_remove:
    bpy.data.objects.remove(obj, do_unlink=True)
"""
        lines = script_code.split('\n')
        modified_lines = []
        skip_next = False
        for line in lines:
            if skip_next:
                skip_next = False
                continue
            if "bpy.ops.object.select_all(action='SELECT')" in line:
                modified_lines.append(preservation_code)
                skip_next = True
            elif "bpy.ops.object.delete" in line and not skip_next:
                continue
            else:
                modified_lines.append(line)
        return '\n'.join(modified_lines)
    
    def _validate_script_safety(self, script_code: str):
        dangerous_patterns = [
            'import os', 'import subprocess', 'import shutil', 'open(', 'file(',
            'exec(', 'eval(', '__import__', 'bpy.ops.wm.quit', 'bpy.ops.wm.save', 'bpy.app.quit'
        ]
        for pattern in dangerous_patterns:
            if pattern in script_code:
                self.result.warnings.append(f"Potentially dangerous operation detected: {pattern}")
        try:
            compile(script_code, '<ai_script>', 'exec')
        except SyntaxError as e:
            raise Exception(f"Script has syntax errors: {e}")
    
    def _execute_with_monitoring(self, script_code: str):
        exec_namespace = {
            '__builtins__': {
                'len': len, 'range': range, 'enumerate': enumerate, 'print': print,
                'abs': abs, 'min': min, 'max': max, 'round': round,
                'int': int, 'float': float, 'str': str, 'list': list,
                'dict': dict, 'tuple': tuple, 'bool': bool,
            },
            'bpy': bpy,
            'math': __import__('math'),
        }
        try:
            exec(script_code, exec_namespace)
        except Exception as e:
            error_msg = f"Script execution error: {str(e)}"
            if hasattr(e, 'lineno'):
                lines = script_code.split('\n')
                if e.lineno <= len(lines):
                    error_line = lines[e.lineno - 1]
                    error_msg += f"\nError at line {e.lineno}: {error_line}"
            raise Exception(error_msg)
    
    def _refresh_viewport(self):
        try:
            bpy.context.view_layer.update()
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            if len(bpy.data.objects) <= 50:
                bpy.ops.view3d.view_all()
        except Exception as e:
            self.result.warnings.append(f"Viewport refresh failed: {e}")

class ScriptValidator:
    @staticmethod
    def validate_bpy_script(script_code: str) -> List[str]:
        issues = []
        if not script_code.strip():
            issues.append("Script is empty")
            return issues
        if "import bpy" not in script_code:
            issues.append("Missing required 'import bpy' statement")
        deprecated_apis = {
            "material.nodes": "material.node_tree.nodes",
            "bpy.ops.mesh.cube_add": "bpy.ops.mesh.primitive_cube_add",
            "scene.objects.active": "bpy.context.active_object"
        }
        for old_api, new_api in deprecated_apis.items():
            if old_api in script_code:
                issues.append(f"Deprecated API detected: Use {new_api} instead of {old_api}")
        if "bpy.ops.anim" in script_code and "bpy.context.scene.frame_set" not in script_code:
            issues.append("Animation script may need frame_set for proper keyframing")
        if "bpy.ops.rigging" in script_code and "bpy.ops.object.armature_add" not in script_code:
            issues.append("Rigging script may require armature creation")
        line_count = len(script_code.split('\n'))
        if line_count > 1000:
            issues.append(f"Script too complex: {line_count} lines exceed recommended limit")
        return issues

# Configuration
ADDON_DIR = str(Path(__file__).parent)
RAG_DATA_DIR = os.path.join(ADDON_DIR, "llammy_rag_data")
MEMORY_CSV = os.path.join(ADDON_DIR, "llammy_memory.csv")
METRICS_CSV = os.path.join(ADDON_DIR, "llammy_metrics.csv")
FRONTEND_JSONL = os.path.join(ADDON_DIR, "llammy_frontend_data.jsonl")
OLLAMA_ENDPOINT = "http://localhost:11434"
MODEL_NAME = "llama3.1:8b"
TEST_INPUT = "Create a character rig with a red outfit and keyframe a walking cycle over 48 frames"
TIMESTAMP = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

# Ensure RAG data directory
def ensure_rag_data_dir():
    try:
        if not os.path.exists(RAG_DATA_DIR):
            os.makedirs(RAG_DATA_DIR)
            with open(os.path.join(RAG_DATA_DIR, "blender_api_441.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"module": "bpy.ops.mesh", "name": "primitive_cube_add", "description": "Add a cube to the scene"}) + "\n")
                f.write(json.dumps({"module": "bpy.data.materials", "name": "new", "description": "Create a new material"}) + "\n")
                f.write(json.dumps({"module": "bpy.ops.anim", "name": "keyframe_insert", "description": "Insert a keyframe for object properties"}) + "\n")
                f.write(json.dumps({"module": "bpy.ops.object", "name": "armature_add", "description": "Add an armature for rigging"}) + "\n")
        return os.path.exists(RAG_DATA_DIR)
    except Exception as e:
        print(f"Error creating RAG data directory: {e}")
        save_to_frontend_jsonl({"timestamp": TIMESTAMP, "type": "error", "message": f"RAG data directory creation failed: {e}"})
        return False

# Ensure memory CSV
def ensure_memory_csv():
    try:
        if not os.path.exists(MEMORY_CSV):
            with open(MEMORY_CSV, "w", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "user_input", "code", "success", "model_info"])
                writer.writeheader()
                writer.writerow({
                    "timestamp": TIMESTAMP,
                    "user_input": "Create a cube",
                    "code": "bpy.ops.mesh.primitive_cube_add(size=2.0)",
                    "success": "True",
                    "model_info": f"{MODEL_NAME} | RAG: True | Auto-Debug: Active | Score: 95%"
                })
        return os.path.exists(MEMORY_CSV)
    except Exception as e:
        print(f"Error creating memory CSV: {e}")
        save_to_frontend_jsonl({"timestamp": TIMESTAMP, "type": "error", "message": f"Memory CSV creation failed: {e}"})
        return False

# Ensure metrics CSV
def ensure_metrics_csv():
    try:
        if not os.path.exists(METRICS_CSV):
            with open(METRICS_CSV, "w", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "success", "response_time", "total_requests", "success_rate", "current_stage", "ram_usage"])
                writer.writeheader()
                writer.writerow({
                    "timestamp": TIMESTAMP,
                    "success": "True",
                    "response_time": "2.5",
                    "total_requests": "1",
                    "success_rate": "100.0",
                    "current_stage": "Initialization",
                    "ram_usage": "65.0"
                })
        return os.path.exists(METRICS_CSV)
    except Exception as e:
        print(f"Error creating metrics CSV: {e}")
        save_to_frontend_jsonl({"timestamp": TIMESTAMP, "type": "error", "message": f"Metrics CSV creation failed: {e}"})
        return False

# Save data to frontend JSONL
def save_to_frontend_jsonl(data):
    try:
        with open(FRONTEND_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        print(f"Error saving to frontend JSONL: {e}")

# Export data for fine-tuning
def export_for_finetuning():
    try:
        with open(MEMORY_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            with open(os.path.join(ADDON_DIR, "finetune_data.jsonl"), "a", encoding="utf-8") as j:
                for row in reader:
                    if row["success"] == "True" and any(keyword in row["user_input"].lower() for keyword in ["rig", "armature", "keyframe"]):
                        j.write(json.dumps({"prompt": row["user_input"], "completion": row["code"]}) + "\n")
        save_to_frontend_jsonl({"timestamp": TIMESTAMP, "type": "finetune_export", "message": "Rigging dataset exported"})
    except Exception as e:
        print(f"Error exporting fine-tuning data: {e}")
        save_to_frontend_jsonl({"timestamp": TIMESTAMP, "type": "error", "message": f"Fine-tuning export failed: {e}"})

# Enhanced SelfDebuggingSystem
class EnhancedSelfDebuggingSystem(SelfDebuggingSystem):
    def update_fix_patterns_from_memory(self):
        if not os.path.exists(MEMORY_CSV):
            print("Memory CSV not found")
            save_to_frontend_jsonl({"timestamp": TIMESTAMP, "type": "error", "message": "Memory CSV not found"})
            return
        try:
            with open(MEMORY_CSV, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for entry in reader:
                    if entry.get("success") == "True":
                        user_input = entry.get("user_input", "").lower()
                        code = entry.get("code", "")
                        error_type = self.detect_error_type_from_code(code, user_input)
                        if error_type and error_type in self.fix_patterns:
                            fix_pattern = self.extract_fix_pattern(code, user_input)
                            if fix_pattern and fix_pattern not in self.fix_patterns[error_type]:
                                self.fix_patterns[error_type].append(fix_pattern)
                                print(f"Added new fix pattern for {error_type}: {fix_pattern}")
                                save_to_frontend_jsonl({
                                    "timestamp": TIMESTAMP,
                                    "type": "fix_pattern",
                                    "error_type": error_type,
                                    "fix_pattern": fix_pattern
                                })
        except Exception as e:
            print(f"Error updating fix patterns: {e}")
            save_to_frontend_jsonl({"timestamp": TIMESTAMP, "type": "error", "message": f"Fix patterns update failed: {e}"})

    def detect_error_type_from_code(self, code, user_input):
        if "indentation" in user_input or "    " not in code:
            return "indentation_errors"
        elif "import " in code and "ModuleNotFoundError" in user_input:
            return "import_errors"
        elif any(api in code for api in ["material.nodes", "bpy.ops.mesh.cube_add"]):
            return "blender_api_errors"
        elif any(keyword in user_input for keyword in ["rig", "armature"]):
            return "rigging_errors"
        return None

    def extract_fix_pattern(self, code, user_input):
        for old_api, new_api in BLENDER_API_CORRECTIONS.items():
            if new_api in code:
                return f"Replace {old_api} with {new_api}"
        if "bpy.ops.object.armature_add" in code:
            return "Ensure armature is parented correctly for rigging"
        return None

# Enhanced LlammyRAG
class EnhancedLlammyRAG(LlammyRAG):
    def get_context_for_request(self, user_request):
        context_parts = []
        api_results = self.search_api(user_request, limit=2)
        if api_results:
            context_parts.append("=== RELEVANT BLENDER API ===")
            for item in api_results[:2]:
                module = item.get("module", "")
                name = item.get("name", "")
                context_parts.append(f"• {module}.{name}")
                save_to_frontend_jsonl({
                    "timestamp": TIMESTAMP,
                    "type": "api_result",
                    "module": module,
                    "name": name
                })
        doc_response = self.query_documentation(user_request)
        if doc_response:
            context_parts.append("=== DOCUMENTATION CONTEXT ===")
            context_parts.append(doc_response)
            save_to_frontend_jsonl({
                "timestamp": TIMESTAMP,
                "type": "doc_response",
                "content": doc_response[:100]
            })
        context_parts.append("=== LEARNED PATTERNS ===")
        learning_entries = self.load_relevant_learning_entries(user_request)
        for entry in learning_entries[:3]:
            context_parts.append(f"• Successful example: {entry['code'][:100]}...")
            save_to_frontend_jsonl({
                "timestamp": TIMESTAMP,
                "type": "learning_entry",
                "user_input": entry["user_input"],
                "code": entry["code"][:100]
            })
        context_parts.append("=== PROVEN PATTERNS ===")
        context_parts.append("• Always use material.node_tree.nodes for materials")
        context_parts.append("• Use bpy.ops.mesh.primitive_* for mesh creation")
        context_parts.append("• Enable material.use_nodes before node operations")
        context_parts.append("• Use bpy.ops.object.armature_add for rigging")
        return "\n".join(context_parts)

    def load_relevant_learning_entries(self, user_request, limit=3):
        if not os.path.exists(MEMORY_CSV):
            return []
        results = []
        query_lower = user_request.lower()
        try:
            with open(MEMORY_CSV, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for entry in reader:
                    if entry.get("success") == "True":
                        user_input = entry.get("user_input", "").lower()
                        score = 0
                        if query_lower in user_input:
                            score += 10
                        if any(word in user_input for word in query_lower.split()):
                            score += 5
                        if score > 0:
                            results.append((score, entry))
        except Exception as e:
            print(f"Error loading learning entries: {e}")
            save_to_frontend_jsonl({"timestamp": TIMESTAMP, "type": "error", "message": f"Learning entries load failed: {e}"})
        results.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in results[:limit]]

# Main test function
def test_rag_dynamic_rigging():
    print("Starting RAG, Dynamic Learning, and Datasets Accumulation Test for Llammy Frontend Model...")

    # Step 1: Ensure prerequisites
    if not all([ensure_rag_data_dir(), ensure_memory_csv(), ensure_metrics_csv()]):
        print("Failed to set up required files. Check console and llammy_frontend_data.jsonl.")
        return

    # Step 2: Initialize RAG
    print("Initializing RAG...")
    try:
        rag = EnhancedLlammyRAG()
        rag.initialize()
        if not rag.is_active:
            print("RAG initialization failed. Check console and llammy_frontend_data.jsonl.")
            save_to_frontend_jsonl({"timestamp": TIMESTAMP, "type": "error", "message": "RAG initialization failed"})
            return
        print("RAG initialized successfully")
        save_to_frontend_jsonl({"timestamp": TIMESTAMP, "type": "status", "message": "RAG initialized"})
    except Exception as e:
        print(f"RAG initialization error: {e}")
        save_to_frontend_jsonl({"timestamp": TIMESTAMP, "type": "error", "message": f"RAG initialization failed: {e}"})
        traceback.print_exc()
        return

    # Step 3: Initialize debugging system
    debug_system = EnhancedSelfDebuggingSystem()
    debug_system.update_fix_patterns_from_memory()
    print("Debug system initialized with dynamic fix patterns")
    save_to_frontend_jsonl({"timestamp": TIMESTAMP, "type": "status", "message": "Debug system initialized"})

    # Step 4: Generate context-enhanced prompt
    context = rag.get_context_for_request(TEST_INPUT)
    prompt = f"""You are a Blender 4.4.1 expert specializing in character rigging.
USER REQUEST: {TEST_INPUT}
CONTEXT: {context}
Generate professional Blender Python code for a robust character rig with IK controls and a red outfit."""
    print(f"Generated prompt:\n{prompt}")
    save_to_frontend_jsonl({"timestamp": TIMESTAMP, "type": "prompt", "content": prompt[:200]})

    # Step 5: Validate and execute code
    try:
        start_time = time.time()
        response = universal_api_call(MODEL_NAME, prompt, OLLAMA_ENDPOINT)
        response_time = time.time() - start_time
        if not response:
            print("No response from Ollama. Check server status.")
            save_to_frontend_jsonl({"timestamp": TIMESTAMP, "type": "error", "message": "No response from Ollama"})
            return
        print(f"Generated code:\n{response}")
        save_to_frontend_jsonl({"timestamp": TIMESTAMP, "type": "generated_code", "code": response[:200]})

        # Validate script
        validator = ScriptValidator()
        issues = validator.validate_bpy_script(response)
        if issues:
            print(f"Validation issues: {issues}")
            save_to_frontend_jsonl({"timestamp": TIMESTAMP, "type": "validation", "issues": issues})
            if any("Script is empty" in issue or "syntax errors" in issue.lower() for issue in issues):
                return

        # Execute with AIExecutionEngine
        engine = AIExecutionEngine()
        exec_result = engine.execute_generated_script(response, preserve_existing=True)
        if exec_result['success']:
            print("Code executed successfully")
            with open(MEMORY_CSV, "a", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "user_input", "code", "success", "model_info"])
                writer.writerow({
                    "timestamp": TIMESTAMP,
                    "user_input": TEST_INPUT,
                    "code": response,
                    "success": "True",
                    "model_info": f"{MODEL_NAME} | RAG: True | Auto-Debug: Active"
                })
            save_to_frontend_jsonl({"timestamp": TIMESTAMP, "type": "execution_result", "success": True, "code": response[:200], "objects_created": exec_result['objects_created']})
        else:
            print(f"Code execution failed: {exec_result['error']}")
            save_to_frontend_jsonl({"timestamp": TIMESTAMP, "type": "execution_result", "success": False, "code": response[:200], "error": exec_result['error']})
    except Exception as e:
        print(f"Error during code generation/execution: {e}")
        save_to_frontend_jsonl({"timestamp": TIMESTAMP, "type": "error", "message": f"Code execution failed: {e}"})
        traceback.print_exc()
        return

    # Step 6: Update metrics
    try:
        metrics = MetricsTracker()
        metrics.total_requests += 1
        metrics.success = exec_result['success']
        metrics.response_time = response_time
        metrics.success_rate = (metrics.successful_requests / metrics.total_requests * 100) if metrics.total_requests > 0 else 0
        metrics.current_stage = "Rigging Execution"
        metrics.ram_usage = metrics.get_ram_usage()
        metrics.save_metrics_to_csv()
        print("Metrics updated")
        save_to_frontend_jsonl({
            "timestamp": TIMESTAMP,
            "type": "metrics",
            "total_requests": metrics.total_requests,
            "success_rate": metrics.success_rate,
            "response_time": response_time,
            "ram_usage": metrics.ram_usage
        })
    except Exception as e:
        print(f"Error updating metrics: {e}")
        save_to_frontend_jsonl({"timestamp": TIMESTAMP, "type": "error", "message": f"Metrics update failed: {e}"})

    # Step 7: Export fine-tuning data
    export_for_finetuning()
    print("Fine-tuning data exported for Llammy frontend model")

# Register and run
if __name__ == "__main__":
    test_rag_dynamic_rigging()