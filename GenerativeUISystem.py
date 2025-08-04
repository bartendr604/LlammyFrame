ui_integration.py - Modular UI Hookup System
Dynamically discovers and connects to any available framework components
import bpyfrom typing import Dict, Any, Optional, Listimport importlibimport sys
class ModularUIConnector:    """Discovers and connects to framework components dynamically"""
def __init__(self):
    self.framework_instance = None
    self.discovered_components = {}
    self.discovery_methods = []
    self.last_discovery = None
    
def discover_framework(self):
    """Try multiple methods to find the framework instance"""
    discovery_attempts = [
        self._try_llammy_core_coordinator,
        self._try_global_framework,
        self._try_scene_framework,
        self._try_addon_framework
    ]
    
    for attempt in discovery_attempts:
        try:
            framework = attempt()
            if framework and self._validate_framework(framework):
                self.framework_instance = framework
                print(f"✅ Framework discovered via {attempt.__name__}")
                return framework
        except Exception as e:
            print(f"⚠️ {attempt.__name__} failed: {e}")
            continue
    
    print("❌ No framework instance found")
    return None

def _try_llammy_core_coordinator(self):
    """Try to get coordinator from llammy_core"""
    try:
        # Import our enhanced core
        from . import llammy_core
        return llammy_core.coordinator
    except:
        # Try alternative import paths
        if 'llammy_core' in sys.modules:
            core_module = sys.modules['llammy_core']
            return getattr(core_module, 'coordinator', None)
        return None

def _try_global_framework(self):
    """Look for global framework variables"""
    # Check for common global variable names
    global_names = ['framework', 'llammy_framework', 'coordinator', 'main_framework']
    
    for name in global_names:
        if name in globals():
            return globals()[name]
    return None

def _try_scene_framework(self):
    """Check if framework is stored in Blender scene"""
    scene = bpy.context.scene
    framework_attrs = ['llammy_framework', 'framework_instance', 'coordinator']
    
    for attr in framework_attrs:
        if hasattr(scene, attr):
            return getattr(scene, attr)
    return None

def _try_addon_framework(self):
    """Try to find framework in addon modules"""
    addon_prefs = bpy.context.preferences.addons.get(__package__)
    if addon_prefs and hasattr(addon_prefs, 'framework'):
        return addon_prefs.framework
    return None

def _validate_framework(self, framework) -> bool:
    """Check if discovered object is a valid framework"""
    required_attrs = ['initialized', 'systems']
    return all(hasattr(framework, attr) for attr in required_attrs)

def discover_components(self, framework) -> Dict[str, Any]:
    """Dynamically discover all available components"""
    if not framework:
        return {}
    
    components = {}
    
    # Method 1: Check systems dict
    if hasattr(framework, 'systems') and isinstance(framework.systems, dict):
        for name, system in framework.systems.items():
            if system is not None:
                components[name] = {
                    'instance': system,
                    'status': self._get_component_status(system),
                    'type': type(system).__name__,
                    'methods': [m for m in dir(system) if not m.startswith('_')]
                }
    
    # Method 2: Check direct attributes
    for attr_name in dir(framework):
        if not attr_name.startswith('_'):
            attr = getattr(framework, attr_name)
            if self._looks_like_component(attr):
                components[attr_name] = {
                    'instance': attr,
                    'status': self._get_component_status(attr),
                    'type': type(attr).__name__,
                    'methods': [m for m in dir(attr) if not m.startswith('_')]
                }
    
    self.discovered_components = components
    return components

def _looks_like_component(self, obj) -> bool:
    """Heuristics to identify framework components"""
    # Skip basic types
    if isinstance(obj, (str, int, float, bool, list, dict)):
        return False
    
    # Look for component-like attributes
    component_indicators = [
        'initialize', 'initialized', 'status', 'enabled',
        'execute', 'process', 'run', 'handle'
    ]
    
    obj_attrs = dir(obj)
    return any(indicator in obj_attrs for indicator in component_indicators)

def _get_component_status(self, component) -> str:
    """Determine component status dynamically"""
    # Try different ways to get status
    status_checks = [
        lambda c: "active" if getattr(c, 'initialized', False) else "inactive",
        lambda c: "active" if getattr(c, 'enabled', False) else "inactive", 
        lambda c: getattr(c, 'status', 'unknown'),
        lambda c: "active" if callable(getattr(c, 'execute', None)) else "inactive"
    ]
    
    for check in status_checks:
        try:
            status = check(component)
            if status in ['active', 'inactive', 'error']:
                return status
        except:
            continue
    
    return "unknown"

def get_component_capabilities(self, component_name: str) -> List[str]:
    """Get capabilities of a specific component"""
    if component_name not in self.discovered_components:
        return []
    
    component_info = self.discovered_components[component_name]
    return component_info.get('methods', [])

def execute_component_method(self, component_name: str, method_name: str, *args, **kwargs):
    """Safely execute a method on a component"""
    if component_name not in self.discovered_components:
        return {"error": f"Component {component_name} not found"}
    
    component = self.discovered_components[component_name]['instance']
    
    if not hasattr(component, method_name):
        return {"error": f"Method {method_name} not found on {component_name}"}
    
    try:
        method = getattr(component, method_name)
        if callable(method):
            result = method(*args, **kwargs)
            return {"success": True, "result": result}
        else:
            return {"error": f"{method_name} is not callable"}
    except Exception as e:
        return {"error": f"Method execution failed: {str(e)}"}

Enhanced UI Panel with Dynamic Discovery
class LLAMMY_PT_ModularMainPanel(bpy.types.Panel):    """Modular main panel that adapts to available components"""    bl_label = "Llammy Framework v8.5 - Modular"    bl_idname = "LLAMMY_PT_modular_main_panel"    bl_space_type = 'VIEW_3D'    bl_region_type = 'UI'    bl_category = 'Llammy'
# Class-level connector instance
_connector = ModularUIConnector()

def draw(self, context):
    layout = self.layout
    scene = context.scene
    
    # Discover framework if not already done
    framework = self._connector.framework_instance
    if not framework:
        framework = self._connector.discover_framework()
    
    # If still no framework, show setup UI
    if not framework:
        self.draw_setup_ui(layout)
        return
    
    # Discover components dynamically
    components = self._connector.discover_components(framework)
    
    # Draw adaptive UI based on discovered components
    self.draw_framework_status(layout, framework)
    self.draw_discovered_components(layout, components)
    self.draw_adaptive_controls(layout, scene, components)
    self.draw_results_section(layout, scene)

def draw_setup_ui(self, layout):
    """Show setup UI when framework not found"""
    setup_box = layout.box()
    setup_box.alert = True
    setup_box.label(text="❌ FRAMEWORK NOT FOUND", icon='ERROR')
    
    setup_box.label(text="Possible solutions:")
    setup_box.label(text="• Run 'Initialize Framework'")
    setup_box.label(text="• Check console for errors")
    setup_box.label(text="• Restart Blender")
    
    setup_box.operator("llammy.discover_framework", text="🔍 Search for Framework")
    setup_box.operator("llammy.initialize_framework", text="🚀 Initialize Framework")

def draw_framework_status(self, layout, framework):
    """Draw framework status with real-time info"""
    status_box = layout.box()
    status_box.label(text="📊 FRAMEWORK STATUS", icon='SYSTEM')
    
    # Framework health
    health_row = status_box.row()
    if framework.initialized:
        health_row.label(text="🟢 ONLINE", icon='CHECKMARK')
    else:
        health_row.alert = True
        health_row.label(text="🔴 OFFLINE", icon='ERROR')
    
    # Component count
    components = self._connector.discovered_components
    comp_row = status_box.row()
    comp_row.label(text=f"Components: {len(components)}")
    
    # System resources (if available)
    if 'resources' in components:
        self.draw_resource_status(status_box, components['resources']['instance'])

def draw_discovered_components(self, layout, components):
    """Draw UI for all discovered components"""
    comp_box = layout.box()
    comp_box.label(text="🔧 ACTIVE COMPONENTS", icon='PREFERENCES')
    
    if not components:
        comp_box.label(text="No components discovered")
        return
    
    # Create grid for components
    grid = comp_box.grid_flow(row_major=True, columns=2, align=True)
    
    for comp_name, comp_info in components.items():
        comp_row = grid.row()
        
        status = comp_info['status']
        if status == 'active':
            comp_row.label(text=f"🟢 {comp_name.title()}")
        elif status == 'error':
            comp_row.alert = True
            comp_row.label(text=f"🔴 {comp_name.title()}")
        else:
            comp_row.label(text=f"🟡 {comp_name.title()}")
    
    # Component details toggle
    details_row = comp_box.row()
    details_row.operator("llammy.toggle_component_details", text="Show Details")

def draw_adaptive_controls(self, layout, scene, components):
    """Draw controls based on available components"""
    controls_box = layout.box()
    controls_box.label(text="🎮 CONTROLS", icon='TOOL_SETTINGS')
    
    # Basic input (always available)
    controls_box.prop(scene, "llammy_user_input", text="Request")
    
    # Model selection (if model manager available)
    if 'models' in components or 'model_manager' in components:
        model_row = controls_box.row()
        model_row.prop(scene, "llammy_creative_model", text="Creative")
        model_row.prop(scene, "llammy_technical_model", text="Technical")
    
    # Backend selection (if available)
    if hasattr(scene, 'llammy_backend'):
        controls_box.prop(scene, "llammy_backend", text="Backend")
    
    # Main action button
    action_row = controls_box.row()
    action_row.scale_y = 2.0
    
    # Check what pipeline methods are available
    pipeline_component = self._get_pipeline_component(components)
    if pipeline_component:
        action_row.operator("llammy.run_modular_pipeline", text="🚀 EXECUTE")
    else:
        action_row.enabled = False
        action_row.operator("llammy.run_modular_pipeline", text="❌ NO PIPELINE")

def draw_resource_status(self, layout, resource_checker):
    """Draw resource status if resource checker is available"""
    try:
        # Try to get resource info
        if hasattr(resource_checker, 'check_system_resources'):
            resources = resource_checker.check_system_resources()
            
            res_row = layout.row()
            ram_usage = resources.get('ram', {}).get('available', 0) / 1024  # Convert to GB
            res_row.label(text=f"RAM: {ram_usage:.1f}GB")
            
            if resources.get('ollama', {}).get('running'):
                res_row.label(text="🟢 Ollama")
            else:
                res_row.label(text="🔴 Ollama")
    except:
        pass

def draw_results_section(self, layout, scene):
    """Draw results (reuse from original)"""
    # Creative response
    if hasattr(scene, 'llammy_director_response') and scene.llammy_director_response:
        creative_box = layout.box()
        creative_box.label(text="🎨 CREATIVE VISION", icon='LIGHT_SUN')
        creative_box.prop(scene, "llammy_director_response", text="")
    
    # Technical response
    if hasattr(scene, 'llammy_technical_response') and scene.llammy_technical_response:
        tech_box = layout.box()
        tech_box.label(text="⚙️ GENERATED CODE", icon='SCRIPT')
        tech_box.prop(scene, "llammy_technical_response", text="")

def _get_pipeline_component(self, components):
    """Find pipeline component from discovered components"""
    pipeline_names = ['pipeline', 'processing_pipeline', 'main_pipeline']
    
    for name in pipeline_names:
        if name in components:
            return components[name]
    return None

Modular operators
class LLAMMY_OT_DiscoverFramework(bpy.types.Operator):    """Discover framework instance"""    bl_idname = "llammy.discover_framework"    bl_label = "Discover Framework"    bl_description = "Search for framework instance"
def execute(self, context):
    connector = LLAMMY_PT_ModularMainPanel._connector
    framework = connector.discover_framework()
    
    if framework:
        components = connector.discover_components(framework)
        self.report({'INFO'}, f"✅ Found framework with {len(components)} components")
    else:
        self.report({'ERROR'}, "❌ No framework found")
    
    return {'FINISHED'}

class LLAMMY_OT_RunModularPipeline(bpy.types.Operator):    """Execute pipeline using modular discovery"""    bl_idname = "llammy.run_modular_pipeline"    bl_label = "Run Modular Pipeline"    bl_description = "Execute pipeline with discovered components"
def execute(self, context):
    scene = context.scene
    connector = LLAMMY_PT_ModularMainPanel._connector
    
    # Get user input
    user_input = getattr(scene, 'llammy_user_input', '').strip()
    if not user_input:
        self.report({'WARNING'}, "Please enter a request")
        return {'CANCELLED'}
    
    # Find pipeline component
    components = connector.discovered_components
    pipeline_comp = None
    
    for name, comp_info in components.items():
        if 'execute' in comp_info.get('methods', []):
            pipeline_comp = comp_info['instance']
            break
    
    if not pipeline_comp:
        self.report({'ERROR'}, "No executable pipeline found")
        return {'CANCELLED'}
    
    # Execute pipeline
    try:
        # Try different execution methods
        if hasattr(pipeline_comp, 'execute_full_pipeline'):
            result = pipeline_comp.execute_full_pipeline(
                user_input, "", 
                getattr(scene, 'llammy_creative_model', 'llama3'),
                getattr(scene, 'llammy_technical_model', 'qwen2.5:7b'),
                getattr(scene, 'llammy_backend', 'ollama'),
                getattr(scene, 'llammy_api_key', '')
            )
        elif hasattr(pipeline_comp, 'execute'):
            result = pipeline_comp.execute(user_input)
        else:
            self.report({'ERROR'}, "Pipeline has no execute method")
            return {'CANCELLED'}

generative_ui.py - AI-Powered Dynamic UI Generation
Creates UI layouts, controls, and workflows based on user intent and available components
import bpyimport jsonimport refrom typing import Dict, List, Any, Optional, Tuplefrom datetime import datetimeimport ast
class UIGenerationEngine:    """AI-powered UI generator that creates interfaces based on context"""
def __init__(self):
    self.ui_templates = {}
    self.layout_patterns = {}
    self.control_mappings = {}
    self.workflow_cache = {}
    self.user_preferences = {}
    self.load_ui_knowledge()

def load_ui_knowledge(self):
    """Load UI generation knowledge base"""
    # UI Layout Patterns
    self.layout_patterns = {
        "creative_task": {
            "priority": ["inspiration", "input", "preview", "controls", "output"],
            "style": "visual_heavy",
            "colors": "warm"
        },
        "technical_task": {
            "priority": ["input", "parameters", "execute", "debug", "output"],
            "style": "data_heavy", 
            "colors": "cool"
        },
        "analysis_task": {
            "priority": ["data_input", "filters", "analysis", "visualization", "export"],
            "style": "dashboard",
            "colors": "neutral"
        },
        "modeling_task": {
            "priority": ["references", "parameters", "viewport", "tools", "properties"],
            "style": "spatial",
            "colors": "blender_native"
        }
    }
    
    # Control Type Mappings
    self.control_mappings = {
        "text_input": {"widget": "prop", "type": "StringProperty"},
        "number_input": {"widget": "prop", "type": "FloatProperty"},
        "choice_select": {"widget": "prop", "type": "EnumProperty"},
        "boolean_toggle": {"widget": "prop", "type": "BoolProperty"},
        "file_select": {"widget": "operator", "type": "FileSelect"},
        "color_pick": {"widget": "prop", "type": "FloatVectorProperty"},
        "slider_control": {"widget": "prop", "type": "FloatProperty", "slider": True},
        "button_action": {"widget": "operator", "type": "Action"}
    }

def analyze_user_intent(self, user_input: str, context: Dict) -> Dict:
    """Analyze user input to determine optimal UI layout"""
    intent_analysis = {
        "task_type": "unknown",
        "complexity": "medium",
        "required_controls": [],
        "workflow_steps": [],
        "ui_style": "standard",
        "priority_sections": []
    }
    
    # Intent classification patterns
    creative_keywords = ["create", "design", "art", "visual", "style", "aesthetic", "color", "texture"]
    technical_keywords = ["code", "script", "function", "algorithm", "debug", "optimize", "compile"]
    modeling_keywords = ["model", "mesh", "geometry", "vertex", "face", "sculpt", "modifier"]
    analysis_keywords = ["analyze", "data", "statistics", "graph", "chart", "report", "metrics"]
    
    input_lower = user_input.lower()
    
    # Determine task type
    if any(word in input_lower for word in creative_keywords):
        intent_analysis["task_type"] = "creative_task"
    elif any(word in input_lower for word in technical_keywords):
        intent_analysis["task_type"] = "technical_task"
    elif any(word in input_lower for word in modeling_keywords):
        intent_analysis["task_type"] = "modeling_task"
    elif any(word in input_lower for word in analysis_keywords):
        intent_analysis["task_type"] = "analysis_task"
    
    # Analyze complexity
    complexity_indicators = {
        "simple": ["basic", "simple", "quick", "easy"],
        "medium": ["moderate", "standard", "normal"],
        "complex": ["advanced", "complex", "detailed", "comprehensive", "enterprise"]
    }
    
    for level, indicators in complexity_indicators.items():
        if any(word in input_lower for word in indicators):
            intent_analysis["complexity"] = level
            break
    
    # Extract required controls
    intent_analysis["required_controls"] = self._extract_required_controls(user_input)
    
    # Generate workflow steps
    intent_analysis["workflow_steps"] = self._generate_workflow_steps(user_input, intent_analysis["task_type"])
    
    return intent_analysis

def _extract_required_controls(self, user_input: str) -> List[str]:
    """Extract what controls the user needs based on their input"""
    controls = []
    
    control_patterns = {
        "text_input": ["text", "message", "description", "name", "title"],
        "number_input": ["number", "count", "amount", "size", "scale", "value"],
        "choice_select": ["choose", "select", "option", "type", "mode"],
        "boolean_toggle": ["enable", "disable", "toggle", "on", "off"],
        "file_select": ["file", "import", "load", "open"],
        "color_pick": ["color", "rgb", "hue", "tint"],
        "slider_control": ["adjust", "level", "intensity", "strength"]
    }
    
    input_lower = user_input.lower()
    for control_type, keywords in control_patterns.items():
        if any(keyword in input_lower for keyword in keywords):
            controls.append(control_type)
    
    return controls

def _generate_workflow_steps(self, user_input: str, task_type: str) -> List[Dict]:
    """Generate workflow steps based on task type and input"""
    base_workflows = {
        "creative_task": [
            {"step": "Gather Inspiration", "controls": ["file_select"], "optional": True},
            {"step": "Set Parameters", "controls": ["text_input", "choice_select"]},
            {"step": "Generate Content", "controls": ["button_action"]},
            {"step": "Preview & Refine", "controls": ["boolean_toggle", "slider_control"]},
            {"step": "Finalize Output", "controls": ["button_action"]}
        ],
        "technical_task": [
            {"step": "Define Requirements", "controls": ["text_input"]},
            {"step": "Configure Parameters", "controls": ["number_input", "choice_select"]},
            {"step": "Execute Process", "controls": ["button_action"]},
            {"step": "Debug & Validate", "controls": ["button_action"]},
            {"step": "Export Results", "controls": ["file_select", "button_action"]}
        ],
        "modeling_task": [
            {"step": "Set Base Parameters", "controls": ["number_input", "choice_select"]},
            {"step": "Generate Geometry", "controls": ["button_action"]},
            {"step": "Apply Modifiers", "controls": ["choice_select", "boolean_toggle"]},
            {"step": "Material Setup", "controls": ["color_pick", "slider_control"]},
            {"step": "Final Adjustments", "controls": ["slider_control"]}
        ]
    }
    
    return base_workflows.get(task_type, base_workflows["technical_task"])

def generate_ui_layout(self, intent: Dict, available_components: Dict) -> Dict:
    """Generate complete UI layout based on intent and available components"""
    task_type = intent["task_type"]
    complexity = intent["complexity"]
    
    # Get base layout pattern
    pattern = self.layout_patterns.get(task_type, self.layout_patterns["technical_task"])
    
    # Generate sections based on workflow
    sections = []
    for step in intent["workflow_steps"]:
        section = self._generate_section(step, available_components, complexity)
        sections.append(section)
    
    # Add component-specific sections
    component_sections = self._generate_component_sections(available_components)
    sections.extend(component_sections)
    
    # Generate layout structure
    layout = {
        "task_type": task_type,
        "complexity": complexity,
        "style": pattern["style"],
        "colors": pattern["colors"],
        "sections": sections,
        "priority_order": pattern["priority"],
        "adaptive_controls": self._generate_adaptive_controls(intent, available_components),
        "realtime_elements": self._generate_realtime_elements(available_components)
    }
    
    return layout

def _generate_section(self, workflow_step: Dict, components: Dict, complexity: str) -> Dict:
    """Generate a UI section for a workflow step"""
    section = {
        "title": workflow_step["step"],
        "id": workflow_step["step"].lower().replace(" ", "_"),
        "controls": [],
        "layout_type": "box",
        "collapsible": complexity == "complex"
    }
    
    # Generate controls for this step
    for control_type in workflow_step["controls"]:
        control = self._generate_control(control_type, workflow_step["step"])
        section["controls"].append(control)
    
    return section

def _generate_control(self, control_type: str, context: str) -> Dict:
    """Generate a specific control based on type and context"""
    mapping = self.control_mappings.get(control_type, {})
    
    control = {
        "type": control_type,
        "widget": mapping.get("widget", "prop"),
        "property_type": mapping.get("type", "StringProperty"),
        "label": self._generate_control_label(control_type, context),
        "description": self._generate_control_description(control_type, context),
        "default_value": self._generate_default_value(control_type),
        "validation": self._generate_validation_rules(control_type)
    }
    
    # Add specific attributes based on control type
    if control_type == "slider_control":
        control.update({"min": 0.0, "max": 1.0, "slider": True})
    elif control_type == "choice_select":
        control.update({"items": self._generate_choice_items(context)})
    
    return control

def _generate_control_label(self, control_type: str, context: str) -> str:
    """Generate appropriate label for control"""
    label_mappings = {
        "text_input": f"{context} Text",
        "number_input": f"{context} Value",
        "choice_select": f"{context} Type",
        "boolean_toggle": f"Enable {context}",
        "file_select": f"Select File for {context}",
        "color_pick": f"{context} Color",
        "slider_control": f"{context} Intensity",
        "button_action": f"Execute {context}"
    }
    
    return label_mappings.get(control_type, context)

def _generate_control_description(self, control_type: str, context: str) -> str:
    """Generate helpful description for control"""
    descriptions = {
        "text_input": f"Enter text for {context.lower()}",
        "number_input": f"Set numeric value for {context.lower()}",
        "choice_select": f"Choose option for {context.lower()}",
        "boolean_toggle": f"Enable or disable {context.lower()}",
        "file_select": f"Browse and select file for {context.lower()}",
        "color_pick": f"Pick color for {context.lower()}",
        "slider_control": f"Adjust intensity of {context.lower()}",
        "button_action": f"Click to execute {context.lower()}"
    }
    
    return descriptions.get(control_type, f"Control for {context.lower()}")

def _generate_default_value(self, control_type: str) -> Any:
    """Generate sensible default values"""
    defaults = {
        "text_input": "",
        "number_input": 1.0,
        "choice_select": 0,
        "boolean_toggle": False,
        "color_pick": (1.0, 1.0, 1.0),
        "slider_control": 0.5
    }
    
    return defaults.get(control_type, None)

def _generate_validation_rules(self, control_type: str) -> Dict:
    """Generate validation rules for controls"""
    validations = {
        "text_input": {"min_length": 1, "max_length": 500},
        "number_input": {"min": -1000, "max": 1000},
        "slider_control": {"min": 0.0, "max": 1.0}
    }
    
    return validations.get(control_type, {})

def _generate_choice_items(self, context: str) -> List[Tuple[str, str, str]]:
    """Generate choice items based on context"""
    # This would be enhanced with more intelligent choices based on context
    common_choices = [
        ("option1", "Option 1", "First option"),
        ("option2", "Option 2", "Second option"),
        ("option3", "Option 3", "Third option")
    ]
    
    return common_choices

def _generate_component_sections(self, components: Dict) -> List[Dict]:
    """Generate sections for available components"""
    sections = []
    
    # Health monitoring section (if health monitor available)
    if any("health" in name.lower() for name in components.keys()):
        sections.append({
            "title": "System Health",
            "id": "system_health",
            "controls": [
                {"type": "realtime_display", "data_source": "health_monitor"},
                {"type": "button_action", "label": "Refresh Status"}
            ],
            "layout_type": "status_panel",
            "auto_refresh": True
        })
    
    # Debug section (if debug system available)
    if any("debug" in name.lower() for name in components.keys()):
        sections.append({
            "title": "Debug Controls",
            "id": "debug_controls", 
            "controls": [
                {"type": "boolean_toggle", "label": "Verbose Logging"},
                {"type": "button_action", "label": "Clear Logs"},
                {"type": "button_action", "label": "Export Debug Info"}
            ],
            "layout_type": "box",
            "collapsible": True
        })
    
    return sections

def _generate_adaptive_controls(self, intent: Dict, components: Dict) -> List[Dict]:
    """Generate controls that adapt based on context"""
    adaptive_controls = []
    
    # Add model selection if models available
    if any("model" in name.lower() for name in components.keys()):
        adaptive_controls.append({
            "type": "smart_model_selector",
            "adapts_to": "task_complexity",
            "recommendations": True,
            "auto_switch": intent["complexity"] == "simple"
        })
    
    # Add resource monitoring if resource checker available
    if any("resource" in name.lower() for name in components.keys()):
        adaptive_controls.append({
            "type": "resource_gauge",
            "shows": ["ram", "cpu", "gpu"],
            "alerts": True,
            "adaptive_quality": True
        })
    
    return adaptive_controls

def _generate_realtime_elements(self, components: Dict) -> List[Dict]:
    """Generate real-time UI elements"""
    realtime_elements = []
    
    # Progress tracking
    realtime_elements.append({
        "type": "progress_tracker",
        "shows": "current_operation",
        "estimated_time": True,
        "cancellable": True
    })
    
    # Performance metrics
    if any("metric" in name.lower() for name in components.keys()):
        realtime_elements.append({
            "type": "performance_display",
            "metrics": ["response_time", "success_rate", "error_count"],
            "update_interval": 1000  # ms
        })
    
    return realtime_elements

class GenerativeUIRenderer:    """Renders the generated UI layout in Blender"""
def __init__(self):
    self.property_registry = {}
    self.operator_registry = {}
    self.active_layout = None

def render_layout(self, layout_spec: Dict, panel_layout) -> None:
    """Render the generated layout in Blender"""
    self.active_layout = layout_spec
    
    # Add title
    title_row = panel_layout.row()
    title_row.alignment = 'CENTER'
    title_row.label(text=f"🤖 AI GENERATED UI - {layout_spec['task_type'].replace('_', ' ').title()}")
    
    # Render sections in priority order
    priority_sections = layout_spec.get("priority_order", [])
    rendered_sections = set()
    
    # First, render priority sections
    for priority_section in priority_sections:
        matching_section = self._find_section_by_keyword(layout_spec["sections"], priority_section)
        if matching_section and matching_section["id"] not in rendered_sections:
            self._render_section(matching_section, panel_layout)
            rendered_sections.add(matching_section["id"])
    
    # Then render remaining sections
    for section in layout_spec["sections"]:
        if section["id"] not in rendered_sections:
            self._render_section(section, panel_layout)
    
    # Render adaptive controls
    self._render_adaptive_controls(layout_spec.get("adaptive_controls", []), panel_layout)
    
    # Render real-time elements
    self._render_realtime_elements(layout_spec.get("realtime_elements", []), panel_layout)

def _find_section_by_keyword(self, sections: List[Dict], keyword: str) -> Optional[Dict]:
    """Find section that matches keyword"""
    for section in sections:
        if keyword.lower() in section["id"].lower() or keyword.lower() in section["title"].lower():
            return section
    return None

def _render_section(self, section: Dict, layout) -> None:
    """Render a single section"""
    if section["layout_type"] == "box":
        section_layout = layout.box()
    elif section["layout_type"] == "status_panel":
        section_layout = layout.box()
        section_layout.alert = True  # Make status panels stand out
    else:
        section_layout = layout
    
    # Section header
    header_row = section_layout.row()
    header_row.label(text=section["title"], icon='DISCLOSURE_TRI_DOWN' if section.get("collapsible") else 'NONE')
    
    # Section controls
    for control in section["controls"]:
        self._render_control(control, section_layout)

def _render_control(self, control: Dict, layout) -> None:
    """Render a single control"""
    control_type = control["type"]
    
    if control_type == "realtime_display":
        self._render_realtime_display(control, layout)
    elif control_type == "button_action":
        self._render_button(control, layout)
    elif control["widget"] == "prop":
        self._render_property_control(control, layout)
    elif control["widget"] == "operator":
        self._render_operator_control(control, layout)

def _render_realtime_display(self, control: Dict, layout) -> None:
    """Render real-time data display"""
    display_row = layout.row()
    display_row.alignment = 'CENTER'
    
    # This would connect to actual data sources
    data_source = control.get("data_source", "unknown")
    display_row.label(text=f"📊 {data_source.replace('_', ' ').title()}: Live Data")

def _render_button(self, control: Dict, layout) -> None:
    """Render button control"""
    button_row = layout.row()
    button_row.scale_y = 1.5
    
    # Create dynamic operator if needed
    operator_id = f"llammy.generated_{control['label'].lower().replace(' ', '_')}"
    button_row.operator(operator_id, text=control["label"])

def _render_property_control(self, control: Dict, layout) -> None:
    """Render property-based control"""
    prop_row = layout.row()
    
    # This would need to create dynamic properties
    # For now, show placeholder
    prop_row.label(text=f"{control['label']}: [Dynamic Property]")

def _render_operator_control(self, control: Dict, layout) -> None:
    """Render operator-based control"""
    op_row = layout.row()
    op_row.label(text=f"{control['label']}: [Dynamic Operator]")

def _render_adaptive_controls(self, adaptive_controls: List[Dict], layout) -> None:
    """Render adaptive controls"""
    if not adaptive_controls:
        return
    
    adaptive_box = layout.box()
    adaptive_box.label(text="🧠 ADAPTIVE CONTROLS", icon='AUTO')
    
    for control in adaptive_controls:
        control_row = adaptive_box.row()
        control_row.label(text=f"• {control['type'].replace('_', ' ').title()}")

def _render_realtime_elements(self, realtime_elements: List[Dict], layout) -> None:
    """Render real-time elements"""
    if not realtime_elements:
        return
    
    realtime_box = layout.box()
    realtime_box.label(text="⚡ REAL-TIME STATUS", icon='TIME')
    
    for element in realtime_elements:
        element_row = realtime_box.row()
        element_row.label(text=f"📡 {element['type'].replace('_', ' ').title()}")

Main Generative UI Panel
class LLAMMY_PT_GenerativeUI(bpy.types.Panel):    """AI-Generated UI Panel that adapts to user intent"""    bl_label = "AI-Generated Interface"    bl_idname = "LLAMMY_PT_generative_ui"    bl_space_type = 'VIEW_3D'    bl_region_type = 'UI'    bl_category = 'Llammy'
# Class instances
ui_engine = UIGenerationEngine()
ui_renderer = GenerativeUIRenderer()
last_generated_layout = None

def draw(self, context):
    layout = self.layout
    scene = context.scene
    
    # UI Generation Controls
    gen_box = layout.box()
    gen_box.label(text="🤖 GENERATIVE UI ENGINE", icon='GHOST_ENABLED')
    
    # User input for UI generation
    gen_box.prop(scene, "llammy_user_input", text="Describe Task")
    
    # Generation options
    options_row = gen_box.row()
    options_row.prop(scene, "llammy_ui_complexity", text="Complexity")
    options_row.prop(scene, "llammy_ui_style", text="Style")
    
    # Generate button
    generate_row = gen_box.row()
    generate_row.scale_y = 2.0
    generate_row.operator("llammy.generate_ui", text="🎨 GENERATE UI", icon='ADD')
    
    # Show last generation timestamp
    if hasattr(scene, 'llammy_last_ui_generation'):
        gen_box.label(text=f"Last Generated: {scene.llammy_last_ui_generation}")
    
    layout.separator()
    
    # Render the generated UI if available
    if self.last_generated_layout:
        self.ui_renderer.render_layout(self.last_generated_layout, layout)
    else:
        placeholder_box = layout.box()
        placeholder_box.label(text="💭 Enter task description above to generate custom UI")

Generative UI Operator
class LLAMMY_OT_GenerateUI(bpy.types.Operator):    """Generate custom UI based on user input"""    bl_idname = "llammy.generate_ui"    bl_label = "Generate UI"    bl_description = "Generate adaptive UI based on task description"
def execute(self, context):
    scene = context.scene
    
    user_input = getattr(scene, 'llammy_user_input', '').strip()
    if not user_input:
        self.report({'WARNING'}, "Please describe your task first")
        return {'CANCELLED'}
    
    try:
        # Get available components (you'd connect this to your framework)
        available_components = {"models": {}, "debug": {}, "health": {}}
        
        # Analyze user intent
        intent = LLAMMY_PT_GenerativeUI.ui_engine.analyze_user_intent(user_input, {})
        
        # Generate UI layout
        generated_layout = LLAMMY_PT_GenerativeUI.ui_engine.generate_ui_layout(intent, available_components)
        
        # Store the generated layout
        LLAMMY_PT_GenerativeUI.last_generated_layout = generated_layout
        
        # Update timestamp
        scene.llammy_last_ui_generation = datetime.now().strftime("%H:%M:%S")
        
        # Report success
        task_type = intent["task_type"].replace("_", " ").title()
        complexity = intent["complexity"].title()
        section_count = len(generated_layout["sections"])
        
        self.report({'INFO'}, f"✅ Generated {complexity} {task_type} UI with {section_count} sections")
        
    except Exception as e:
        self.report({'ERROR'}, f"UI generation failed: {str(e)}")
        return {'CANCELLED'}
    
    return {'FINISHED'}

Registration
generative_classes = [    LLAMMY_PT_GenerativeUI,    LLAMMY_OT_GenerateUI,]
def register_generative():    for cls in generative_classes:        bpy.utils.register_class(cls)
# Register properties for UI generation
bpy.types.Scene.llammy_ui_complexity = bpy.props.EnumProperty(
    name="UI Complexity",
    items=[
        ('simple', 'Simple', 'Basic controls only'),
        ('medium', 'Medium', 'Standard interface'),
        ('complex', 'Complex', 'Advanced controls with details')
    ],
    default='medium'
)

bpy.types.Scene.llammy_ui_style = bpy.props.EnumProperty(
    name="UI Style", 
    items=[
        ('minimal', 'Minimal', 'Clean, minimal interface'),
        ('detailed', 'Detailed', 'Rich, detailed interface'),
        ('dashboard', 'Dashboard', 'Data-heavy dashboard style')
    ],
    default='detailed'
)

bpy.types.Scene.llammy_last_ui_generation = bpy.props.StringProperty(
    name="Last Generation Time",
    default=""
)

print("✅ Generative UI system registered")

def unregister_generative():    # Remove properties    props_to_remove = ['llammy_ui_complexity', 'llammy_ui_style', 'llammy_last_ui_generation']    for prop in props_to_remove:        if hasattr(bpy.types.Scene, prop):            delattr(bpy.types.Scene, prop)
for cls in reversed(generative_classes):
    bpy.utils.unregister_class(cls)

print("🔄 Generative UI system unregistered")         
        # Handle result
        if isinstance(result, dict) and result.get('success'):
            scene.llammy_director_response = result.get('creative_response', '')[:800]
            scene.llammy_technical_response = result.get('code', '')
            self.report({'INFO'}, "✅ Pipeline executed successfully")
        else:
            error = result.get('error', 'Unknown error') if isinstance(result, dict) else str(result)
            self.report({'ERROR'}, f"Pipeline failed: {error}")
        
    except Exception as e:
        self.report({'ERROR'}, f"Pipeline execution failed: {str(e)}")
    
    return {'FINISHED'}

Registration
modular_classes = [    LLAMMY_PT_ModularMainPanel,    LLAMMY_OT_DiscoverFramework,    LLAMMY_OT_RunModularPipeline,]
def register_modular():    for cls in modular_classes:        bpy.utils.register_class(cls)    print("✅ Modular UI system registered")
def unregister_modular():    for cls in reversed(modular_classes):        bpy.utils.unregister_class(cls)    print("🔄 Modular UI system unregistered")