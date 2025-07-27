# ui_integration.py - Modular UI Hookup System
# Dynamically discovers and connects to any available framework components

import bpy
from typing import Dict, Any, Optional, List
import importlib
import sys

class ModularUIConnector:
    """Discovers and connects to framework components dynamically"""
    
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

# Enhanced UI Panel with Dynamic Discovery
class LLAMMY_PT_ModularMainPanel(bpy.types.Panel):
    """Modular main panel that adapts to available components"""
    bl_label = "Llammy Framework v8.5 - Modular"
    bl_idname = "LLAMMY_PT_modular_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Llammy'
    
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

# Modular operators
class LLAMMY_OT_DiscoverFramework(bpy.types.Operator):
    """Discover framework instance"""
    bl_idname = "llammy.discover_framework"
    bl_label = "Discover Framework"
    bl_description = "Search for framework instance"
    
    def execute(self, context):
        connector = LLAMMY_PT_ModularMainPanel._connector
        framework = connector.discover_framework()
        
        if framework:
            components = connector.discover_components(framework)
            self.report({'INFO'}, f"✅ Found framework with {len(components)} components")
        else:
            self.report({'ERROR'}, "❌ No framework found")
        
        return {'FINISHED'}

class LLAMMY_OT_RunModularPipeline(bpy.types.Operator):
    """Execute pipeline using modular discovery"""
    bl_idname = "llammy.run_modular_pipeline"
    bl_label = "Run Modular Pipeline"
    bl_description = "Execute pipeline with discovered components"
    
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

# Registration
modular_classes = [
    LLAMMY_PT_ModularMainPanel,
    LLAMMY_OT_DiscoverFramework, 
    LLAMMY_OT_RunModularPipeline,
]

def register_modular():
    for cls in modular_classes:
        bpy.utils.register_class(cls)
    print("✅ Modular UI system registered")

def unregister_modular():
    for cls in reversed(modular_classes):
        bpy.utils.unregister_class(cls)
    print("🔄 Modular UI system unregistered")# ui_integration.py - Modular UI Hookup System
# Dynamically discovers and connects to any available framework components

import bpy
from typing import Dict, Any, Optional, List
import importlib
import sys

class ModularUIConnector:
    """Discovers and connects to framework components dynamically"""
    
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

# Enhanced UI Panel with Dynamic Discovery
class LLAMMY_PT_ModularMainPanel(bpy.types.Panel):
    """Modular main panel that adapts to available components"""
    bl_label = "Llammy Framework v8.5 - Modular"
    bl_idname = "LLAMMY_PT_modular_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Llammy'
    
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

# Modular operators
class LLAMMY_OT_DiscoverFramework(bpy.types.Operator):
    """Discover framework instance"""
    bl_idname = "llammy.discover_framework"
    bl_label = "Discover Framework"
    bl_description = "Search for framework instance"
    
    def execute(self, context):
        connector = LLAMMY_PT_ModularMainPanel._connector
        framework = connector.discover_framework()
        
        if framework:
            components = connector.discover_components(framework)
            self.report({'INFO'}, f"✅ Found framework with {len(components)} components")
        else:
            self.report({'ERROR'}, "❌ No framework found")
        
        return {'FINISHED'}

class LLAMMY_OT_RunModularPipeline(bpy.types.Operator):
    """Execute pipeline using modular discovery"""
    bl_idname = "llammy.run_modular_pipeline"
    bl_label = "Run Modular Pipeline"
    bl_description = "Execute pipeline with discovered components"
    
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

# Registration
modular_classes = [
    LLAMMY_PT_ModularMainPanel,
    LLAMMY_OT_DiscoverFramework, 
    LLAMMY_OT_RunModularPipeline,
]

def register_modular():
    for cls in modular_classes:
        bpy.utils.register_class(cls)
    print("✅ Modular UI system registered")

def unregister_modular():
    for cls in reversed(modular_classes):
        bpy.utils.unregister_class(cls)
    print("🔄 Modular UI system unregistered")