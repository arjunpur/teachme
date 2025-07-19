"""Safe Manim execution utility."""

import ast
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ManimRunner:
    """Safely execute Manim scripts with resource limits."""
    
    def __init__(self, timeout: int = 180):
        """Initialize the Manim runner."""
        self.timeout = timeout
    
    def validate_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate Python code for safety."""
        try:
            # Parse the code to check for syntax errors
            tree = ast.parse(code)
            
            # Check for potentially dangerous operations
            dangerous_nodes = []
            
            for node in ast.walk(tree):
                # Check for dangerous imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ['os', 'subprocess', 'sys', 'shutil']:
                            dangerous_nodes.append(f"Import of '{alias.name}' module")
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module in ['os', 'subprocess', 'sys', 'shutil']:
                        dangerous_nodes.append(f"Import from '{node.module}' module")
                
                # Check for file operations
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['open', 'exec', 'eval', '__import__']:
                            dangerous_nodes.append(f"Call to '{node.func.id}' function")
            
            if dangerous_nodes:
                return False, f"Potentially dangerous operations detected: {', '.join(dangerous_nodes)}"
            
            return True, None
            
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Code validation error: {e}"
    
    def extract_scene_name(self, code: str) -> Optional[str]:
        """Extract the main Scene class name from the code."""
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if class inherits from Scene or a Manim scene class
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id in ['Scene', 'MovingCameraScene', 'ThreeDScene']:
                            return node.name
                        elif isinstance(base, ast.Attribute) and base.attr in ['Scene', 'MovingCameraScene', 'ThreeDScene']:
                            return node.name
            
            return None
            
        except Exception:
            return None
    
    async def render_animation(
        self,
        code: str,
        scene_name: str,
        quality: str = "low",
        output_dir: Path = None
    ) -> Tuple[bool, Optional[Path], Optional[str]]:
        """Render a Manim animation safely."""
        
        # Validate the code first
        is_valid, error_msg = self.validate_code(code)
        if not is_valid:
            return False, None, error_msg
        
        # Create temporary directory for rendering
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            script_path = temp_path / "scene.py"
            
            # Write the script
            script_path.write_text(code)
            
            # Determine quality flags
            quality_flags = self._get_quality_flags(quality)
            
            # Construct manim command
            cmd = [
                "manim",
                str(script_path),
                scene_name,
                *quality_flags,
                "--output_file", f"{scene_name}.mp4"
            ]
            
            try:
                # Run manim with timeout
                result = subprocess.run(
                    cmd,
                    cwd=temp_path,
                    timeout=self.timeout,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    return False, None, f"Manim error: {result.stderr}"
                
                # Find the generated video file
                media_dir = temp_path / "media"
                video_files = list(media_dir.rglob("*.mp4"))
                
                if not video_files:
                    return False, None, "No video file generated"
                
                video_file = video_files[0]
                
                # Move to output directory if specified
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    final_path = output_dir / f"{scene_name}.mp4"
                    video_file.rename(final_path)
                    return True, final_path, None
                else:
                    # Return path in temp directory (caller should copy)
                    return True, video_file, None
                
            except subprocess.TimeoutExpired:
                return False, None, f"Manim rendering timed out after {self.timeout} seconds"
            except Exception as e:
                return False, None, f"Manim execution error: {e}"
    
    def _get_quality_flags(self, quality: str) -> List[str]:
        """Get manim quality flags based on quality setting."""
        quality_map = {
            "low": ["-ql"],      # Low quality, fast render
            "medium": ["-qm"],   # Medium quality
            "high": ["-qh"]      # High quality
        }
        return quality_map.get(quality, ["-ql"])
    
    def check_manim_installation(self) -> Tuple[bool, Optional[str]]:
        """Check if Manim is properly installed."""
        try:
            result = subprocess.run(
                ["manim", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                return False, "Manim not found or not working"
                
        except subprocess.TimeoutExpired:
            return False, "Manim version check timed out"
        except FileNotFoundError:
            return False, "Manim not installed"
        except Exception as e:
            return False, f"Error checking Manim installation: {e}"