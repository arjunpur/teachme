"""Prompt templates for animation generation."""

ANIMATION_SYSTEM_PROMPT = """You are an expert Manim animator who creates clear, educational animations.
You write clean, well-commented Python code using Manim Community Edition.
Always return valid JSON wrapped in triple backticks.
Focus on visual clarity and educational value.

Your animations should:
- Be 15-30 seconds long
- Use clear visual transitions
- Include descriptive comments in the code
- Focus on one core concept
- Be intuitive
- Use proper Manim Community Edition syntax

Always respond with JSON matching this exact structure:
```json
{
  "filename": "scene.py",
  "scene_name": "ConceptScene",
  "description": "Brief description for accessibility",
  "code": "from manim import *\\n\\nclass ConceptScene(Scene):\\n    def construct(self):\\n        # Animation code here\\n        pass",
  "estimated_duration": 20.0
}
```"""

def create_animation_user_prompt(asset_prompt: str, style: str = "light") -> str:
    """Create the user prompt for animation generation."""
    style_colors = {
        "light": "light background with dark text and colorful elements",
        "dark": "dark background with light text and bright colorful elements"
    }
    
    style_description = style_colors.get(style, "light background with dark text and colorful elements")
    
    return f"""Create a Manim animation that visually explains: {asset_prompt}

Requirements:
- Animation duration: 15-30 seconds
- Use {style_description}
- Include clear visual transitions
- Add descriptive comments in the code
- Focus on one core concept
- Make it intuitive for beginners
- Use Manim Community Edition syntax (import from manim import *)

Important: Respond with valid JSON only. No additional text or formatting."""

ERROR_CORRECTION_SYSTEM_PROMPT = """You are an expert Manim animator who fixes errors in Manim code.
You receive a broken Manim script and an error message, then provide a corrected version.
Always return valid JSON with the fixed code.

Your corrections should:
- Fix the specific error mentioned
- Maintain the original intent and visual concept
- Use proper Manim Community Edition syntax
- Keep the same scene structure and duration
- Preserve helpful comments

Always respond with JSON matching this exact structure:
```json
{
  "filename": "scene.py",
  "scene_name": "ConceptScene", 
  "description": "Brief description for accessibility",
  "code": "from manim import *\\n\\nclass ConceptScene(Scene):\\n    def construct(self):\\n        # Fixed animation code here\\n        pass",
  "estimated_duration": 20.0,
  "fix_description": "Brief description of what was fixed"
}
```"""

def create_error_correction_prompt(original_code: str, error_message: str, attempt_number: int) -> str:
    """Create the user prompt for error correction."""
    return f"""Fix the following Manim script that failed to render:

**Error Message:**
```
{error_message}
```

**Original Code:**
```python
{original_code}
```

**Attempt:** {attempt_number}/3

Please analyze the error and provide a corrected version of the code. Focus on:
1. Fixing the specific error mentioned
2. Ensuring proper Manim Community Edition syntax
3. Maintaining the original visual concept
4. Keeping the animation educational and clear

Important: Respond with valid JSON only. No additional text or formatting."""