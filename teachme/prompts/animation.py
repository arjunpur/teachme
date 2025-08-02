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


def create_enhanced_animation_user_prompt(expanded_prompt, style: str = "light") -> str:
    """Create an enhanced user prompt for animation generation using ExpandedPrompt."""
    from ..models.schemas import ExpandedPrompt
    
    style_colors = {
        "light": "light background with dark text and colorful elements",
        "dark": "dark background with light text and bright colorful elements"
    }
    
    style_description = style_colors.get(style, "light background with dark text and colorful elements")
    
    # Format animation sequence
    sequence_text = ""
    for step in expanded_prompt.animation_sequence:
        sequence_text += f"\nStep {step.step_number}: {step.visual_description}\nKey insight: {step.key_insight}\n"
    
    # Format text overlays
    text_overlays = ""
    for text in expanded_prompt.explanatory_text:
        text_overlays += f'\n"{text.text}" - {text.timing_description}'
    
    # Format quality requirements
    quality_requirements = ""
    for requirement in expanded_prompt.quality_checklist:
        quality_requirements += f"\n- {requirement}"
    
    return f"""Create a Manim animation with this detailed specification:

OBJECTIVE: {expanded_prompt.learning_objective}

CONCEPTS: {', '.join(expanded_prompt.key_concepts)}

VISUAL STRATEGY: {expanded_prompt.visual_strategy}

ANIMATION SEQUENCE:{sequence_text}

TEXT TO INCLUDE:{text_overlays}

QUALITY REQUIREMENTS:{quality_requirements}

CRITICAL: Ensure no text overlaps with visual elements. Position all labels and equations in clear, unobstructed areas. Use appropriate timing so text appears synchronized with relevant visuals.

Additional Requirements:
- Animation duration: 20-30 seconds
- Use {style_description}
- Include clear visual transitions
- Add descriptive comments in the code
- Make it intuitive for beginners
- Use Manim Community Edition syntax (import from manim import *)
- Follow the step-by-step sequence precisely
- Include all specified text overlays with proper timing
- Meet all quality requirements listed above

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