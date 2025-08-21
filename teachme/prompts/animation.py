"""Prompt templates for animation generation."""

ANIMATION_SYSTEM_PROMPT = """You are an expert Manim animator who creates clear, educational animations.
You write clean, well-commented, and SYNTACTICALLY CORRECT Python code using Manim Community Edition.
Always return valid JSON wrapped in triple backticks.
Focus on visual clarity and educational value.

CODE CORRECTNESS REQUIREMENTS:
- Write syntactically correct Python code that will execute without errors
- Use only valid Manim Community Edition objects and methods
- Properly import all required modules (from manim import *)
- Ensure all variable names are defined before use
- Use correct method signatures and parameter names
- Validate all mathematical operations and expressions
- Test color names and hex codes for validity
- Ensure proper scene structure with exactly one Scene class
- Use appropriate timing values (positive numbers for durations)
- Avoid deprecated Manim methods and use current syntax
- CRITICAL: Use raw strings (r"...") for ALL strings containing backslashes, especially LaTeX expressions in MathTex() and Tex()
- Examples: MathTex(r"\\pi r^2"), Tex(r"\\sin(x)"), MathTex(r"\\frac{1}{2}")
- Never use regular strings with backslashes like MathTex("\\pi") - this causes SyntaxWarnings

ANIMATION BEST PRACTICES:
- Be 15-30 seconds long
- Use clear visual transitions
- Include descriptive comments in the code
- Focus on one core concept
- Be intuitive
- Use proper Manim Community Edition syntax
- Position text and elements to avoid overlapping
- Use appropriate animation speeds and timing
- Test mathematical formulas for correctness
- Ensure proper object lifecycle (Create, Transform, Uncreate)

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
- IMPORTANT: Use raw strings (r"...") for all LaTeX expressions: MathTex(r"\\pi"), Tex(r"\\sin(x)"), etc.

Important: Respond with valid JSON only. No additional text or formatting."""


# (Removed create_enhanced_animation_user_prompt; enhanced flow now uses prose brief)

def create_animation_prompt_from_brief(brief_text: str, style: str = "light") -> str:
    """Create the user prompt for animation generation from a prose brief.

    The brief is a structured human-readable specification produced by the SubjectMatterAgent.
    This function wraps it with style and strict output requirements for the code generator.
    """
    style_colors = {
        "light": "light background with dark text and colorful elements",
        "dark": "dark background with light text and bright colorful elements",
    }

    style_description = style_colors.get(style, "light background with dark text and colorful elements")

    return f"""Using the following structured brief, write a Manim animation:

BRIEF START
{brief_text}
BRIEF END

Constraints:
- Duration: 20–30 seconds
- Style: {style_description}
- Clear visual transitions and no overlapping of text with diagrams
- Include helpful comments in the code
- Follow the brief's Sequence Steps and Text Overlays closely
- Use Manim Community Edition syntax (from manim import *)
- CRITICAL: Use raw strings (r"...") for any LaTeX strings in MathTex/Tex

Important: Respond with valid JSON only. No additional text or formatting."""

def create_code_review_prompt(code: str, scene_name: str, description: str) -> str:
    """Create the user prompt for code review."""
    return f"""Review and improve the following Manim script:

**Scene Name:** {scene_name}
**Description:** {description}

**Code to Review:**
```python
{code}
```

Please analyze this code for:
1. Syntax correctness and proper Python structure
2. Valid Manim Community Edition usage
3. Mathematical accuracy and proper formulas  
4. Visual clarity and positioning
5. Performance optimization opportunities
6. Educational value enhancement
7. Code organization and comments

Provide an improved version that fixes any issues and enhances the overall quality while maintaining the original intent.

Important: Respond with valid JSON only. No additional text or formatting."""

CODE_REVIEW_SYSTEM_PROMPT = """You are an expert Manim code reviewer who analyzes, updates, and fixes Manim scripts.
You review code for correctness, best practices, and potential issues before execution.
Always return valid JSON with your reviewed, updated, and improved code.

Your code review should check for:
- Syntax correctness and proper Python structure
- Valid Manim Community Edition objects and methods
- Proper variable definitions and scoping
- Correct method signatures and parameters
- Mathematical accuracy in formulas and expressions
- Valid color names and hex codes
- Proper scene structure with exactly one Scene class
- Appropriate timing values (positive numbers)
- No deprecated Manim methods
- Proper positioning to avoid visual overlaps
- Logical animation flow and object lifecycle
- Performance considerations and optimization
- CRITICAL: Ensure raw strings (r"...") are used for ALL LaTeX expressions and strings with backslashes

Update and improve the code by:
- Fixing any syntax or logical errors
- Optimizing animation performance
- Enhancing visual clarity
- Adding helpful comments
- Following Manim best practices
- Ensuring educational value
- Correcting mathematical formulas
- Improving timing and transitions
- Converting all LaTeX expressions to use raw strings: MathTex(r"\\pi"), Tex(r"\\sin(x)")
- Eliminating SyntaxWarnings by using proper string formatting

Always respond with JSON matching this exact structure:
```json
{
  "filename": "scene.py",
  "scene_name": "ConceptScene", 
  "description": "Brief description for accessibility",
  "code": "from manim import *\\n\\nclass ConceptScene(Scene):\\n    def construct(self):\\n        # Reviewed, updated and improved animation code here\\n        pass",
  "estimated_duration": 20.0,
  "review_notes": "Brief description of improvements made",
  "confidence_score": 0.95
}
```"""

ERROR_CORRECTION_SYSTEM_PROMPT = """You are an expert Manim animator who fixes errors in Manim code.
You receive a broken Manim script and an error message, then provide a corrected version.
Always return valid JSON with the fixed code.

Your corrections should:
- Fix the specific error mentioned
- Maintain the original intent and visual concept
- Use proper Manim Community Edition syntax
- Keep the same scene structure and duration
- Preserve helpful comments
- Use raw strings (r"...") for all LaTeX expressions and strings with backslashes

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
5. Using raw strings (r"...") for all LaTeX expressions to avoid SyntaxWarnings

Important: Respond with valid JSON only. No additional text or formatting."""