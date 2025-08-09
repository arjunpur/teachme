"""Prompt templates for subject matter analysis and educational planning."""

# Stage 1: Content Analysis
CONTENT_ANALYSIS_SYSTEM_PROMPT = """You are an expert educational content analyst who identifies core concepts and learning objectives from user prompts.

Your role is to analyze a user's request and determine:
1. The fundamental learning objective (what should the student understand?)
2. The 3-5 most important concepts to cover
3. Any prerequisite knowledge assumptions
4. Potential misconceptions to address

Always respond with valid JSON matching this exact structure:
```json
{
  "learning_objective": "Clear, specific statement of what the viewer should understand after watching",
  "key_concepts": ["concept1", "concept2", "concept3"],
  "prerequisite_knowledge": ["prereq1", "prereq2"],
  "common_misconceptions": ["misconception1", "misconception2"],
  "difficulty_level": "beginner|intermediate|advanced"
}
```"""

def create_content_analysis_prompt(user_prompt: str) -> str:
    """Create prompt for content analysis stage."""
    return f"""Analyze this educational request and identify the core learning elements:

"{user_prompt}"

Your task is to:
1. Define a clear, specific learning objective
2. Identify 3-5 key concepts that must be covered
3. List any prerequisite knowledge assumptions
4. Identify common misconceptions students have about this topic
5. Assess the difficulty level

Focus on educational clarity and building intuitive understanding.
Respond with valid JSON only."""

# Stage 2: Visual Planning
VISUAL_PLANNING_SYSTEM_PROMPT = """You are an expert educational animator who designs visualization strategies for mathematical and scientific concepts.

Your role is to plan how concepts should be visualized, including:
1. Overall visual metaphors and analogies
2. Color coding and visual organization
3. The progression from concrete to abstract
4. How to address misconceptions visually
5. Specific visual techniques for maximum clarity

Always respond with valid JSON matching this exact structure:
```json
{
  "visual_strategy": "Overall approach to visualization",
  "visual_metaphors": ["metaphor1", "metaphor2"],
  "color_scheme": {
    "primary_concept": "color",
    "secondary_elements": "color",
    "highlighting": "color"
  },
  "progression_strategy": "How to build from simple to complex",
  "misconception_corrections": ["How to visually show what the concept is NOT"],
  "key_visual_techniques": ["technique1", "technique2"]
}
```"""

def create_visual_planning_prompt(content_analysis: dict) -> str:
    """Create prompt for visual planning stage."""
    return f"""Design a comprehensive visualization strategy for this educational content:

**Learning Objective:** {content_analysis['learning_objective']}
**Key Concepts:** {', '.join(content_analysis['key_concepts'])}
**Difficulty Level:** {content_analysis['difficulty_level']}
**Common Misconceptions:** {', '.join(content_analysis['common_misconceptions'])}

Your task is to:
1. Develop an overall visual strategy that builds intuition
2. Choose appropriate visual metaphors and analogies
3. Design a color scheme that enhances understanding
4. Plan the progression from concrete examples to abstract concepts
5. Design visual techniques to explicitly address misconceptions
6. Recommend specific visual techniques for maximum educational impact

Focus on clarity, intuition-building, and preventing confusion.
Respond with valid JSON only."""

# Stage 3: Sequence Generation  
SEQUENCE_GENERATION_SYSTEM_PROMPT = """You are an expert educational sequence designer who creates step-by-step animation breakdowns.

Your role is to create a detailed sequence that:
1. Builds understanding methodically, step by step
2. Uses effective pacing and timing
3. Includes specific text overlays and their timing
4. Provides quality requirements for flawless execution
5. Ensures exceptional educational clarity

Always respond with valid JSON matching this exact structure:
```json
{
  "animation_sequence": [
    {
      "step_number": 1,
      "visual_description": "Detailed description of what appears on screen",
      "explanation": "Why this step builds understanding",
      "key_insight": "Main takeaway from this step"
    }
  ],
  "explanatory_text": [
    {
      "text": "Exact text to display",
      "timing_description": "When to show (e.g., 'during step 2')",
      "purpose": "Why this text is needed"
    }
  ],
  "quality_checklist": [
    "Specific requirement 1",
    "Specific requirement 2"
  ],
  "total_estimated_duration": 25.0,
  "pacing_notes": "Guidance on timing and rhythm"
}
```"""

def create_sequence_generation_prompt(content_analysis: dict, visual_planning: dict) -> str:
    """Create prompt for sequence generation stage."""
    return f"""Create a detailed step-by-step animation sequence for this educational content:

**Learning Objective:** {content_analysis['learning_objective']}
**Key Concepts:** {', '.join(content_analysis['key_concepts'])}
**Visual Strategy:** {visual_planning['visual_strategy']}
**Visual Metaphors:** {', '.join(visual_planning['visual_metaphors'])}
**Misconceptions to Address:** {', '.join(content_analysis['common_misconceptions'])}

**Animation Quality Requirements:**

**Educational Excellence:**
- Build intuition first with concrete, familiar examples before abstract concepts
- Methodical progression where each step clearly follows from the previous
- Show the concept from multiple perspectives (algebraic, geometric, practical)
- Connect to fundamentals and previously established knowledge
- Explicitly show what the concept is NOT, not just what it is

**Visual Polish & Bug Prevention:**
- Ensure all text is clearly readable and never overlaps with diagrams
- Synchronize text appearance with relevant visual elements
- Use high contrast colors for accessibility and clarity
- Smooth transitions, not jarring or too fast to follow
- Efficient use of screen space - no cramped or cluttered visuals

**Technical Excellence:**
- Double-check all formulas, calculations, and visual representations
- Use standard mathematical notation throughout
- Objects appropriately sized relative to each other
- Clean rendering without visual artifacts or glitches
- All axes, variables, and key elements clearly labeled

**Pedagogical Structure:**
- Clear narrative arc with beginning, middle, and end
- Appropriate pacing allowing time to process each concept
- Use highlighting, zooming, or color changes to direct attention
- End with clear recap of the main insight
- Standalone clarity without external explanation needed

Your task is to:
1. Break down the animation into 5-8 clear steps
2. Specify exact text overlays and when they should appear
3. Create a comprehensive quality checklist with specific requirements
4. Ensure the sequence builds understanding methodically
5. Target 20-30 seconds total duration

Focus on exceptional educational clarity and zero visual bugs.
Respond with valid JSON only."""