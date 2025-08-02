#!/usr/bin/env python3
"""Example of the simplified animation system usage."""

import asyncio
from pathlib import Path
from teachme.models.schemas import AnimationRequest
from teachme.agents.animation import ManimCodeGenerator

async def main():
    """Demonstrate the simplified animation system."""
    
    # Create output directory
    output_dir = Path("./test_output")
    
    # Initialize the generator
    generator = ManimCodeGenerator(output_dir=output_dir, verbose=True)
    
    # Example 1: Enhanced prompt (default behavior)
    enhanced_request = AnimationRequest(
        user_prompt="Explain how quadratic equations work with visual examples",
        enhance=True,  # This is the default
        style="light"
    )
    
    print("🧠 Enhanced Request:")
    print(f"  User prompt: {enhanced_request.user_prompt}")
    print(f"  Should enhance: {enhanced_request.should_enhance()}")
    print(f"  Style: {enhanced_request.style}")
    
    # Example 2: Simple prompt (skipping enhancement)
    simple_request = AnimationRequest(
        user_prompt="Create a simple animation showing a circle moving in a square",
        enhance=False,  # Skip subject matter enhancement
        style="dark"
    )
    
    print("\n📝 Simple Request:")
    print(f"  User prompt: {simple_request.user_prompt}")
    print(f"  Should enhance: {simple_request.should_enhance()}")
    print(f"  Style: {simple_request.style}")
    
    # Note: We're not actually calling generate() here to avoid needing API keys
    # But the interface would be:
    # result = await generator.generate(enhanced_request.model_dump())
    
    print("\n✅ Simplified system is ready to use!")
    print("   - Single AnimationRequest class with enhance flag")
    print("   - Single code path in ManimCodeGenerator")
    print("   - Direct retry logic without complex managers")
    print("   - Unified _create_prompt() function handles both cases")

if __name__ == "__main__":
    asyncio.run(main())