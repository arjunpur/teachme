# TeachMe

Convert natural language prompts into educational content with animations.

## M1: Animate Command ✅

Generate Manim animations from natural language prompts.

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd teachme

# Install dependencies with uv
uv sync

# Configure your OpenAI API key
cp .env.example .env  # Copy template
# Edit .env and add your OpenAI API key

# Verify installation
uv run teachme --help
```

### Configuration

Create a `.env` file in the project root with your OpenAI API key:

```bash
# TeachMe Environment Variables
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Override default model
# OPENAI_MODEL=gpt-4o-mini

# Optional: Override default temperature  
# OPENAI_TEMPERATURE=0.7
```

### Usage

#### Basic Animation Generation

```bash
# Generate a simple animation (uses .env file)
uv run teachme animate "explain the unit circle and sine wave relationship"

# Or set API key via environment variable
export OPENAI_API_KEY="your-api-key-here"
uv run teachme animate "explain the unit circle and sine wave relationship"

# With options
uv run teachme animate "visualize how derivatives work" \
  --style dark \
  --quality medium \
  --output-dir ./my-animations \
  --verbose
```

#### Command Options

```bash
uv run teachme animate --help
```

- `--style` - Visual style (`light` or `dark`, default: `light`)
- `--output-dir` - Output directory (default: `./outputs`)
- `--quality` - Video quality (`low`, `medium`, `high`, default: `low`)
- `--verbose` - Show detailed progress
- `--api-key` - OpenAI API key (overrides environment variable)

### Output

Generated animations are saved to:
```
outputs/
└── animations/
    └── ConceptScene.mp4
```

### Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test files
uv run pytest tests/test_manim_runner.py -v
uv run pytest tests/test_integration.py -v
```

### Development Status

- ✅ **M1: Animate Command** - Complete
- 🔄 **M2: Markdown Wrapper** - Next milestone
- ⏳ **M3: Outline & Explanatory Text** - Planned
- ⏳ **M4: Diagram Generation** - Planned

### Requirements

- Python ≥ 3.13
- OpenAI API key
- Manim Community Edition (installed automatically)
