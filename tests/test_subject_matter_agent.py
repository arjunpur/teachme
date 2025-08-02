"""Tests for SubjectMatterAgent."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from teachme.agents.subject_matter import SubjectMatterAgent
from teachme.models.schemas import SubjectMatterInput, ExpandedPrompt, AnimationStep, TextOverlay
from teachme.utils.llm_client import LLMClient


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = Mock(spec=LLMClient)
    client.verbose = False
    return client


@pytest.fixture
def subject_matter_agent(tmp_path, mock_llm_client):
    """Create a SubjectMatterAgent for testing."""
    return SubjectMatterAgent(output_dir=tmp_path, llm_client=mock_llm_client)


@pytest.fixture
def sample_content_analysis():
    """Sample content analysis response."""
    return {
        "learning_objective": "Understand how derivatives represent rates of change",
        "key_concepts": ["rate of change", "slope", "tangent line", "limit"],
        "prerequisite_knowledge": ["functions", "graphs"],
        "common_misconceptions": ["derivatives are just slopes", "derivatives only work for lines"],
        "difficulty_level": "intermediate"
    }


@pytest.fixture
def sample_visual_planning():
    """Sample visual planning response."""
    return {
        "visual_strategy": "Start with a car moving along a curved path, show how speed changes",
        "visual_metaphors": ["car on curved road", "tangent line as instantaneous direction"],
        "color_scheme": {
            "primary_concept": "blue",
            "secondary_elements": "gray", 
            "highlighting": "red"
        },
        "progression_strategy": "Concrete to abstract: car → position function → derivative",
        "misconception_corrections": ["Show derivative at a point, not over interval"],
        "key_visual_techniques": ["zoom in animation", "color highlighting", "step-by-step build"]
    }


@pytest.fixture
def sample_sequence_data():
    """Sample sequence generation response."""
    return {
        "animation_sequence": [
            {
                "step_number": 1,
                "visual_description": "Show a car moving along a curved path",
                "explanation": "Establish the concrete example",
                "key_insight": "Motion along curves has varying speed"
            },
            {
                "step_number": 2,
                "visual_description": "Zoom in to show position at specific time",
                "explanation": "Focus attention on single point",
                "key_insight": "We can examine instantaneous behavior"
            }
        ],
        "explanatory_text": [
            {
                "text": "The derivative measures instantaneous rate of change",
                "timing_description": "after step 2",
                "purpose": "Connect visual to mathematical concept"
            }
        ],
        "quality_checklist": [
            "Ensure smooth car animation",
            "Clear text positioning",
            "Synchronized highlighting"
        ],
        "total_estimated_duration": 25.0,
        "pacing_notes": "Allow 3 seconds per step for comprehension"
    }


class TestSubjectMatterAgent:
    """Test cases for SubjectMatterAgent."""
    
    def test_initialization(self, tmp_path, mock_llm_client):
        """Test SubjectMatterAgent initialization."""
        agent = SubjectMatterAgent(output_dir=tmp_path, llm_client=mock_llm_client)
        
        assert agent.output_dir == tmp_path
        assert agent.llm_client == mock_llm_client
        assert agent.output_dir.exists()
    
    def test_initialization_with_defaults(self, tmp_path):
        """Test SubjectMatterAgent initialization with default LLM client."""
        with patch('teachme.agents.subject_matter.LLMClient') as mock_client_class:
            agent = SubjectMatterAgent(output_dir=tmp_path)
            
            mock_client_class.assert_called_once()
            assert agent.output_dir == tmp_path
    
    @pytest.mark.asyncio
    async def test_analyze_content(self, subject_matter_agent, sample_content_analysis):
        """Test content analysis stage."""
        # Mock the LLM response
        subject_matter_agent.llm_client.generate_json_response = AsyncMock()
        
        # Create a mock response object
        mock_response = Mock()
        mock_response.model_dump.return_value = sample_content_analysis
        subject_matter_agent.llm_client.generate_json_response.return_value = mock_response
        
        result = await subject_matter_agent._analyze_content("explain derivatives")
        
        assert result == sample_content_analysis
        assert "learning_objective" in result
        assert "key_concepts" in result
        assert len(result["key_concepts"]) > 0
    
    @pytest.mark.asyncio
    async def test_plan_visuals(self, subject_matter_agent, sample_content_analysis, sample_visual_planning):
        """Test visual planning stage."""
        # Mock the LLM response
        subject_matter_agent.llm_client.generate_json_response = AsyncMock()
        
        # Create a mock response object
        mock_response = Mock()
        mock_response.model_dump.return_value = sample_visual_planning
        subject_matter_agent.llm_client.generate_json_response.return_value = mock_response
        
        result = await subject_matter_agent._plan_visuals(sample_content_analysis)
        
        assert result == sample_visual_planning
        assert "visual_strategy" in result
        assert "visual_metaphors" in result
    
    @pytest.mark.asyncio
    async def test_generate_sequence(self, subject_matter_agent, sample_content_analysis, 
                                   sample_visual_planning, sample_sequence_data):
        """Test sequence generation stage."""
        # Mock the LLM response
        subject_matter_agent.llm_client.generate_json_response = AsyncMock()
        
        # Create mock response objects
        mock_steps = [Mock() for _ in sample_sequence_data["animation_sequence"]]
        for i, step_data in enumerate(sample_sequence_data["animation_sequence"]):
            mock_steps[i].model_dump.return_value = step_data
        
        mock_texts = [Mock() for _ in sample_sequence_data["explanatory_text"]]
        for i, text_data in enumerate(sample_sequence_data["explanatory_text"]):
            mock_texts[i].model_dump.return_value = text_data
        
        mock_response = Mock()
        mock_response.animation_sequence = mock_steps
        mock_response.explanatory_text = mock_texts
        mock_response.model_dump.return_value = sample_sequence_data
        
        subject_matter_agent.llm_client.generate_json_response.return_value = mock_response
        
        result = await subject_matter_agent._generate_sequence(sample_content_analysis, sample_visual_planning)
        
        assert "animation_sequence" in result
        assert "explanatory_text" in result
        assert "quality_checklist" in result
        assert len(result["animation_sequence"]) == 2
    
    @pytest.mark.asyncio
    async def test_full_generate_pipeline(self, subject_matter_agent, sample_content_analysis,
                                        sample_visual_planning, sample_sequence_data):
        """Test the full generate pipeline."""
        # Mock all three stages
        subject_matter_agent._analyze_content = AsyncMock(return_value=sample_content_analysis)
        subject_matter_agent._plan_visuals = AsyncMock(return_value=sample_visual_planning)
        subject_matter_agent._generate_sequence = AsyncMock(return_value=sample_sequence_data)
        
        input_data = {"user_prompt": "explain derivatives"}
        result = await subject_matter_agent.generate(input_data)
        
        # Verify all stages were called
        subject_matter_agent._analyze_content.assert_called_once_with("explain derivatives")
        subject_matter_agent._plan_visuals.assert_called_once_with(sample_content_analysis)
        subject_matter_agent._generate_sequence.assert_called_once_with(sample_content_analysis, sample_visual_planning)
        
        # Verify result structure
        assert "learning_objective" in result
        assert "key_concepts" in result
        assert "visual_strategy" in result
        assert "animation_sequence" in result
        assert "explanatory_text" in result
        assert "quality_checklist" in result
    
    @pytest.mark.asyncio
    async def test_process_with_timeout_success(self, subject_matter_agent):
        """Test process_with_timeout with successful completion."""
        # Mock the generate method
        expected_result = {
            "learning_objective": "Test objective",
            "key_concepts": ["concept1"],
            "visual_strategy": "Test strategy",
            "animation_sequence": [],
            "explanatory_text": [],
            "quality_checklist": []
        }
        
        subject_matter_agent.generate = AsyncMock(return_value=expected_result)
        
        result = await subject_matter_agent.process_with_timeout("test prompt", timeout_seconds=30)
        
        assert isinstance(result, ExpandedPrompt)
        assert result.learning_objective == "Test objective"
        subject_matter_agent.generate.assert_called_once_with({"user_prompt": "test prompt"})
    
    @pytest.mark.asyncio
    async def test_process_with_timeout_failure(self, subject_matter_agent):
        """Test process_with_timeout with timeout."""
        # Mock the generate method to take too long
        async def slow_generate(input_data):
            await asyncio.sleep(2)  # Longer than timeout
            return {}
        
        subject_matter_agent.generate = slow_generate
        
        with pytest.raises(RuntimeError, match="timed out"):
            await subject_matter_agent.process_with_timeout("test prompt", timeout_seconds=0.1)
    
    @pytest.mark.asyncio
    async def test_error_handling_in_generate(self, subject_matter_agent):
        """Test error handling in generate method."""
        # Mock analyze_content to raise an exception
        subject_matter_agent._analyze_content = AsyncMock(side_effect=Exception("LLM error"))
        
        input_data = {"user_prompt": "test prompt"}
        
        with pytest.raises(RuntimeError, match="Subject matter analysis failed"):
            await subject_matter_agent.generate(input_data)
    
    def test_verbose_mode_detection(self, tmp_path):
        """Test verbose mode detection from LLM client."""
        # Test with verbose client
        verbose_client = Mock(spec=LLMClient)
        verbose_client.verbose = True
        
        agent = SubjectMatterAgent(output_dir=tmp_path, llm_client=verbose_client)
        assert agent._is_verbose() is True
        
        # Test with non-verbose client
        non_verbose_client = Mock(spec=LLMClient)
        non_verbose_client.verbose = False
        
        agent = SubjectMatterAgent(output_dir=tmp_path, llm_client=non_verbose_client)
        assert agent._is_verbose() is False


class TestSubjectMatterInputValidation:
    """Test input validation for SubjectMatterAgent."""
    
    def test_valid_subject_matter_input(self):
        """Test valid SubjectMatterInput creation."""
        input_data = SubjectMatterInput(user_prompt="explain calculus")
        assert input_data.user_prompt == "explain calculus"
    
    def test_expanded_prompt_creation(self):
        """Test ExpandedPrompt creation with all fields."""
        steps = [
            AnimationStep(
                step_number=1,
                visual_description="Show a curve",
                explanation="Introduce the concept",
                key_insight="Curves have varying slopes"
            )
        ]
        
        texts = [
            TextOverlay(
                text="This is a curve",
                timing_description="during step 1",
                purpose="Label the visual"
            )
        ]
        
        expanded = ExpandedPrompt(
            learning_objective="Understand curves",
            key_concepts=["curve", "slope"],
            visual_strategy="Show progression from simple to complex",
            animation_sequence=steps,
            explanatory_text=texts,
            quality_checklist=["Clear visuals", "Proper timing"]
        )
        
        assert expanded.learning_objective == "Understand curves"
        assert len(expanded.key_concepts) == 2
        assert len(expanded.animation_sequence) == 1
        assert len(expanded.explanatory_text) == 1
        assert len(expanded.quality_checklist) == 2