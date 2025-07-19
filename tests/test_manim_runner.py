"""Test the Manim runner utility."""

import pytest
from teachme.utils.manim_runner import ManimRunner


def test_manim_installation_check():
    """Test that Manim installation can be checked."""
    runner = ManimRunner()
    is_installed, version_info = runner.check_manim_installation()
    
    assert is_installed is True
    assert "Manim Community" in version_info


def test_code_validation_valid():
    """Test validation of valid Manim code."""
    runner = ManimRunner()
    
    valid_code = '''
from manim import *

class TestScene(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
        self.wait()
'''
    
    is_valid, error_msg = runner.validate_code(valid_code)
    assert is_valid is True
    assert error_msg is None


def test_code_validation_invalid():
    """Test validation of invalid/dangerous code."""
    runner = ManimRunner()
    
    dangerous_code = '''
import os
from manim import *

class TestScene(Scene):
    def construct(self):
        os.system("rm -rf /")  # This should be caught
'''
    
    is_valid, error_msg = runner.validate_code(dangerous_code)
    assert is_valid is False
    assert "os" in error_msg.lower()


def test_scene_name_extraction():
    """Test extraction of scene name from code."""
    runner = ManimRunner()
    
    code = '''
from manim import *

class MyTestScene(Scene):
    def construct(self):
        pass
'''
    
    scene_name = runner.extract_scene_name(code)
    assert scene_name == "MyTestScene"


def test_quality_flags():
    """Test quality flag mapping."""
    runner = ManimRunner()
    
    assert runner._get_quality_flags("low") == ["-ql"]
    assert runner._get_quality_flags("medium") == ["-qm"]
    assert runner._get_quality_flags("high") == ["-qh"]
    assert runner._get_quality_flags("unknown") == ["-ql"]  # Default to low