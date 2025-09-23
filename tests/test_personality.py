import pytest
from src.personality import AgentPersonality, Archetype, Trait, AgentTrait
from pydantic import ValidationError

def test_personality_serialization(): 
    """Test that personality converts to valid system prompt."""
    personality = AgentPersonality(
        name="TestBot",
        gender="female",
        primary_archetype=Archetype.ANALYTICAL,
        primary_weight=1.0,
        job_description="Test assistant"
    )

    prompt = personality.to_system_prompt()

    # Basic assertions
    assert personality.name == "TestBot"
    assert "you are testbot" in prompt.lower()
    assert "precise, logical" in prompt.lower()
    assert "test assistant" in prompt.lower()

def test_personality_validation(): 
    """Test personality validation rules."""
    # Should fail: weights don't sum to 1.0
    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        AgentPersonality(
            name="BadBot",
            gender="male",
            primary_archetype=Archetype.DRIVER,
            primary_weight=0.7,
            secondary_archetype=Archetype.EXPRESSIVE,
            secondary_weight=0.4, # Wrong: 0.7 + 0.4 = 1.1
            job_description="Bad assistant"
        )

def test_trait_limits(): 
    """Test trait validation (max 5)."""
    too_many_traits = {f"trait{i}": AgentTrait(trait=Trait.EMPATHY, intensity=50) for i in range(6)}
    with pytest.raises(ValidationError): 
        AgentPersonality(
            name="TooMany",
            gender="male",
            primary_archetype=Archetype.AMIABLE,
            job_description="Overloaded assistant",
            traits=too_many_traits,
            primary_weight=1.0 # Add missing required field
        )