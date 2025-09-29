import pytest
from personality import AgentPersonality, Archetype, Trait, AgentTrait
from pydantic import ValidationError
import logging  # For synergy warning checks

def test_personality_serialization():
    """Test that personality converts to valid system prompt."""
    traits = [AgentTrait(trait=Trait.EMPATHY, intensity=70), AgentTrait(trait=Trait.WIT, intensity=50)]
    personality = AgentPersonality(
        name="TestBot",
        gender="female",
        primary_archetype=Archetype.ANALYTICAL,
        primary_weight=1.0,
        job_description="Test assistant",
        traits=traits
    )
    prompt = personality.system_prompt  # Use cached property
    assert "TestBot" in prompt
    assert "precise, logical" in prompt.lower()
    assert "test assistant" in prompt.lower()
    assert "empathy (intensity: 70%)" in prompt.lower()  # Updated for lowercase
    assert "wit (intensity: 50%)" in prompt.lower()
    # Check caching
    assert personality.system_prompt is personality.system_prompt  # Same object (cached)

def test_personality_validation():
    """Test personality validation rules."""
    # Weights don't sum to 1.0 (custom validator raises ValueError after fields pass)
    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        AgentPersonality(
            name="BadBot",
            gender="male",
            primary_archetype=Archetype.DRIVER,
            primary_weight=0.7,
            secondary_archetype=Archetype.EXPRESSIVE,
            secondary_weight=0.4,  # 0.7 + 0.4 = 1.1
            job_description="Bad assistant"
        )
    
    # Primary <0.5 with secondary (Pydantic field validation raises first)
    with pytest.raises(ValidationError) as exc_info:
        AgentPersonality(
            name="WeakPrimary",
            gender="unspecified",  # Updated to valid value
            primary_archetype=Archetype.AMIABLE,
            primary_weight=0.4,
            secondary_archetype=Archetype.ANALYTICAL,
            job_description="Assistant"
        )
    assert "greater than or equal to 0.5" in str(exc_info.value)  # Matches Pydantic message
    
    # Auto-fill secondary_weight
    p = AgentPersonality(
        name="AutoFill",
        gender="unspecified",
        primary_archetype=Archetype.EXPRESSIVE,
        primary_weight=0.6,
        secondary_archetype=Archetype.DRIVER,
        job_description="Assistant"
    )
    assert p.secondary_weight == 0.4

def test_trait_limits_and_uniqueness():
    """Test trait validation (max 5, uniqueness)."""
    # Too many
    too_many_traits = [AgentTrait(trait=Trait.EMPATHY, intensity=50) for _ in range(6)]
    with pytest.raises(ValueError, match="Max 5 traits"):
        AgentPersonality(
            name="TooMany",
            gender="male",
            primary_archetype=Archetype.AMIABLE,
            primary_weight=1.0,
            job_description="Overloaded assistant",
            traits=too_many_traits
        )
    
    # Duplicates
    dup_traits = [AgentTrait(trait=Trait.EMPATHY, intensity=50), AgentTrait(trait=Trait.EMPATHY, intensity=60)]
    with pytest.raises(ValueError, match="Duplicate trait"):
        AgentPersonality(
            name="DupTraits",
            gender="female",
            primary_archetype=Archetype.DRIVER,
            primary_weight=1.0,
            job_description="Assistant",
            traits=dup_traits
        )

def test_get_manifestation():
    """Test dynamic phrasing for trait manifestation."""
    p = AgentPersonality(name="Dummy", gender="unspecified", primary_archetype=Archetype.DRIVER, primary_weight=1.0, job_description="Test")
    assert "sparingly" in p._get_manifestation(Trait.CURIOSITY, 20)
    assert "occasionally" in p._get_manifestation(Trait.CURIOSITY, 40)
    assert "frequently" in p._get_manifestation(Trait.CURIOSITY, 80)
    assert "always prominently" in p._get_manifestation(Trait.CURIOSITY, 100)

def test_trait_synergies(caplog):
    """Test synergy warnings for traits and archetypes."""
    caplog.set_level(logging.WARNING)
    # Trigger Amiable warmth tip
    traits = [AgentTrait(trait=Trait.WARMTH, intensity=40)]  # Low intensity
    AgentPersonality(
        name="SynergyTest",
        gender="unspecified",
        primary_archetype=Archetype.AMIABLE,
        primary_weight=1.0,
        job_description="Test",
        traits=traits
    )
    assert "Boost 'warmth' >=50 for Amiable" in caplog.text  # Check warning fired
    
    # Add similar checks for other synergies if needed