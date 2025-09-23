import enum
from pydantic import BaseModel, Field, model_validator
from typing import Dict, Callable, Any, List, Optional, Literal

# Enum for archetypes with docstrings for Intellisense
class Archetype(str, enum.Enum):
    """DISC-based archetypes for AI Agent personalities. Hover for details."""
    
    DRIVER = "driver"
    """Dominance (D): Direct, decisive, results-oriented. Prioritize action and control."""

    EXPRESSIVE = "expressive"
    """Influence (I): Enthusiastic, persuasive, creative. Focus on relationships and inspiration."""
    
    AMIABLE = "amiable"
    """Steadiness (S): Patient, supportive, reliable. Emphasize harmony and collaboration."""
    
    ANALYTICAL = "analytical"
    """Conscientiousness (C): Precise, logical, detail-focused. Stress accuracy and procedures."""

class Trait(str, enum.Enum):
    """Core adjustable traits for AI agents. Select 3-5; intensity modulates prominence. Hover for engagement tips."""
    
    EMPATHY = "empathy"
    """Emotional attunement: Builds trust via validation. High: Mirrors feelings deeply."""
    
    WIT = "wit"
    """Humor/cleverness: Adds delight without derailing. High: Playful banter for rapport."""
    
    PRECISION = "precision"
    """Accuracy/detail: Ensures credibility. High: Evidence-based, structured replies."""
    
    CURIOSITY = "curiosity"
    """Exploratory probing: Deepens dialogue. High: Follow-up questions for collaboration."""
    
    CONFIDENCE = "confidence"
    """Assertive conviction: Drives decisions. High: Bold recommendations with rationale."""
    
    WARMTH = "warmth"
    """Friendliness/inclusivity: Softens tone. High: Personal, emoji-enhanced phrasing."""
    
    CREATIVITY = "creativity"
    """Novel ideas/analogies: Sparks inspiration. High: Metaphor-rich brainstorming."""
    
    RESPONSIVENESS = "responsiveness"
    """Adaptive flow: Mirrors user cues. High: Fluid pivots for natural conversation."""

class AgentTrait(BaseModel):
    """Single trait instance: Enum-based for type-safety."""
    trait: Trait = Field(..., description="Trait from enum; auto-generates description")
    intensity: int = Field(..., ge=0, le=100, description="0=absent, 100=dominant")

class AgentPersonality(BaseModel):
    """
        Structured personality definition with DISC archetype blends for consistent, relatable AI agents.
        
        Supports primary/secondary archetypes with weights (e.g., 60% Driver / 40% Amiable) for nuanced behavior.
        Weights must sum to 100% (or 1.0 as float). Single archetype: set secondary=None, primary_weight=1.0.
        """
    name: str = Field(..., description="Agent's name, e.g., 'Dr. Riley' for humanization")
    gender: Literal['male', 'female', 'nonbinary', 'unspecified'] = Field(
        ..., description="Agent's gender for pronoun/tone adaptation (keeps responses inclusive)"
    )
    # Archetype blend fields with descriptions for IntelliSense
    primary_archetype: Archetype = Field(..., description="Dominant DISC style; sets core behavioral baseline")
    primary_weight: float = Field(..., ge=0.5, le=1.0, description="Weight for primary (e.g., 0.6 = 60%); must be >=0.5 if secondary is set")
    secondary_archetype: Optional[Archetype] = Field(
        None, description="Optional secondary DISC style for blending (e.g., Amiable to soften a Driver)"
    )
    secondary_weight: Optional[float] = Field(
        None, description="Weight for secondary (auto-calculated as 1 - primary_weight if not set)"
    )
    job_description: str = Field(..., description="Role shaping behaviors, e.g., 'psychological research assistant'")
    traits: Dict[str, AgentTrait] = Field(
        default_factory=dict, max_items=5, description="Dict of trait_name: AgentTrait; limit 3-5 for focus"
    )

    @model_validator(mode='after')
    def validate_weights(self) -> 'AgentPersonality':
        """Ensure weights sum to 1.0; auto-fill secondary_weight if missing."""
        if self.secondary_archetype:
            if self.secondary_weight is None:
                self.secondary_weight = 1.0 - self.primary_weight
            if abs(self.primary_weight + self.secondary_weight - 1.0) > 0.01:
                raise ValueError("Primary and secondary weights must sum to 1.0")
            if self.primary_weight < 0.5:
                raise ValueError("Primary weight must be >=0.5 when secondary is set")
        else:
            self.primary_weight = 1.0
            self.secondary_weight = None
        return self
    
    @model_validator(mode='after')
    def validate_traits(self) -> 'AgentPersonality':
        """Cap at 5 traits; suggest DISC synergies (e.g., warn if low empathy on Amiable)."""
        if len(self.traits) > 5:
            raise ValueError("Max 5 traits for prompt clarity")
        
        # Fix: check if 'warmth' exists and is an AgentTrait, then access .intensity
        warmth_trait = self.traits.get('warmth')
        if warmth_trait and hasattr(warmth_trait, 'intensity') and warmth_trait.intensity < 50:
            print("💡 Tip: Boost 'warmth' >=50 for Amiable archetype synergy.")
        
        return self

    def to_system_prompt(self) -> str:
        """
        Generate a blended system prompt. Weights interpolate archetype styles:
        - High primary_weight: Mostly primary traits (e.g., 80% Driver = direct, 20% Expressive = add enthusiasm).
        - Blends create hybrid tones for engagement (e.g., Driver/Amiable = decisive but empathetic).
        
        Returns a string ready for xAI chat_completion's system role.
        """
        archetype_prompts = {
            Archetype.DRIVER: "You are direct, action-oriented, and decisive. Prioritize concise, goal-focused responses.",
            Archetype.EXPRESSIVE: "You are enthusiastic, persuasive, and creative. Use vibrant language and storytelling.",
            Archetype.AMIABLE: "You are empathetic, supportive, and warm. Use comforting, inclusive language.",
            Archetype.ANALYTICAL: "You are precise, logical, and detail-oriented. Emphasize data and clarity."
        }
        
        # Blend prompts based on weights
        if self.secondary_archetype:
            primary_style = archetype_prompts[self.primary_archetype]
            secondary_style = archetype_prompts[self.secondary_archetype]
            # Simple interpolation: weight * style (in practice, LLM blends naturally)
            blended_style = f"{primary_style} Blend {self.secondary_weight*100:.0f}% with {secondary_style} for balanced tone."
        else:
            blended_style = archetype_prompts[self.primary_archetype]
        
        traits_prompt = "\n".join([
            f"- {t.trait.value.upper()} (intensity: {t.intensity}%): {t.trait.value} at this level by {self._get_manifestation(t.trait, t.intensity)}"
            for t in self.traits.values()
        ]) if self.traits else "No additional traits specified."
            
        return f"""You are {self.name}, a {self.gender} {self.job_description}. {blended_style} {traits_prompt} 
        Always respond in character: adapt tone/style to this personality while accurately addressing queries. 
        Keep responses engaging and relatable."""
    
    def _get_manifestation(self, trait: Trait, intensity: int) -> str:
        """Dynamic phrasing based on intensity for vivid prompts."""
        thresholds = {33: "occasionally", 66: "frequently", 100: "always prominently"}
        level = next((desc for thresh, desc in thresholds.items() if intensity > thresh), "sparingly")
        return f"incorporating {trait.value} {level} in responses"
    
    def to_json(self) -> str:
        """Serialize to JSON for storage or API use."""
        return self.model_dump_json()