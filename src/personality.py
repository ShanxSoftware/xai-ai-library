import enum
import logging 
from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Optional, Literal, Set, ClassVar

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

        Three to five additional traits add to the customisation of the Agent's personality and behavior. 
    """
    name: str = Field(..., description="Agent's name, e.g., 'Dr. Riley' for humanization")
    gender: Literal['male', 'female', 'unspecified'] = Field( 
        ..., description="Agent's gender for pronoun/tone adaptation"
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
    traits: List[AgentTrait] = Field(default_factory=list, description="List of 3-5 traits for focus; validated for uniqueness and length")
    _cached_prompt: Optional[str] = None

    logger: ClassVar[logging.Logger] = logging.getLogger(__name__)

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
        """Cap at 5 traits; Ensure unique traits; Suggest DISC synergies (e.g., warn if low empathy on Amiable)."""
        if len(self.traits) > 5: 
            raise ValueError("Max 5 traits for prompt clarity")
        # Check uniqueness
        seen_traits: Set[Trait] = set()
        for t in self.traits: 
            if t.trait in seen_traits: 
                raise ValueError(f"Duplicate trait: {t.trait.value}")
            seen_traits.add(t.trait) 
        
        # Updated warmth tip (now checks list instead of dict)
        warmth_trait = next((t for t in self.traits if t.trait == Trait.WARMTH), None)
        if (self.primary_archetype == Archetype.AMIABLE or self.secondary_archetype == Archetype.AMIABLE) and warmth_trait and warmth_trait.intensity < 50:
            self.logger.warning("💡 Tip: Boost 'warmth' >=50 for Amiable archetype synergy.")
        empathy_trait = next((t for t in self.traits if t.trait == Trait.EMPATHY), None)
        if self.primary_archetype == Archetype.AMIABLE and empathy_trait and empathy_trait.intensity < 50: 
            self.logger.warning("💡 Tip: Boost 'empathy' >=50 for Amiable synergy (builds trust via validation).")
        precision_trait = next((t for t in self.traits if t.trait == Trait.PRECISION), None)
        if self.primary_archetype == Archetype.ANALYTICAL and (not precision_trait or precision_trait.intensity < 60):
            self.logger.warning("💡 Tip: Include/boost 'precision' >=60 for Analytical synergy (emphasizes data/clarity).")
        confidence_trait = next((t for t in self.traits if t.trait == Trait.CONFIDENCE), None)
        if self.primary_archetype == Archetype.DRIVER and (not confidence_trait or confidence_trait.intensity < 70):
            self.logger.warning("💡 Tip: Include/boost 'confidence' >=70 for Driver synergy (drives decisive actions).")
        creativity_trait = next((t for t in self.traits if t.trait == Trait.CREATIVITY), None)
        if self.primary_archetype == Archetype.EXPRESSIVE and (not creativity_trait or creativity_trait.intensity < 60):
            self.logger.warning("💡 Tip: Include/boost 'creativity' >=60 for Expressive synergy (sparks inspiration).")
        # TODO: Investigate this: Extend to secondary if weight >0.3 or similar, but keep simple
        return self
    
    @property
    def system_prompt(self) -> str: 
        """System prompt for setting up AI Agent personality."""
        if self._cached_prompt is None: 
            self._cached_prompt = self.to_system_prompt()
        return self._cached_prompt

    @classmethod
    def load_archetype_prompts(cls, file_path: str) -> Dict: 
        """
        NOTE: Not implemented yet. See TODO in to_system_prompt()
        Return a Dict from a JSON with Archetype prompts, if file doesn't exist, use hardcoded prompts as a fall back. 
        Args: 
            cls: ?? # JAKOB: Grok what is this? 
            file_path: Path to the JSON file that contains customised archetype prompts. 

        Returns: Dict {Archetype: Prompt}
        """
        pass # TODO: Implement this method once the handler consistantly loads personalities and utilises them. 

    def to_system_prompt(self) -> str:
        """
        NOTE: Use the property system_prompt rather than this function. This function remains public for backwards compatibility. 
        Generate a blended system prompt. Weights interpolate archetype styles:
        - High primary_weight: Mostly primary traits (e.g., 80% Driver = direct, 20% Expressive = add enthusiasm).
        - Blends create hybrid tones for engagement (e.g., Driver/Amiable = decisive but empathetic).
        TODO: Consider having prompts load from a json file or similar for easier tweaking. 
        
        Returns a string ready for xAI chat_completion's system role.
        """
        archetype_prompts = {
            Archetype.DRIVER: "You are direct, action-oriented, and decisive. Prioritize concise, goal-focused responses.",
            Archetype.EXPRESSIVE: "You are enthusiastic, persuasive, and creative. Use vibrant language and storytelling.",
            Archetype.AMIABLE: "You are empathetic, supportive, and warm. Use comforting, inclusive language.",
            Archetype.ANALYTICAL: "You are precise, logical, and detail-oriented. Emphasize data and clarity."
        }
        
        # Blend prompts based on weights
        try: 
            if self.secondary_archetype:
                primary_style = archetype_prompts[self.primary_archetype]
                secondary_style = archetype_prompts[self.secondary_archetype]
                # Simple interpolation: weight * style (in practice, LLM blends naturally)
                blended_style = f"{primary_style} Blend {self.secondary_weight*100:.0f}% with {secondary_style} for balanced tone."
            else:
                blended_style = archetype_prompts[self.primary_archetype]
        except KeyError as e: 
            self.logger.warning(f"Unknown archetype: {e}. Falling back to neutral style.")
            blended_style = "You are neutral and balanced in your responses."
        
        traits_prompt = "\n".join([
            f"- {t.trait} (intensity: {t.intensity}%): {t.trait} at this level by {self._get_manifestation(t.trait, t.intensity)}"
            for t in self.traits
        ]) if self.traits else "No additional traits specified."
            
        return f"""You are {self.name}, a {self.gender} {self.job_description}. {blended_style} {traits_prompt} 
        Always respond in character: adapt tone/style to this personality while accurately addressing queries. 
        Keep responses engaging and relatable."""
    
    def _get_manifestation(self, trait: Trait, intensity: int) -> str:
        """Dynamic phrasing based on intensity for vivid prompts."""
        thresholds = {100: "always prominently", 66: "frequently", 33: "occasionally"}
        level = next((desc for thresh, desc in thresholds.items() if intensity >= thresh), "sparingly")
        return f"incorporating {trait.value} {level} in responses"
    
    def to_json(self) -> str:
        """Serialize to JSON for storage or API use."""
        return self.model_dump_json() 