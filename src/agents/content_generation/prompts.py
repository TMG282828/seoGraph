"""
Prompts for Content Generation Agent.

This module contains all the prompt templates used by the Content Generation Agent.
"""

import json
from typing import Dict, Any


class ContentGenerationPrompts:
    """Prompt templates for content generation."""
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Content Generation Agent."""
        return """You are an expert Content Generation Agent specializing in creating high-quality, SEO-optimized content.

Your role is to generate content that excels in:
- SEO optimization with natural keyword integration
- Brand voice consistency and authenticity
- User engagement and value delivery
- Search intent alignment and satisfaction
- Semantic coherence and topic authority
- Technical optimization for search engines
- Conversion optimization when appropriate

Always consider:
1. The organization's brand voice, tone, and style guidelines
2. Target keyword integration without compromising readability
3. User intent and search behavior patterns
4. Content structure for both users and search engines
5. Internal linking opportunities for topical authority
6. Competitive differentiation and unique value proposition
7. Content format optimization for target audience

Generate content that ranks well, engages users, and drives business objectives while maintaining editorial excellence and authenticity."""
    
    def get_outline_prompt(self, request, brand_voice: Dict[str, Any], seo_requirements: Dict[str, Any]) -> str:
        """Get the prompt for generating content outlines."""
        
        return f"""
        Create a comprehensive content outline for:

        TOPIC: {request.topic}
        CONTENT TYPE: {request.content_type}
        TARGET KEYWORDS: {', '.join(request.target_keywords)}
        CONTENT LENGTH: {request.content_length}
        WRITING STYLE: {request.writing_style}
        TARGET AUDIENCE: {request.target_audience}

        BRAND VOICE GUIDELINES:
        {json.dumps(brand_voice, indent=2)}

        SEO REQUIREMENTS:
        {json.dumps(seo_requirements, indent=2)}

        Create a detailed outline including:
        1. SEO-optimized title variations
        2. Introduction hook and key points
        3. Main sections with subsections
        4. Key points and supporting details for each section
        5. Internal linking opportunities
        6. Conclusion and call-to-action
        7. Meta tags and SEO elements

        Use the available tools to create title variations and outline structure.
        """
    
    def get_full_content_prompt(self, request, selected_title: str, content_outline: Dict[str, Any], 
                               brand_voice: Dict[str, Any], seo_requirements: Dict[str, Any]) -> str:
        """Get the prompt for generating full content."""
        
        hil_enhancement = self._get_human_in_loop_prompt_enhancement(request)
        
        return f"""
        Generate a complete {request.content_type} based on this outline and requirements:

        TITLE: {selected_title}
        TOPIC: {request.topic}
        TARGET KEYWORDS: {', '.join(request.target_keywords)}
        CONTENT LENGTH: {request.content_length}
        WRITING STYLE: {request.writing_style}
        TARGET AUDIENCE: {request.target_audience}

        CONTENT OUTLINE:
        {json.dumps(content_outline, indent=2)}

        BRAND VOICE GUIDELINES:
        {json.dumps(brand_voice, indent=2)}

        SEO REQUIREMENTS:
        {json.dumps(seo_requirements, indent=2)}

        COMPETITOR ANALYSIS DATA:
        {json.dumps(request.competitor_analysis_data or {}, indent=2)}

        REFERENCE CONTENT:
        {chr(10).join(request.reference_content)}

        {hil_enhancement}

        Generate comprehensive content that:
        1. Follows the brand voice and style guidelines exactly
        2. Naturally integrates target keywords without stuffing
        3. Provides unique value and insights
        4. Maintains excellent readability and flow
        5. Includes proper header structure (H1, H2, H3)
        6. Optimizes for search intent and user experience
        7. Incorporates internal linking opportunities
        8. Differentiates from competitor content

        Use the available tools to generate sections, optimize for snippets, and suggest improvements.
        """
    
    def _get_human_in_loop_prompt_enhancement(self, request) -> str:
        """Generate enhanced prompts based on human-in-loop settings."""
        if not request.human_in_loop:
            return ""
        
        hil_settings = request.human_in_loop
        enhancement_parts = []
        
        # Check-in frequency instructions
        checkin_freq = hil_settings.get("checkin_frequency", "medium")
        if checkin_freq == "high":
            enhancement_parts.append("""
            HUMAN-IN-LOOP INSTRUCTIONS (HIGH FREQUENCY):
            - Provide detailed step-by-step progress updates
            - Ask for approval before each major section
            - Include checkpoint questions for user feedback
            - Offer alternative approaches at key decision points
            """)
        elif checkin_freq == "medium":
            enhancement_parts.append("""
            HUMAN-IN-LOOP INSTRUCTIONS (MEDIUM FREQUENCY):
            - Provide progress updates at major milestones
            - Ask for feedback on content direction and tone
            - Include summary checkpoints for user review
            """)
        else:  # low
            enhancement_parts.append("""
            HUMAN-IN-LOOP INSTRUCTIONS (LOW FREQUENCY):
            - Work autonomously with minimal interruption
            - Provide final result with comprehensive analysis
            - Include brief summary of decisions made
            """)
        
        # Agent aggressiveness
        aggressiveness = hil_settings.get("agent_aggressiveness", 5)
        if aggressiveness >= 8:
            enhancement_parts.append("- Take bold, innovative approaches to content creation")
            enhancement_parts.append("- Push creative boundaries while maintaining quality")
        elif aggressiveness <= 3:
            enhancement_parts.append("- Take conservative, well-tested approaches")
            enhancement_parts.append("- Focus on proven strategies and safe implementations")
        
        # Content goals integration
        if request.content_goals:
            goals = request.content_goals
            primary_goal = goals.get("primary", "")
            enhancement_parts.append(f"\nPRIMARY CONTENT GOAL: {primary_goal}")
            
            if primary_goal == "SEO-Focused":
                enhancement_parts.append("- Prioritize keyword optimization and search rankings")
                enhancement_parts.append("- Include comprehensive SEO elements (headers, meta tags, internal links)")
            elif primary_goal == "Brand-Focused":
                enhancement_parts.append("- Emphasize brand voice consistency and messaging")
                enhancement_parts.append("- Integrate brand values and personality throughout")
            elif primary_goal == "Research-Heavy":
                enhancement_parts.append("- Include data-driven insights and statistics")
                enhancement_parts.append("- Reference authoritative sources and studies")
            elif primary_goal == "Thought Leadership":
                enhancement_parts.append("- Demonstrate industry expertise and unique insights")
                enhancement_parts.append("- Position content as authoritative and forward-thinking")
        
        # Brand voice integration
        if request.brand_voice:
            brand_voice = request.brand_voice
            enhancement_parts.append(f"\nBRAND VOICE REQUIREMENTS:")
            enhancement_parts.append(f"- Tone: {brand_voice.get('tone', 'professional')}")
            enhancement_parts.append(f"- Formality: {brand_voice.get('formality', 'semi-formal')}")
            
            if brand_voice.get("description"):
                enhancement_parts.append(f"- Brand Description: {brand_voice['description']}")
            
            if brand_voice.get("keywords"):
                keywords = ", ".join(brand_voice["keywords"])
                enhancement_parts.append(f"- Brand Keywords to Emphasize: {keywords}")
        
        return "\n".join(enhancement_parts) if enhancement_parts else ""