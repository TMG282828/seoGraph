"""
Content Generation Tools for SEO Content Knowledge Graph System.

This module contains all the tool functions used by the Content Generation Agent.
"""

import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from .rag_tools import RAGTools

logger = logging.getLogger(__name__)


class ContentGenerationTools:
    """Tools for content generation functionality."""
    
    def __init__(self):
        self.rag_tools = RAGTools()
    
    def set_neo4j_client(self, client):
        """Set the Neo4j client for knowledge graph operations."""
        self.rag_tools.set_neo4j_client(client)
    
    def set_qdrant_client(self, client):
        """Set the Qdrant client for vector search operations."""
        self.rag_tools.set_qdrant_client(client)
    
    # RAG Tool delegates
    async def search_knowledge_graph(self, topic: str, keywords: List[str]) -> Dict[str, Any]:
        """Search the knowledge graph for related content and topics."""
        return await self.rag_tools.search_knowledge_graph(topic, keywords)
    
    async def find_similar_content(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar content using vector search."""
        return await self.rag_tools.find_similar_content(query, limit)
    
    async def get_content_relationships(self, topic: str) -> Dict[str, Any]:
        """Get relationships and connections for the topic from knowledge graph."""
        return await self.rag_tools.get_content_relationships(topic)
    
    async def enhance_with_context(self, content: str, topic: str) -> str:
        """Enhance content with contextual information from knowledge base."""
        return await self.rag_tools.enhance_with_context(content, topic)
    
    async def create_content_outline(self, topic: str, keywords: List[str], content_type: str) -> Dict[str, Any]:
        """Create a comprehensive content outline."""
        
        # Define section templates based on content type
        section_templates = {
            "blog_post": [
                {"title": "Introduction", "type": "intro", "estimated_words": 150},
                {"title": f"What is {topic}?", "type": "definition", "estimated_words": 300},
                {"title": f"Benefits of {topic}", "type": "benefits", "estimated_words": 400},
                {"title": f"How to {topic.lower()}", "type": "how_to", "estimated_words": 500},
                {"title": f"Best Practices for {topic}", "type": "best_practices", "estimated_words": 400},
                {"title": "Common Mistakes to Avoid", "type": "mistakes", "estimated_words": 300},
                {"title": "Conclusion", "type": "conclusion", "estimated_words": 150}
            ],
            "guide": [
                {"title": "Introduction", "type": "intro", "estimated_words": 200},
                {"title": f"Complete Guide to {topic}", "type": "overview", "estimated_words": 400},
                {"title": "Getting Started", "type": "getting_started", "estimated_words": 500},
                {"title": "Advanced Strategies", "type": "advanced", "estimated_words": 600},
                {"title": "Tools and Resources", "type": "tools", "estimated_words": 300},
                {"title": "Conclusion and Next Steps", "type": "conclusion", "estimated_words": 200}
            ],
            "article": [
                {"title": "Introduction", "type": "intro", "estimated_words": 150},
                {"title": f"Understanding {topic}", "type": "understanding", "estimated_words": 400},
                {"title": "Key Components", "type": "components", "estimated_words": 450},
                {"title": "Implementation Strategy", "type": "implementation", "estimated_words": 400},
                {"title": "Results and Benefits", "type": "results", "estimated_words": 300},
                {"title": "Conclusion", "type": "conclusion", "estimated_words": 150}
            ]
        }
        
        # Get base sections for content type
        base_sections = section_templates.get(content_type, section_templates["blog_post"])
        
        # Customize sections with keywords
        sections = []
        for section in base_sections:
            customized_section = section.copy()
            customized_section["keywords"] = self._assign_keywords_to_section(section["type"], keywords)
            customized_section["key_points"] = self._generate_section_key_points(section["title"], keywords)
            sections.append(customized_section)
        
        return {
            "content_type": content_type,
            "topic": topic,
            "target_keywords": keywords,
            "sections": sections,
            "estimated_total_words": sum(section["estimated_words"] for section in sections),
            "seo_elements": {
                "title_keywords": keywords[:2],
                "header_keywords": keywords,
                "meta_description_keywords": keywords[:3]
            }
        }
    
    async def generate_title_variations(self, topic: str, keywords: List[str], target_audience: str) -> List[str]:
        """Generate multiple title variations optimized for SEO and engagement."""
        
        primary_keyword = keywords[0] if keywords else topic
        
        # Template-based title generation
        title_templates = [
            f"The Complete Guide to {primary_keyword}",
            f"How to {primary_keyword}: A {target_audience.title()} Guide",
            f"{primary_keyword}: Everything You Need to Know",
            f"Mastering {primary_keyword}: Tips and Best Practices",
            f"The Ultimate {primary_keyword} Strategy for {target_audience.title()}",
            f"{primary_keyword} Explained: A Comprehensive Overview",
            f"Best {primary_keyword} Techniques for {datetime.now().year}",
            f"{primary_keyword} vs Alternatives: Which is Better?",
            f"Top {primary_keyword} Strategies That Actually Work",
            f"Why {primary_keyword} is Essential for {target_audience.title()}"
        ]
        
        # Filter and customize titles
        titles = []
        for template in title_templates:
            if len(template) <= 60:  # SEO title length
                titles.append(template)
        
        # Add keyword variations
        for keyword in keywords[1:3]:  # Additional keywords
            titles.extend([
                f"{keyword} and {primary_keyword}: Complete Guide",
                f"How {keyword} Improves {primary_keyword} Results"
            ])
        
        return titles[:10]  # Return top 10 variations
    
    async def create_introduction(self, title: str, keywords: List[str], hook_type: str) -> str:
        """Create an engaging introduction section."""
        
        primary_keyword = keywords[0] if keywords else "this topic"
        
        hook_templates = {
            "question": f"Are you struggling with {primary_keyword}? You're not alone.",
            "statistic": f"Studies show that 73% of businesses see improved results with proper {primary_keyword} implementation.",
            "story": f"When I first started working with {primary_keyword}, I made every mistake in the book.",
            "problem": f"Most people approach {primary_keyword} completely wrong.",
            "benefit": f"Imagine doubling your results with {primary_keyword} in just 30 days."
        }
        
        hook = hook_templates.get(hook_type, hook_templates["question"])
        
        introduction = f"""{hook}

In this comprehensive guide, we'll explore everything you need to know about {primary_keyword}. Whether you're a beginner just getting started or looking to improve your existing {primary_keyword} strategy, this guide will provide you with actionable insights and proven techniques.

Here's what you'll learn:
• The fundamentals of {primary_keyword} and why it matters
• Step-by-step implementation strategies
• Common mistakes to avoid
• Advanced techniques for better results
• Tools and resources to accelerate your progress

Let's dive in and transform your approach to {primary_keyword}."""
        
        return introduction
    
    async def generate_section_content(self, section_title: str, keywords: List[str], word_count: int) -> str:
        """Generate content for a specific section."""
        
        primary_keyword = keywords[0] if keywords else "the topic"
        
        # Base content structure
        content_parts = []
        
        # Section introduction
        content_parts.append(f"When it comes to {section_title.lower()}, {primary_keyword} plays a crucial role in achieving your goals.")
        
        # Main content (adjust based on word count)
        if word_count >= 300:
            content_parts.extend([
                f"\n## Key Aspects of {section_title}\n",
                f"Understanding {primary_keyword} requires focusing on several important factors:",
                "\n• **Strategy**: Developing a clear approach to implementation",
                "• **Best Practices**: Following proven methods for success", 
                "• **Optimization**: Continuously improving your results",
                "• **Measurement**: Tracking progress and making adjustments"
            ])
        
        if word_count >= 500:
            content_parts.extend([
                f"\n### Implementation Steps\n",
                f"To effectively implement {primary_keyword} in your strategy:",
                "\n1. **Assessment**: Evaluate your current situation",
                "2. **Planning**: Create a detailed action plan",
                "3. **Execution**: Put your plan into action", 
                "4. **Monitoring**: Track results and adjust as needed",
                "5. **Optimization**: Refine your approach based on data"
            ])
        
        # Section conclusion
        content_parts.append(f"\nBy focusing on these elements of {section_title.lower()}, you'll be well-positioned to maximize the benefits of {primary_keyword} in your overall strategy.")
        
        return " ".join(content_parts)
    
    async def create_conclusion(self, main_points: List[str], cta_type: str) -> str:
        """Create a compelling conclusion with call-to-action."""
        
        # Summarize main points
        conclusion_parts = [
            "## Conclusion\n",
            "In this comprehensive guide, we've covered the essential aspects of implementing an effective strategy:"
        ]
        
        # Add main points summary
        for i, point in enumerate(main_points[:5], 1):
            conclusion_parts.append(f"\n{i}. **{point}**: Key insights and actionable strategies")
        
        # Add call-to-action based on type
        cta_templates = {
            "action": "\n### Ready to Get Started?\n\nNow that you have a solid understanding of the fundamentals, it's time to put this knowledge into practice. Start with the basics and gradually implement more advanced strategies as you gain experience.",
            "engagement": "\n### What's Your Experience?\n\nWe'd love to hear about your experiences and any additional tips you'd like to share. Leave a comment below or reach out to us directly.",
            "resource": "\n### Need More Help?\n\nIf you're looking for additional resources or personalized guidance, check out our related articles and tools to continue your learning journey.",
            "conversion": "\n### Take Your Strategy to the Next Level\n\nReady to implement these strategies but need expert guidance? Our team of specialists can help you develop and execute a customized plan that delivers results."
        }
        
        conclusion_parts.append(cta_templates.get(cta_type, cta_templates["action"]))
        
        # Final thought
        conclusion_parts.append("\nRemember, success comes from consistent implementation and continuous improvement. Start today and build momentum toward achieving your goals.")
        
        return "".join(conclusion_parts)
    
    async def suggest_internal_links(self, content: str, topic: str) -> List[Dict[str, str]]:
        """Suggest internal linking opportunities."""
        
        # Extract potential link targets from content
        words = content.lower().split()
        
        # Mock internal link suggestions (in production, this would query the knowledge graph)
        suggested_links = [
            {
                "anchor_text": f"{topic} best practices",
                "url": f"/guides/{topic.lower().replace(' ', '-')}-best-practices",
                "relevance_score": 0.9,
                "context": "Mentioned in best practices section"
            },
            {
                "anchor_text": f"advanced {topic} strategies",
                "url": f"/articles/advanced-{topic.lower().replace(' ', '-')}-strategies",
                "relevance_score": 0.8,
                "context": "Referenced in implementation section"
            },
            {
                "anchor_text": f"{topic} tools and resources",
                "url": f"/resources/{topic.lower().replace(' ', '-')}-tools",
                "relevance_score": 0.7,
                "context": "Mentioned in tools section"
            }
        ]
        
        # Filter based on content relevance
        relevant_links = []
        for link in suggested_links:
            anchor_words = link["anchor_text"].lower().split()
            if any(word in words for word in anchor_words):
                relevant_links.append(link)
        
        return relevant_links[:5]  # Return top 5 suggestions
    
    async def generate_meta_tags(self, title: str, content: str, keywords: List[str]) -> Dict[str, str]:
        """Generate meta title, description, and keywords."""
        
        # Meta title (optimize for 60 characters)
        meta_title = title
        if len(meta_title) > 60:
            meta_title = meta_title[:57] + "..."
        
        # Meta description (optimize for 160 characters)
        first_paragraph = content.split('\n')[0]
        meta_description = first_paragraph
        if len(meta_description) > 160:
            meta_description = meta_description[:157] + "..."
        
        # Ensure primary keyword is in meta description
        if keywords and keywords[0].lower() not in meta_description.lower():
            meta_description = f"Learn about {keywords[0]}. {meta_description}"
            if len(meta_description) > 160:
                meta_description = meta_description[:157] + "..."
        
        # Meta keywords (comma-separated)
        meta_keywords = ", ".join(keywords[:10])
        
        # Additional structured data suggestions
        return {
            "title": meta_title,
            "description": meta_description,
            "keywords": meta_keywords,
            "og_title": title,
            "og_description": meta_description,
            "og_type": "article",
            "twitter_card": "summary_large_image",
            "twitter_title": meta_title,
            "twitter_description": meta_description
        }
    
    async def optimize_for_featured_snippets(self, content: str, target_query: str) -> str:
        """Optimize content sections for featured snippet capture."""
        
        # Add direct answer format at the beginning of relevant sections
        if "what is" in target_query.lower():
            definition_snippet = f"\n**{target_query.title()}**\n\n{target_query.split()[-1]} is a [concise definition that directly answers the query]. This approach helps [main benefit] and is essential for [key use case].\n"
            content = definition_snippet + content
        
        elif "how to" in target_query.lower():
            # Add numbered list format
            steps_snippet = f"\n**{target_query.title()}**\n\n1. **Step 1**: [First action to take]\n2. **Step 2**: [Second action to take]\n3. **Step 3**: [Third action to take]\n\n"
            content = steps_snippet + content
        
        elif any(word in target_query.lower() for word in ["best", "top", "comparison"]):
            # Add comparison table or list format
            list_snippet = f"\n**{target_query.title()}**\n\n• **Option 1**: [Brief description and key benefit]\n• **Option 2**: [Brief description and key benefit]\n• **Option 3**: [Brief description and key benefit]\n\n"
            content = list_snippet + content
        
        return content
    
    # Helper methods
    
    def _assign_keywords_to_section(self, section_type: str, keywords: List[str]) -> List[str]:
        """Assign relevant keywords to content sections."""
        
        # Distribute keywords based on section type
        keyword_distribution = {
            "intro": keywords[:2],
            "definition": keywords[:1],
            "benefits": keywords[1:3] if len(keywords) > 1 else keywords,
            "how_to": keywords,
            "best_practices": keywords[:3],
            "mistakes": keywords[:2],
            "conclusion": keywords[:1]
        }
        
        return keyword_distribution.get(section_type, keywords[:2])
    
    def _generate_section_key_points(self, section_title: str, keywords: List[str]) -> List[str]:
        """Generate key points for a content section."""
        
        primary_keyword = keywords[0] if keywords else "this topic"
        
        key_points = [
            f"Essential {primary_keyword} fundamentals",
            f"Practical implementation strategies",
            f"Common challenges and solutions",
            f"Measurable outcomes and benefits"
        ]
        
        return key_points
    
    def estimate_word_count(self, outline: Dict[str, Any], content_length: str) -> int:
        """Estimate total word count based on outline and length preference."""
        
        base_count = outline.get("estimated_total_words", 0)
        
        length_multipliers = {
            "short": 0.7,
            "medium": 1.0,
            "long": 1.5
        }
        
        multiplier = length_multipliers.get(content_length, 1.0)
        return int(base_count * multiplier)
    
    async def analyze_outline_seo(self, outline: Dict[str, Any], keywords: List[str]) -> Dict[str, Any]:
        """Analyze SEO potential of content outline."""
        
        sections = outline.get("sections", [])
        
        return {
            "keyword_distribution": len([s for s in sections if s.get("keywords")]),
            "header_optimization": len([s for s in sections if any(kw in s.get("title", "").lower() for kw in keywords)]),
            "content_depth_score": len(sections) * 10,  # Simple scoring
            "seo_potential": "high" if len(sections) >= 5 else "medium"
        }
    
    def combine_content_sections(self, title: str, introduction: str, 
                               sections_content: Dict[str, str], conclusion: str) -> str:
        """Combine all content sections into a complete piece."""
        
        content_parts = [f"# {title}\n", introduction]
        
        for section_title, section_content in sections_content.items():
            content_parts.extend([f"\n\n## {section_title}\n", section_content])
        
        content_parts.append(f"\n\n{conclusion}")
        
        return "".join(content_parts)
    
    async def analyze_generated_content(self, content: str, request, brand_voice: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the generated content for quality and compliance."""
        
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Keyword density analysis
        keyword_densities = {}
        content_lower = content.lower()
        total_words = len(content.split())
        
        for keyword in request.target_keywords:
            keyword_count = content_lower.count(keyword.lower())
            density = (keyword_count / total_words) * 100 if total_words > 0 else 0
            keyword_densities[keyword] = {
                "count": keyword_count,
                "density_percentage": round(density, 2)
            }
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "average_sentence_length": round(avg_sentence_length, 1),
            "keyword_densities": keyword_densities,
            "header_count": len(re.findall(r'^#+\s+', content, re.MULTILINE)),
            "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
            "reading_level": "intermediate",  # Would calculate actual reading level
            "content_structure_score": 85
        }
    
    def calculate_readability_score(self, content: str) -> float:
        """Calculate readability score for content."""
        
        words = content.split()
        sentences = [s for s in content.split('.') if s.strip()]
        
        if not sentences:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Simple readability formula (simplified Flesch Reading Ease)
        if avg_sentence_length <= 10:
            score = 90  # Very easy
        elif avg_sentence_length <= 15:
            score = 80  # Easy
        elif avg_sentence_length <= 20:
            score = 70  # Fairly easy
        elif avg_sentence_length <= 25:
            score = 60  # Standard
        else:
            score = 50  # Difficult
        
        return float(score)
    
    async def calculate_content_seo_score(self, content: str, keywords: List[str]) -> float:
        """Calculate SEO score for generated content."""
        
        score = 0
        content_lower = content.lower()
        word_count = len(content.split())
        
        # Word count score (800-2000 optimal)
        if 800 <= word_count <= 2000:
            score += 25
        elif 500 <= word_count <= 3000:
            score += 15
        else:
            score += 5
        
        # Keyword optimization score
        for keyword in keywords[:3]:  # Check top 3 keywords
            if keyword.lower() in content_lower:
                score += 15
                # Bonus for keyword in headers
                if any(keyword.lower() in header.lower() for header in re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)):
                    score += 5
        
        # Header structure score
        headers = re.findall(r'^#+\s+', content, re.MULTILINE)
        if len(headers) >= 3:
            score += 20
        elif len(headers) >= 1:
            score += 10
        
        # Internal structure score
        if re.search(r'\n\n', content):  # Has paragraph breaks
            score += 10
        
        if re.search(r'^\d+\.|^•|^\*', content, re.MULTILINE):  # Has lists
            score += 10
        
        return min(100, score)
    
    async def check_brand_voice_compliance(self, content: str, brand_voice: Dict[str, Any]) -> Dict[str, Any]:
        """Check content compliance with brand voice guidelines."""
        
        if not brand_voice:
            return {"compliance_score": 50, "note": "No brand voice guidelines provided"}
        
        compliance_score = 100
        violations = []
        
        # Check prohibited terms
        prohibited_terms = brand_voice.get("prohibitedTerms", [])
        content_lower = content.lower()
        
        for term in prohibited_terms:
            if term.lower() in content_lower:
                compliance_score -= 10
                violations.append(f"Contains prohibited term: {term}")
        
        # Check preferred phrases
        preferred_phrases = brand_voice.get("preferredPhrases", [])
        used_preferred = []
        
        for phrase in preferred_phrases:
            if phrase.lower() in content_lower:
                used_preferred.append(phrase)
                compliance_score += 5
        
        compliance_score = max(0, min(100, compliance_score))
        
        return {
            "compliance_score": compliance_score,
            "violations": violations,
            "used_preferred_phrases": used_preferred,
            "tone_assessment": f"Content aligns with {brand_voice.get('tone', 'specified')} tone"
        }
    
    async def generate_improvement_suggestions(self, content: str, request) -> List[str]:
        """Generate improvement suggestions for the content."""
        
        suggestions = []
        word_count = len(content.split())
        
        # Length suggestions
        if word_count < 500:
            suggestions.append("Consider expanding content to at least 500 words for better SEO performance")
        elif word_count > 3000:
            suggestions.append("Content is quite long - consider breaking into multiple pieces or adding a table of contents")
        
        # Structure suggestions
        headers = re.findall(r'^#+\s+', content, re.MULTILINE)
        if len(headers) < 3:
            suggestions.append("Add more headers to improve content structure and scannability")
        
        # Keyword suggestions
        if request.target_keywords:
            primary_keyword = request.target_keywords[0]
            if primary_keyword.lower() not in content.lower():
                suggestions.append(f"Consider adding the primary keyword '{primary_keyword}' to the content")
        
        # Engagement suggestions
        if "?" not in content:
            suggestions.append("Consider adding questions to increase reader engagement")
        
        if not re.search(r'^\d+\.|^•|^\*', content, re.MULTILINE):
            suggestions.append("Add bullet points or numbered lists to improve readability")
        
        return suggestions