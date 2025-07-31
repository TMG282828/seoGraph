"""
SEO research-related API routes for keyword research, competitor analysis, and trends.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
import logging
import json
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/seo-research", tags=["seo-research"])

@router.post("/keywords")
async def research_keywords(request: dict):
    """Research keywords using Google Ads API with algorithmic fallback."""
    try:
        keywords = request.get("keywords", [])
        country = request.get("country", "US")
        language = request.get("language", "en")
        
        if not keywords:
            raise HTTPException(status_code=400, detail="Keywords are required")
        
        # Normalize keywords
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]
        
        # Use Google Ads service for real data
        try:
            from src.services.google_ads_service import google_ads_service
            
            # Get data from Google Ads service (with algorithmic fallback)
            keyword_data = await google_ads_service.get_keyword_ideas(keywords, country, language)
            
            # Transform to expected format
            results = []
            for kw_data in keyword_data.get("keywords", []):
                results.append({
                    "term": kw_data["term"],
                    "volume": kw_data["volume"],
                    "difficulty": kw_data["difficulty"],
                    "opportunity": kw_data["opportunity"],
                    "intent": kw_data["intent"],
                    "related_keywords": kw_data["related_keywords"],
                    "cpc": kw_data.get("cpc", 0),
                    "competition": kw_data.get("competition", "UNKNOWN"),
                    "data_source": kw_data["data_source"],
                    "confidence_score": kw_data["confidence_score"],
                    "last_updated": kw_data["last_updated"],
                    "analysis_context": f"Generated for {country}-{language} market"
                })
            
            return {
                "success": True,
                "keywords": results,
                "total_results": len(results),
                "country": country,
                "language": language,
                "data_source": keyword_data.get("data_source", "Enhanced Algorithmic Analysis"),
                "api_status": keyword_data.get("api_status", "fallback_mode"),
                "note": keyword_data.get("note", "Using enhanced algorithmic analysis")
            }
            
        except ImportError:
            logger.error("Google Ads service not available, using basic fallback")
            # Basic fallback if service isn't available
            return await _basic_keyword_fallback(keywords, country, language)
        
    except Exception as e:
        logger.error(f"Keyword research failed: {e}")
        raise HTTPException(status_code=500, detail=f"Keyword research failed: {str(e)}")

async def _basic_keyword_fallback(keywords: List[str], country: str, language: str):
    """Basic fallback for keyword research when services are unavailable."""
    import random
    from datetime import datetime
    
    results = []
    for keyword in keywords:
        if isinstance(keyword, str) and keyword.strip():
            keyword = keyword.strip().lower()
            
            # Generate basic data
            seed = hash(keyword + str(datetime.now().date()) + country + language) % 1000
            random.seed(seed)
            
            base_volume = random.randint(500, 50000)
            base_difficulty = random.randint(20, 80)
            
            results.append({
                "term": keyword,
                "volume": base_volume,
                "difficulty": base_difficulty,
                "opportunity": "Medium",
                "intent": "Informational",
                "related_keywords": [f"{keyword} guide", f"how to {keyword}"],
                "cpc": round(random.uniform(0.5, 5.0), 2),
                "competition": "MEDIUM",
                "data_source": "basic_fallback",
                "confidence_score": 0.3,
                "last_updated": datetime.now().isoformat(),
                "analysis_context": f"Basic fallback for {country}-{language} market"
            })
    
    return {
        "success": True,
        "keywords": results,
        "total_results": len(results),
        "country": country,
        "language": language,
        "data_source": "Basic Fallback Analysis",
        "api_status": "service_unavailable",
        "note": "⚠️ Using basic fallback - Google Ads service unavailable"
    }

@router.post("/competitors")
async def analyze_competitors(request: dict):
    """Analyze competitors using SEO data sources."""
    try:
        competitors = request.get("competitors", [])
        keywords = request.get("keywords", [])
        
        if not competitors:
            raise HTTPException(status_code=400, detail="Competitors are required")
        
        # Mock competitor analysis data
        results = []
        for competitor in competitors:
            if isinstance(competitor, str) and competitor.strip():
                results.append({
                    "domain": competitor.strip(),
                    "domain_authority": max(10, hash(competitor) % 100),
                    "organic_traffic": max(1000, hash(competitor) % 1000000),
                    "keywords_ranking": max(100, hash(competitor) % 10000),
                    "backlinks": max(500, hash(competitor) % 50000),
                    "content_score": max(20, hash(competitor) % 100),
                    "top_keywords": [
                        {"keyword": f"{competitor} review", "position": 3, "volume": 12000},
                        {"keyword": f"{competitor} pricing", "position": 7, "volume": 8500},
                        {"keyword": f"best {competitor} alternative", "position": 12, "volume": 5400}
                    ]
                })
        
        return {
            "success": True,
            "competitors": results,
            "total_analyzed": len(results),
            "note": "Demo data - Connect SEO tools API for real competitor analysis"
        }
        
    except Exception as e:
        logger.error(f"Competitor analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Competitor analysis failed: {str(e)}")

@router.post("/content-gaps")
async def analyze_content_gaps(request: dict):
    """Analyze content gaps using competitor and keyword data."""
    try:
        competitors = request.get("competitors", [])
        keywords = request.get("keywords", [])
        
        # Generate dynamic content gap analysis based on input data
        import random
        from datetime import datetime
        
        # Analyze provided keywords and competitors to generate contextual gaps
        seed = hash(str(competitors) + str(keywords) + str(datetime.now().date())) % 1000
        random.seed(seed)
        
        gaps = []
        
        # If we have keywords, generate gaps based on them
        if keywords:
            keyword_themes = []
            for kw in keywords[:5]:  # Analyze top 5 keywords
                # Extract theme from keyword
                if any(word in kw.lower() for word in ['seo', 'search', 'ranking']):
                    keyword_themes.append('SEO')
                elif any(word in kw.lower() for word in ['content', 'blog', 'article']):
                    keyword_themes.append('Content Marketing')
                elif any(word in kw.lower() for word in ['local', 'business', 'store']):
                    keyword_themes.append('Local Business')
                elif any(word in kw.lower() for word in ['ecommerce', 'shop', 'product']):
                    keyword_themes.append('E-commerce')
                else:
                    keyword_themes.append('Digital Marketing')
            
            # Generate gaps for each unique theme
            unique_themes = list(set(keyword_themes))
            for theme in unique_themes[:3]:  # Limit to 3 gaps
                
                # Dynamic opportunity scoring based on theme and competition
                base_score = random.randint(60, 95)
                competitor_count = len(competitors) if competitors else random.randint(1, 5)
                
                # Adjust score based on competition level
                if competitor_count > 3:
                    base_score = max(60, base_score - 15)
                
                # Generate realistic search volume
                volume = random.randint(2000, 50000)
                
                # Determine content type based on theme
                content_types = {
                    'SEO': ['Technical Guide', 'Comprehensive Tutorial', 'Case Study Series'],
                    'Content Marketing': ['Strategy Guide', 'Template Collection', 'Best Practices'],
                    'Local Business': ['Local SEO Guide', 'GMB Optimization', 'Citation Building'],
                    'E-commerce': ['Product SEO Guide', 'Category Optimization', 'Conversion Guide'],
                    'Digital Marketing': ['Multi-Channel Guide', 'Analytics Setup', 'ROI Optimization']
                }
                
                suggested_type = random.choice(content_types.get(theme, ['Comprehensive Guide']))
                
                # Generate related keywords for the theme
                theme_keywords = []
                if theme == 'SEO':
                    theme_keywords = [f"{kw} seo" for kw in keywords[:3] if kw] + ['technical seo', 'on-page optimization']
                elif theme == 'Content Marketing':
                    theme_keywords = [f"{kw} content" for kw in keywords[:3] if kw] + ['content strategy', 'blog optimization']
                elif theme == 'Local Business':
                    theme_keywords = [f"local {kw}" for kw in keywords[:3] if kw] + ['google my business', 'local citations']
                else:
                    theme_keywords = keywords[:3] if keywords else [f"{theme.lower()} optimization"]
                
                gaps.append({
                    "topic": f"{theme} Content Gap Analysis",
                    "content_angle": f"Leverage {theme.lower()} opportunities your competitors are missing",
                    "opportunity_score": base_score,
                    "search_volume": volume,
                    "difficulty": random.randint(25, 75),
                    "competitors_covering": competitor_count,
                    "suggested_content_type": suggested_type,
                    "keywords": theme_keywords[:5],
                    "analysis_context": f"Based on {len(keywords) if keywords else 0} seed keywords and {competitor_count} competitors"
                })
        else:
            # Generate general gaps if no specific input
            general_gaps = [
                {
                    "topic": "AI-Powered SEO Strategy",
                    "content_angle": "How AI is revolutionizing search optimization",
                    "opportunity_score": random.randint(75, 95),
                    "search_volume": random.randint(8000, 25000),
                    "difficulty": random.randint(40, 70),
                    "competitors_covering": random.randint(2, 6),
                    "suggested_content_type": "Future-Focused Guide",
                    "keywords": ["ai seo", "machine learning seo", "automated optimization"],
                    "analysis_context": "Generated based on current market trends"
                }
            ]
            gaps = general_gaps
        
        return {
            "success": True,
            "content_gaps": gaps,
            "total_opportunities": len(gaps),
            "note": "Demo data - Real analysis requires competitor content crawling"
        }
        
    except Exception as e:
        logger.error(f"Content gap analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Content gap analysis failed: {str(e)}")

@router.get("/trends")
async def get_trends_analysis(
    keywords: Optional[str] = None,
    timeframe: str = "1m",
    geo: str = "US"
):
    """Get trending topics using pytrends integration with rate limiting and error handling."""
    import time
    import asyncio
    
    try:
        from pytrends.request import TrendReq
        
        # Add delay to prevent rate limiting
        await asyncio.sleep(0.5)
        
        # Initialize pytrends with better error handling
        pytrends = TrendReq(
            hl='en-US', 
            tz=360, 
            timeout=(15, 30),  # Increased timeout
            backoff_factor=0.1,  # Add backoff
            requests_args={'verify': False}  # Skip SSL verification if needed
        )
        
        if keywords:
            # Get trends for specific keywords
            keyword_list = [k.strip() for k in keywords.split(",")][:5]  # Limit to 5 keywords
            
            # Try multiple times with increasing delays
            for attempt in range(3):
                try:
                    pytrends.build_payload(
                        keyword_list, 
                        cat=0, 
                        timeframe=f'today {timeframe}', 
                        geo=geo, 
                        gprop=''
                    )
                    
                    # Get interest over time
                    interest_data = pytrends.interest_over_time()
                    break  # Success, exit retry loop
                    
                except Exception as retry_error:
                    if attempt < 2:  # If not last attempt
                        logger.warning(f"Trends API attempt {attempt + 1} failed: {retry_error}, retrying...")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        raise retry_error  # Re-raise on final attempt
            
            # Convert to JSON-serializable format
            trends_data = []
            if not interest_data.empty:
                for keyword in keyword_list:
                    if keyword in interest_data.columns:
                        trends_data.append({
                            "keyword": keyword,
                            "data_points": [
                                {
                                    "date": date.strftime("%Y-%m-%d"),
                                    "value": int(value)
                                }
                                for date, value in interest_data[keyword].items()
                            ],
                            "average_interest": float(interest_data[keyword].mean())
                        })
        else:
            # Get trending searches
            trending_searches = pytrends.trending_searches(pn=geo.lower())
            trends_data = [
                {
                    "keyword": keyword,
                    "trend_type": "trending_search",
                    "region": geo
                }
                for keyword in trending_searches[0].head(10).tolist()
            ]
        
        return {
            "success": True,
            "trends": trends_data,
            "timeframe": timeframe,
            "geo": geo
        }
        
    except ImportError:
        return {
            "success": False,
            "error": "Install with: pip install pytrends"
        }
    except Exception as e:
        logger.error(f"Trends analysis failed: {e}")
        # Generate dynamic synthetic data instead of static mock data
        import random
        from datetime import datetime, timedelta
        
        keywords_list = keywords.split(",") if keywords else ["unknown"]
        trends_data = []
        
        for keyword in keywords_list[:3]:  # Limit to 3 keywords
            keyword = keyword.strip()
            # Generate dynamic data points based on keyword hash and current time
            base_seed = hash(keyword + str(datetime.now().date())) % 1000
            random.seed(base_seed)
            
            # Generate realistic trend data with some variance
            base_interest = random.randint(20, 90)
            data_points = []
            
            # Generate data points for the timeframe
            num_points = {"1m": 4, "3m": 12, "12m": 52, "5y": 260}.get(timeframe, 4)
            
            for i in range(num_points):
                # Create dates going backwards from today
                days_back = (num_points - i - 1) * {"1m": 7, "3m": 7, "12m": 7, "5y": 7}.get(timeframe, 7)
                date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
                
                # Add realistic variance to the base interest
                variance = random.randint(-15, 15)
                value = max(0, min(100, base_interest + variance))
                data_points.append({"date": date, "value": value})
            
            trends_data.append({
                "keyword": keyword,
                "data_points": data_points,
                "average_interest": sum(point["value"] for point in data_points) / len(data_points)
            })
        
        return {
            "success": True,
            "trends": trends_data,
            "timeframe": timeframe,
            "geo": geo,
            "note": f"Generated dynamic trends for {geo} - Real Google Trends API connection failed"
        }

@router.get("/suggested-keywords")
async def get_suggested_keywords():
    """Get AI-powered suggested keywords based on user account/onboarding data."""
    try:
        from src.services.openai_seo import openai_seo_service
        
        # Generate intelligent keyword suggestions using OpenAI
        # TODO: Get actual user industry/business type from onboarding data
        user_context = {
            "industry": "technology",  # Default, should come from user profile
            "business_type": "content_marketing",  # Should come from onboarding
            "target_audience": "business_professionals",  # Should come from user data
            "goals": ["increase_traffic", "improve_rankings", "content_optimization"]
        }
        
        # Create a prompt for OpenAI to generate relevant keywords
        prompt = f"""
        Generate 15 high-value keyword suggestions for a {user_context['business_type']} business in the {user_context['industry']} industry.
        
        Target Audience: {user_context['target_audience']}
        Business Goals: {', '.join(user_context['goals'])}
        
        For each keyword, provide:
        - keyword: The actual keyword phrase
        - category: Category (commercial, informational, transactional, navigational)
        - confidence: Float 0.0-1.0 indicating how relevant this keyword is
        - estimated_volume: Estimated monthly search volume
        - difficulty: SEO difficulty estimate (1-100)
        - intent: User search intent (buy, learn, find, compare)
        - reason: Brief explanation why this keyword is valuable
        
        Focus on keywords with:
        - Good search volume (1000+ monthly searches)
        - Moderate competition (difficulty 30-70)
        - High commercial or informational value
        - Relevance to content marketing and SEO
        
        Return as a JSON-like structure that can be parsed.
        """
        
        try:
            # Use OpenAI for intelligent suggestions
            response = openai_seo_service.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert SEO keyword strategist. Provide high-value keyword suggestions with detailed analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.4
            )
            
            ai_response = response.choices[0].message.content
            
            # Parse AI response (simplified parsing - in production you'd want more robust parsing)
            suggested_keywords = await _parse_ai_keyword_suggestions(ai_response)
            
            return {
                "success": True,
                "suggested_keywords": suggested_keywords[:15],
                "source": "openai_analysis",
                "note": "✨ AI-powered keyword suggestions based on your business profile",
                "model": "gpt-4o-mini"
            }
            
        except Exception as ai_error:
            logger.warning(f"OpenAI keyword generation failed, using fallback: {ai_error}")
            # Fallback to enhanced algorithmic approach
            return await _get_fallback_keyword_suggestions(user_context)
        
    except Exception as e:
        logger.error(f"Failed to get suggested keywords: {e}")
        return await _get_fallback_keyword_suggestions({})

async def _parse_ai_keyword_suggestions(ai_response: str) -> List[Dict[str, Any]]:
    """Parse AI response into structured keyword suggestions."""
    import json
    import re
    
    suggestions = []
    
    try:
        # Try to extract JSON-like structures from the response
        lines = ai_response.split('\n')
        current_keyword = {}
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Look for keyword patterns
            if 'keyword:' in line.lower():
                if current_keyword.get('keyword'):
                    suggestions.append(current_keyword)
                current_keyword = {
                    'keyword': _extract_value(line, ['keyword:']),
                    'category': 'informational',
                    'confidence': 0.8,
                    'estimated_volume': 5000,
                    'difficulty': 50,
                    'intent': 'learn',
                    'reason': 'AI-generated suggestion'
                }
            elif 'category:' in line.lower():
                current_keyword['category'] = _extract_value(line, ['category:']).lower()
            elif 'confidence:' in line.lower():
                try:
                    current_keyword['confidence'] = float(_extract_value(line, ['confidence:']))
                except:
                    pass
            elif 'volume:' in line.lower() or 'estimated_volume:' in line.lower():
                try:
                    volume_str = _extract_value(line, ['volume:', 'estimated_volume:'])
                    # Extract numbers from string (handle formats like "5,000" or "5K")
                    volume_match = re.search(r'(\d+(?:,\d+)*)', volume_str)
                    if volume_match:
                        volume = int(volume_match.group(1).replace(',', ''))
                        if 'k' in volume_str.lower():
                            volume *= 1000
                        current_keyword['estimated_volume'] = volume
                except:
                    pass
            elif 'difficulty:' in line.lower():
                try:
                    current_keyword['difficulty'] = int(_extract_value(line, ['difficulty:']))
                except:
                    pass
            elif 'intent:' in line.lower():
                current_keyword['intent'] = _extract_value(line, ['intent:']).lower()
            elif 'reason:' in line.lower():
                current_keyword['reason'] = _extract_value(line, ['reason:'])
        
        if current_keyword.get('keyword'):
            suggestions.append(current_keyword)
        
        # If parsing didn't work well, create some default suggestions
        if len(suggestions) < 5:
            base_keywords = ["content marketing", "SEO strategy", "digital marketing", "keyword research", "content optimization"]
            for i, keyword in enumerate(base_keywords):
                if i >= len(suggestions):
                    suggestions.append({
                        'keyword': keyword,
                        'category': 'informational',
                        'confidence': 0.75,
                        'estimated_volume': 8000,
                        'difficulty': 45,
                        'intent': 'learn',
                        'reason': 'High-value content marketing keyword'
                    })
        
        return suggestions[:15]
        
    except Exception as e:
        logger.error(f"Failed to parse AI keyword suggestions: {e}")
        return []

def _extract_value(line: str, prefixes: List[str]) -> str:
    """Extract value after specified prefixes."""
    for prefix in prefixes:
        if prefix in line.lower():
            return line.lower().split(prefix, 1)[1].strip()
    return line.strip()

async def _get_fallback_keyword_suggestions(user_context: Dict[str, Any]) -> Dict[str, Any]:
    """Get enhanced fallback keyword suggestions when OpenAI is unavailable."""
    import random
    from datetime import datetime
    
    # Enhanced keyword categories with more intelligence
    keyword_categories = {
        "SEO & Content": [
            {"keyword": "SEO content strategy", "volume": 12000, "difficulty": 55, "intent": "learn"},
            {"keyword": "content marketing ROI", "volume": 8900, "difficulty": 48, "intent": "analyze"},
            {"keyword": "keyword research tools", "volume": 22000, "difficulty": 65, "intent": "compare"},
            {"keyword": "technical SEO audit", "volume": 15000, "difficulty": 58, "intent": "learn"},
            {"keyword": "local SEO optimization", "volume": 18500, "difficulty": 42, "intent": "implement"}
        ],
        "Digital Marketing": [
            {"keyword": "digital marketing strategy", "volume": 27000, "difficulty": 72, "intent": "learn"},
            {"keyword": "marketing automation tools", "volume": 14000, "difficulty": 60, "intent": "compare"},
            {"keyword": "social media marketing", "volume": 33000, "difficulty": 68, "intent": "learn"},
            {"keyword": "email marketing campaigns", "volume": 19000, "difficulty": 45, "intent": "implement"},
            {"keyword": "conversion rate optimization", "volume": 16000, "difficulty": 55, "intent": "improve"}
        ],
        "Business Intelligence": [
            {"keyword": "business analytics dashboard", "volume": 11000, "difficulty": 52, "intent": "find"},
            {"keyword": "data-driven marketing", "volume": 9500, "difficulty": 48, "intent": "learn"},
            {"keyword": "customer analytics platform", "volume": 7800, "difficulty": 58, "intent": "compare"},
            {"keyword": "marketing metrics tracking", "volume": 13000, "difficulty": 44, "intent": "implement"},
            {"keyword": "ROI measurement tools", "volume": 8200, "difficulty": 50, "intent": "find"}
        ]
    }
    
    # Generate diverse suggestions
    suggested_keywords = []
    
    for category, keywords in keyword_categories.items():
        selected = random.sample(keywords, min(5, len(keywords)))
        for kw_data in selected:
            suggested_keywords.append({
                "keyword": kw_data["keyword"],
                "category": category.lower().replace(" & ", "_").replace(" ", "_"),
                "confidence": random.uniform(0.75, 0.92),
                "estimated_volume": kw_data["volume"],
                "difficulty": kw_data["difficulty"],
                "intent": kw_data["intent"],
                "reason": f"High-value {category.lower()} keyword with good search potential",
                "source": "enhanced_algorithm"
            })
    
    return {
        "success": True,
        "suggested_keywords": suggested_keywords[:15],
        "source": "enhanced_algorithm", 
        "note": "⚡ Enhanced algorithmic suggestions - Enable OpenAI for AI-powered recommendations"
    }

@router.post("/generate-report")
async def generate_seo_report(report_data: dict):
    """Generate an AI-powered SEO research report using OpenAI."""
    try:
        # Import OpenAI service
        try:
            from src.services.openai_seo import openai_seo_service
        except ImportError:
            logger.warning("OpenAI service not available, using template report")
            return await generate_template_report(report_data)
        
        # Extract data for analysis
        keywords = report_data.get('keywords', [])
        competitors = report_data.get('competitors', [])
        content_gaps = report_data.get('content_gaps', [])
        trends = report_data.get('trends', [])
        filters = report_data.get('filters', {})
        
        # Create comprehensive context for OpenAI
        analysis_context = {
            "market": f"{filters.get('country', 'US')}-{filters.get('language', 'en')}",
            "timeframe": filters.get('timeframe', '1m'),
            "keyword_count": len(keywords),
            "competitor_count": len(competitors),
            "gap_count": len(content_gaps),
            "trend_count": len(trends),
            "top_keywords": [kw.get('term') if isinstance(kw, dict) else str(kw) for kw in keywords[:5]] if keywords else [],
            "competitor_domains": [comp.get('domain') if isinstance(comp, dict) else str(comp) for comp in competitors[:3]] if competitors else [],
            "key_opportunities": [gap.get('topic') if isinstance(gap, dict) else str(gap) for gap in content_gaps[:3]] if content_gaps else []
        }
        
        # Generate AI-powered analysis
        prompt = f"""
        You are an expert SEO strategist analyzing comprehensive research data. Generate an intelligent, actionable SEO research report.

        **Research Context:**
        - Market: {analysis_context['market']}
        - Analysis Period: {analysis_context['timeframe']}
        - Keywords Analyzed: {analysis_context['keyword_count']}
        - Competitors Analyzed: {analysis_context['competitor_count']}
        - Content Gaps Identified: {analysis_context['gap_count']}

        **Key Data Points:**
        - Top Keywords: {', '.join(analysis_context['top_keywords'])}
        - Main Competitors: {', '.join(analysis_context['competitor_domains'])}
        - Primary Opportunities: {', '.join(analysis_context['key_opportunities'])}

        **Detailed Research Data:**
        Keywords: {json.dumps(keywords[:10], indent=2) if keywords else 'None'}
        
        Content Gaps: {json.dumps(content_gaps[:5], indent=2) if content_gaps else 'None'}
        
        Trends: {json.dumps(trends[:3], indent=2) if trends else 'None'}

        Generate a comprehensive SEO strategy report in HTML format with:

        1. **Executive Summary** - Key findings and strategic recommendations
        2. **Keyword Strategy Analysis** - Insights from keyword data with specific recommendations
        3. **Competitive Landscape** - Analysis of competitor positioning and opportunities
        4. **Content Gap Opportunities** - Prioritized content recommendations with rationale
        5. **Trend Analysis & Predictions** - Market trends and future opportunities
        6. **Action Plan** - Specific, prioritized next steps with timelines
        7. **Success Metrics** - KPIs to track progress

        Make the report:
        - Actionable with specific recommendations
        - Data-driven with insights from the provided research
        - Strategic with clear reasoning
        - Professional with proper HTML formatting
        - Include specific numbers and metrics from the data
        - Provide clear priority levels for recommendations

        Return only the HTML content without any markdown formatting.
        """

        try:
            # Generate AI report
            ai_response = await openai_seo_service.generate_seo_analysis(
                prompt=prompt,
                context="seo_research_report"
            )
            
            if ai_response.get('success') and ai_response.get('analysis'):
                html_content = ai_response['analysis']
                
                # Enhance HTML with professional styling
                styled_html = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>AI-Powered SEO Research Report</title>
                    <style>
                        body {{ 
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                            line-height: 1.6; 
                            margin: 0; 
                            padding: 40px;
                            background: #f8f9fa;
                        }}
                        .container {{ 
                            max-width: 1000px; 
                            margin: 0 auto; 
                            background: white;
                            padding: 40px;
                            border-radius: 8px;
                            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        }}
                        h1 {{ color: #1f2937; border-bottom: 3px solid #3b82f6; padding-bottom: 10px; }}
                        h2 {{ color: #374151; margin-top: 30px; }}
                        h3 {{ color: #4b5563; }}
                        .metric {{ 
                            background: #eff6ff; 
                            padding: 15px; 
                            border-left: 4px solid #3b82f6;
                            margin: 10px 0;
                        }}
                        .opportunity {{ 
                            background: #f0f9f4; 
                            padding: 15px; 
                            border-left: 4px solid #10b981;
                            margin: 10px 0;
                        }}
                        .warning {{ 
                            background: #fef3c7; 
                            padding: 15px; 
                            border-left: 4px solid #f59e0b;
                            margin: 10px 0;
                        }}
                        table {{ 
                            width: 100%; 
                            border-collapse: collapse; 
                            margin: 20px 0;
                        }}
                        th, td {{ 
                            padding: 12px; 
                            text-align: left; 
                            border-bottom: 1px solid #e5e7eb;
                        }}
                        th {{ 
                            background: #f3f4f6; 
                            font-weight: 600;
                        }}
                        .footer {{
                            margin-top: 40px;
                            padding-top: 20px;
                            border-top: 1px solid #e5e7eb;
                            color: #6b7280;
                            font-size: 14px;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        {html_content}
                        <div class="footer">
                            <p><strong>Report generated on:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                            <p><strong>Generated by:</strong> AI-Powered SEO Content Knowledge Graph System</p>
                            <p><strong>Analysis Context:</strong> {analysis_context['market']} market, {analysis_context['timeframe']} timeframe</p>
                        </div>
                    </div>
                </body>
                </html>
                """
                
                # Return HTML report
                return StreamingResponse(
                    io.BytesIO(styled_html.encode('utf-8')),
                    media_type="text/html",
                    headers={"Content-Disposition": "attachment; filename=ai-seo-research-report.html"}
                )
            else:
                logger.warning("AI analysis failed, falling back to template")
                return await generate_template_report(report_data)
                
        except Exception as ai_error:
            logger.error(f"AI report generation failed: {ai_error}")
            return await generate_template_report(report_data)
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

async def generate_template_report(report_data: dict):
    """Fallback template report when AI is unavailable."""
    keywords = report_data.get('keywords', [])
    gaps = report_data.get('content_gaps', [])
    trends = report_data.get('trends', [])
    filters = report_data.get('filters', {})
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SEO Research Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #1f2937; }}
            .summary {{ background: #f3f4f6; padding: 20px; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <h1>SEO Research Report</h1>
        <div class="summary">
            <h2>Executive Summary</h2>
            <p>Analysis completed for {filters.get('country', 'US')} market with {len(keywords)} keywords researched.</p>
            <p>Key findings: {len(gaps)} content opportunities identified with {len(trends)} trend patterns analyzed.</p>
        </div>
        
        <h2>Keyword Analysis</h2>
        {"".join([f"<p><strong>{kw.get('term')}</strong> - Volume: {kw.get('volume')}, Difficulty: {kw.get('difficulty')}</p>" for kw in keywords[:10]])}
        
        <h2>Content Opportunities</h2>
        {"".join([f"<p><strong>{gap.get('topic')}</strong> - Score: {gap.get('opportunity_score')}/100</p>" for gap in gaps[:5]])}
    </body>
    </html>
    """
    
    return StreamingResponse(
        io.BytesIO(html_content.encode('utf-8')),
        media_type="text/html",
        headers={"Content-Disposition": "attachment; filename=seo-research-report.html"}
    )