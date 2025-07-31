"""
Tests for the Trend Analysis Agent module.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone
import asyncio

from agents.trend_analysis import TrendAnalysisAgent, TrendData, TrendInsight


class TestTrendAnalysisAgent:
    """Test suite for TrendAnalysisAgent."""

    def test_trend_analysis_agent_initialization(self):
        """Test TrendAnalysisAgent initialization."""
        agent = TrendAnalysisAgent()
        
        assert agent is not None
        assert hasattr(agent, 'pytrends')
        assert hasattr(agent, 'neo4j_client')
        assert hasattr(agent, 'cache')

    def test_trend_analysis_agent_default_config(self):
        """Test TrendAnalysisAgent with default configuration."""
        agent = TrendAnalysisAgent()
        
        # Check default values
        assert agent.hl == 'en-US'
        assert agent.tz == 360
        assert agent.max_keywords == 5

    def test_trend_analysis_agent_custom_config(self):
        """Test TrendAnalysisAgent with custom configuration."""
        agent = TrendAnalysisAgent(
            hl='en-GB',
            tz=0,
            max_keywords=10
        )
        
        assert agent.hl == 'en-GB'
        assert agent.tz == 0
        assert agent.max_keywords == 10

    @patch('agents.trend_analysis.TrendReq')
    def test_trend_analysis_agent_pytrends_initialization(self, mock_trend_req):
        """Test pytrends initialization."""
        mock_pytrends = MagicMock()
        mock_trend_req.return_value = mock_pytrends
        
        agent = TrendAnalysisAgent()
        
        # Verify pytrends was initialized with correct parameters
        mock_trend_req.assert_called_once_with(hl='en-US', tz=360)

    @patch('agents.trend_analysis.TrendReq')
    def test_get_trending_keywords(self, mock_trend_req):
        """Test getting trending keywords."""
        mock_pytrends = MagicMock()
        mock_trend_req.return_value = mock_pytrends
        
        # Mock trending searches response
        mock_trending_data = {
            'title': ['AI content marketing', 'SEO optimization', 'content strategy'],
            'traffic': [100000, 80000, 60000],
            'related_queries': [
                ['AI writing', 'content AI'],
                ['SEO tools', 'SEO strategy'],
                ['content planning', 'content calendar']
            ]
        }
        mock_pytrends.trending_searches.return_value = mock_trending_data
        
        agent = TrendAnalysisAgent()
        
        # Get trending keywords
        trending_keywords = agent.get_trending_keywords()
        
        # Verify trending searches was called
        mock_pytrends.trending_searches.assert_called_once()
        
        # Verify result format
        assert isinstance(trending_keywords, list)
        assert len(trending_keywords) <= agent.max_keywords

    @patch('agents.trend_analysis.TrendReq')
    def test_get_keyword_trends(self, mock_trend_req):
        """Test getting keyword trends."""
        mock_pytrends = MagicMock()
        mock_trend_req.return_value = mock_pytrends
        
        # Mock interest over time response
        mock_interest_data = {
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'AI content marketing': [50, 60, 70],
            'SEO optimization': [40, 45, 50]
        }
        mock_pytrends.interest_over_time.return_value = mock_interest_data
        
        agent = TrendAnalysisAgent()
        
        # Get keyword trends
        keywords = ['AI content marketing', 'SEO optimization']
        trends = agent.get_keyword_trends(keywords)
        
        # Verify build_payload and interest_over_time were called
        mock_pytrends.build_payload.assert_called_once_with(
            keywords, 
            cat=0, 
            timeframe='today 3-m', 
            geo='', 
            gprop=''
        )
        mock_pytrends.interest_over_time.assert_called_once()
        
        # Verify result format
        assert isinstance(trends, dict)

    @patch('agents.trend_analysis.TrendReq')
    def test_get_related_queries(self, mock_trend_req):
        """Test getting related queries."""
        mock_pytrends = MagicMock()
        mock_trend_req.return_value = mock_pytrends
        
        # Mock related queries response
        mock_related_data = {
            'AI content marketing': {
                'top': {
                    'query': ['AI writing', 'content AI', 'AI tools'],
                    'value': [100, 80, 60]
                },
                'rising': {
                    'query': ['new AI content', 'AI content tools'],
                    'value': [200, 150]
                }
            }
        }
        mock_pytrends.related_queries.return_value = mock_related_data
        
        agent = TrendAnalysisAgent()
        
        # Get related queries
        keyword = 'AI content marketing'
        related = agent.get_related_queries(keyword)
        
        # Verify build_payload and related_queries were called
        mock_pytrends.build_payload.assert_called_once_with([keyword])
        mock_pytrends.related_queries.assert_called_once()
        
        # Verify result format
        assert isinstance(related, dict)

    @patch('agents.trend_analysis.TrendReq')
    def test_analyze_trend_data(self, mock_trend_req):
        """Test analyzing trend data."""
        mock_pytrends = MagicMock()
        mock_trend_req.return_value = mock_pytrends
        
        agent = TrendAnalysisAgent()
        
        # Mock trend data
        trend_data = {
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'AI content marketing': [50, 60, 70],
            'SEO optimization': [40, 45, 50]
        }
        
        # Analyze trend data
        analysis = agent.analyze_trend_data(trend_data)
        
        # Verify result format
        assert isinstance(analysis, dict)
        assert 'trends' in analysis
        assert 'insights' in analysis
        assert 'recommendations' in analysis

    @patch('agents.trend_analysis.TrendReq')
    def test_generate_trend_insights(self, mock_trend_req):
        """Test generating trend insights."""
        mock_pytrends = MagicMock()
        mock_trend_req.return_value = mock_pytrends
        
        agent = TrendAnalysisAgent()
        
        # Mock keywords and timeframe
        keywords = ['AI content marketing', 'SEO optimization']
        timeframe = 'today 3-m'
        
        # Generate insights
        insights = agent.generate_trend_insights(keywords, timeframe)
        
        # Verify result format
        assert isinstance(insights, list)
        for insight in insights:
            assert isinstance(insight, TrendInsight)
            assert hasattr(insight, 'keyword')
            assert hasattr(insight, 'trend_score')
            assert hasattr(insight, 'insight_type')
            assert hasattr(insight, 'description')

    @patch('agents.trend_analysis.TrendReq')
    def test_get_seasonal_trends(self, mock_trend_req):
        """Test getting seasonal trends."""
        mock_pytrends = MagicMock()
        mock_trend_req.return_value = mock_pytrends
        
        # Mock seasonal data
        mock_seasonal_data = {
            'date': ['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01'],
            'AI content marketing': [50, 70, 60, 80],
            'SEO optimization': [40, 45, 50, 55]
        }
        mock_pytrends.interest_over_time.return_value = mock_seasonal_data
        
        agent = TrendAnalysisAgent()
        
        # Get seasonal trends
        keywords = ['AI content marketing', 'SEO optimization']
        seasonal = agent.get_seasonal_trends(keywords)
        
        # Verify result format
        assert isinstance(seasonal, dict)
        assert 'seasonal_patterns' in seasonal
        assert 'peak_periods' in seasonal
        assert 'recommendations' in seasonal

    @patch('agents.trend_analysis.TrendReq')
    def test_predict_trend_trajectory(self, mock_trend_req):
        """Test predicting trend trajectory."""
        mock_pytrends = MagicMock()
        mock_trend_req.return_value = mock_pytrends
        
        agent = TrendAnalysisAgent()
        
        # Mock historical data
        historical_data = [50, 55, 60, 65, 70, 75, 80]
        keyword = 'AI content marketing'
        
        # Predict trajectory
        prediction = agent.predict_trend_trajectory(keyword, historical_data)
        
        # Verify result format
        assert isinstance(prediction, dict)
        assert 'predicted_values' in prediction
        assert 'trend_direction' in prediction
        assert 'confidence_score' in prediction

    @patch('agents.trend_analysis.TrendReq')
    def test_compare_keywords(self, mock_trend_req):
        """Test comparing keywords."""
        mock_pytrends = MagicMock()
        mock_trend_req.return_value = mock_pytrends
        
        # Mock comparison data
        mock_comparison_data = {
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'AI content marketing': [50, 60, 70],
            'SEO optimization': [40, 45, 50],
            'content strategy': [30, 35, 40]
        }
        mock_pytrends.interest_over_time.return_value = mock_comparison_data
        
        agent = TrendAnalysisAgent()
        
        # Compare keywords
        keywords = ['AI content marketing', 'SEO optimization', 'content strategy']
        comparison = agent.compare_keywords(keywords)
        
        # Verify result format
        assert isinstance(comparison, dict)
        assert 'comparison_data' in comparison
        assert 'rankings' in comparison
        assert 'insights' in comparison


class TestTrendData:
    """Test suite for TrendData model."""

    def test_trend_data_creation(self):
        """Test creation of TrendData with valid data."""
        trend_data = TrendData(
            keyword="AI content marketing",
            values=[50, 60, 70, 80],
            dates=["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            trend_score=0.75,
            region="US"
        )
        
        assert trend_data.keyword == "AI content marketing"
        assert trend_data.values == [50, 60, 70, 80]
        assert trend_data.dates == ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
        assert trend_data.trend_score == 0.75
        assert trend_data.region == "US"

    def test_trend_data_defaults(self):
        """Test TrendData with default values."""
        trend_data = TrendData(
            keyword="AI content marketing",
            values=[50, 60, 70],
            dates=["2024-01-01", "2024-01-02", "2024-01-03"]
        )
        
        assert trend_data.trend_score == 0.0  # Default value
        assert trend_data.region == "US"  # Default value

    def test_trend_data_validation(self):
        """Test TrendData validation."""
        # Test mismatched values and dates length
        with pytest.raises(ValueError, match="Values and dates must have the same length"):
            TrendData(
                keyword="AI content marketing",
                values=[50, 60, 70],
                dates=["2024-01-01", "2024-01-02"]  # Different length
            )


class TestTrendInsight:
    """Test suite for TrendInsight model."""

    def test_trend_insight_creation(self):
        """Test creation of TrendInsight with valid data."""
        insight = TrendInsight(
            keyword="AI content marketing",
            trend_score=0.85,
            insight_type="rising",
            description="This keyword is showing strong upward momentum",
            confidence=0.9
        )
        
        assert insight.keyword == "AI content marketing"
        assert insight.trend_score == 0.85
        assert insight.insight_type == "rising"
        assert insight.description == "This keyword is showing strong upward momentum"
        assert insight.confidence == 0.9

    def test_trend_insight_defaults(self):
        """Test TrendInsight with default values."""
        insight = TrendInsight(
            keyword="AI content marketing",
            trend_score=0.85,
            insight_type="rising",
            description="This keyword is showing strong upward momentum"
        )
        
        assert insight.confidence == 0.5  # Default value
        assert isinstance(insight.created_at, datetime)

    def test_trend_insight_validation(self):
        """Test TrendInsight validation."""
        # Test invalid trend score
        with pytest.raises(ValueError, match="Trend score must be between 0 and 1"):
            TrendInsight(
                keyword="AI content marketing",
                trend_score=1.5,  # Invalid value
                insight_type="rising",
                description="Test description"
            )

        # Test invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            TrendInsight(
                keyword="AI content marketing",
                trend_score=0.85,
                insight_type="rising",
                description="Test description",
                confidence=2.0  # Invalid value
            )