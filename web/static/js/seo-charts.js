/**
 * SEO Dashboard - Chart.js Functionality
 * 
 * This file contains all Chart.js related functionality including:
 * - Chart initialization and configuration
 * - Real-time data updates from SerpBear
 * - Chart destruction and recreation
 * - Error handling and fallback displays
 */

// Use shared debug utilities (loaded from debug-utils.js)

/**
 * Initialize SEO dashboard charts
 * This function should be called from the main AlpineJS component
 */
function initSEOCharts() {
    if (this.chartsInitialized) {
        window.debugLog('Charts already initialized, skipping');
        return;
    }
    
    // Add a small delay to ensure DOM is ready
    setTimeout(() => {
        try {
            // Destroy existing charts if they exist
            if (this.trafficChart) {
                this.trafficChart.destroy();
            }
            if (this.rankingsChart) {
                this.rankingsChart.destroy();
            }
            
            // Initialize Traffic Chart
            this.initTrafficChart();
            
            // Initialize Rankings Chart
            this.initRankingsChart();
            
            window.debugLog('SEO dashboard charts initialized successfully');
            this.chartsInitialized = true;
            
        } catch (error) {
            window.debugError('Error initializing SEO charts:', error);
            this.chartsInitialized = false;
            this.showChartError();
        }
    }, 100);
}

/**
 * Initialize the organic traffic trend chart
 */
function initTrafficChart() {
    const trafficCanvas = document.getElementById('trafficChart');
    if (!trafficCanvas) {
        window.debugError('Traffic chart canvas not found');
        return;
    }
    
    const trafficCtx = trafficCanvas.getContext('2d');
    this.trafficChart = new Chart(trafficCtx, {
        type: 'line',
        data: {
            labels: ['No Data'],
            datasets: [{
                label: 'Organic Traffic',
                data: [0],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#ffffff'
                    }
                }
            },
            scales: {
                y: {
                    grid: {
                        color: '#404040'
                    },
                    ticks: {
                        color: '#b3b3b3',
                        callback: function(value) {
                            return value >= 1000 ? (value / 1000) + 'K' : value;
                        }
                    }
                },
                x: {
                    grid: {
                        color: '#404040'
                    },
                    ticks: {
                        color: '#b3b3b3'
                    }
                }
            }
        }
    });
}

/**
 * Initialize the keyword rankings distribution chart
 */
function initRankingsChart() {
    const rankingsCanvas = document.getElementById('rankingsChart');
    if (!rankingsCanvas) {
        window.debugError('Rankings chart canvas not found');
        return;
    }
    
    const rankingsCtx = rankingsCanvas.getContext('2d');
    this.rankingsChart = new Chart(rankingsCtx, {
        type: 'bar',
        data: {
            labels: ['1-3', '4-10', '11-20', '21-50', '51-100'],
            datasets: [{
                label: 'Keywords',
                data: [0, 0, 0, 0, 0],
                backgroundColor: [
                    '#10b981', // Green for top 3
                    '#3b82f6', // Blue for 4-10
                    '#f59e0b', // Yellow for 11-20
                    '#ef4444', // Red for 21-50
                    '#6b7280'  // Gray for 51-100
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    grid: {
                        color: '#404040'
                    },
                    ticks: {
                        color: '#b3b3b3'
                    }
                },
                x: {
                    grid: {
                        color: '#404040'
                    },
                    ticks: {
                        color: '#b3b3b3'
                    }
                }
            }
        }
    });
}

/**
 * Load ranking trends data from SerpBear API
 */
async function loadRankingTrends() {
    try {
        // Get the primary domain from available domains or first tracked keyword
        let domain = this.selectedDomain;
        
        // Fallback: use domain from first tracked keyword if available
        if (!domain && this.trackedKeywords.length > 0) {
            domain = this.trackedKeywords[0].domain;
        }
        
        // Fallback: use first available domain from SerpBear
        if (!domain && this.availableDomains.length > 0) {
            domain = this.availableDomains[0];
        }
        
        if (!domain) {
            window.debugWarn('No domain available for ranking trends');
            return;
        }
        
        const response = await fetch(`/api/serpbear/ranking-trends?domain=${domain}&days=30`, {
            credentials: 'include'
        });
        const data = await response.json();
        
        if (data.success) {
            // Update chart data with real trend data
            this.updateChartsWithRealData(data);
            window.debugLog('Loaded ranking trends from SerpBear');
        }
    } catch (error) {
        window.debugError('Failed to load ranking trends:', error);
    }
}

/**
 * Update charts with real data from SerpBear
 * 
 * @param {Object} trendData - Data from SerpBear ranking trends API
 */
function updateChartsWithRealData(trendData) {
    try {
        // Update traffic chart with real data
        if (this.trafficChart && trendData.traffic_chart) {
            this.trafficChart.data.labels = trendData.traffic_chart.labels;
            this.trafficChart.data.datasets[0].data = trendData.traffic_chart.data;
            this.trafficChart.update();
        }
        
        // Update rankings chart with real data
        if (this.rankingsChart && trendData.rankings_chart) {
            this.rankingsChart.data.labels = trendData.rankings_chart.labels;
            this.rankingsChart.data.datasets[0].data = trendData.rankings_chart.data;
            this.rankingsChart.update();
        }
        
        window.debugLog('Charts updated with real SerpBear data');
    } catch (error) {
        window.debugError('Failed to update charts:', error);
    }
}

/**
 * Show error message when charts fail to load
 */
function showChartError() {
    // Show error message in chart containers
    const trafficChart = document.getElementById('trafficChart');
    const rankingsChart = document.getElementById('rankingsChart');
    
    if (trafficChart) {
        trafficChart.parentElement.innerHTML = '<div class="flex items-center justify-center h-48 text-gray-400"><i class="fas fa-exclamation-triangle mr-2"></i>Chart failed to load</div>';
    }
    
    if (rankingsChart) {
        rankingsChart.parentElement.innerHTML = '<div class="flex items-center justify-center h-48 text-gray-400"><i class="fas fa-exclamation-triangle mr-2"></i>Chart failed to load</div>';
    }
}

/**
 * Destroy all charts (useful for cleanup)
 */
function destroyCharts() {
    if (this.trafficChart) {
        this.trafficChart.destroy();
        this.trafficChart = null;
    }
    if (this.rankingsChart) {
        this.rankingsChart.destroy();
        this.rankingsChart = null;
    }
    this.chartsInitialized = false;
}

// Make functions globally available for use in AlpineJS components
window.initSEOCharts = initSEOCharts;
window.initTrafficChart = initTrafficChart;
window.initRankingsChart = initRankingsChart;
window.loadRankingTrends = loadRankingTrends;
window.updateChartsWithRealData = updateChartsWithRealData;
window.showChartError = showChartError;
window.destroyCharts = destroyCharts;