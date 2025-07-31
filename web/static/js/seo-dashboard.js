/**
 * SEO Dashboard - Main AlpineJS Component
 * 
 * This file contains the primary dashboard functionality including:
 * - Data management and state
 * - Metrics loading with fallback chain
 * - Keyword management
 * - Domain switching
 * - Helper functions
 */

// Use shared debug utilities (loaded from debug-utils.js)

function seoDashboardData() {
    return {
        // State variables
        keywordSearch: '',
        gscConnected: false,
        connectingGSC: false,
        gscAuthUrl: '',
        gscStatusChecked: false,
        selectedDomain: '',
        availableDomains: [],
        
        // Dashboard metrics
        metrics: {
            organic_traffic: 0,
            organic_traffic_change: 0,
            top_10_keywords: 0,
            top_10_keywords_change: 0,
            avg_position: 0,
            avg_position_change: 0,
            total_keywords: 0,
            total_keywords_change: 0,
            ctr: 0,
            ctr_change: 0
        },
        
        // Data arrays
        keywords: [],
        trackedKeywords: [],
        competitors: [],
        opportunities: [],
        contentGaps: [],
        
        // UI state
        competitorAnalysisLoaded: false,
        competitorAnalysisError: null,
        
        // Tracked Keywords Management
        showAddKeywordModal: false,
        syncingToSerpBear: false,
        selectedKeywords: [],
        selectAllKeywords: false,
        newKeyword: {
            keyword: '',
            domain: '',
            target_url: '',
            notes: ''
        },
        
        // Computed properties
        get filteredKeywords() {
            if (!this.keywordSearch) return this.keywords;
            return this.keywords.filter(k => 
                k.keyword.toLowerCase().includes(this.keywordSearch.toLowerCase())
            );
        },
        
        // Utility functions
        formatNumber(num) {
            if (num >= 1000000) {
                return (num / 1000000).toFixed(1) + 'M';
            } else if (num >= 1000) {
                return (num / 1000).toFixed(1) + 'K';
            }
            return num.toString();
        },
        
        formatDate(dateString) {
            if (!dateString) return 'Unknown';
            const date = new Date(dateString);
            return date.toLocaleDateString();
        },
        
        // Keyword helper functions
        getKeywordPosition(keyword, domain) {
            const foundKeyword = this.keywords.find(k => 
                k.keyword.toLowerCase() === keyword.toLowerCase() && 
                k.domain === domain
            );
            return foundKeyword ? foundKeyword.position || 0 : 0;
        },
        
        isKeywordTracked(keyword, domain) {
            // First check if the keyword object already has tracking status from API
            if (typeof keyword === 'object' && keyword.hasOwnProperty('is_tracked')) {
                return keyword.is_tracked;
            }
            
            // Fallback: check against tracked keywords array for string/domain pairs
            return this.trackedKeywords.some(k => 
                k.keyword.toLowerCase() === keyword.toLowerCase() && 
                k.domain === domain &&
                k.is_active
            );
        },
        
        // Keyword analysis modal state
        showAnalysisModal: false,
        analysisKeyword: null,
        
        // Keyword analysis functionality
        analyzeKeyword(keywordId) {
            const keyword = this.keywords.find(k => k.id === keywordId);
            if (keyword) {
                this.analysisKeyword = keyword;
                this.showAnalysisModal = true;
                window.debugLog('Opening analysis modal for keyword:', keyword.keyword);
            }
        },
        
        closeAnalysisModal() {
            this.showAnalysisModal = false;
            this.analysisKeyword = null;
        },
        
        // API functions (defined in seo-api.js)
        loadCompetitorAnalysis() { 
            if (typeof loadCompetitorAnalysis === 'function') return loadCompetitorAnalysis.call(this); 
            window.debugError('loadCompetitorAnalysis function not found');
        },
        syncTrackedKeywordsToSerpBear() { 
            if (typeof syncTrackedKeywordsToSerpBear === 'function') return syncTrackedKeywordsToSerpBear.call(this); 
            window.debugError('syncTrackedKeywordsToSerpBear function not found');
        },
        addTrackedKeyword() { 
            if (typeof addTrackedKeyword === 'function') return addTrackedKeyword.call(this); 
            window.debugError('addTrackedKeyword function not found');
        },
        removeTrackedKeyword(keywordId) { 
            if (typeof removeTrackedKeyword === 'function') return removeTrackedKeyword.call(this, keywordId); 
            window.debugError('removeTrackedKeyword function not found');
        },
        removeSelectedKeywords() { 
            if (typeof removeSelectedKeywords === 'function') return removeSelectedKeywords.call(this); 
            window.debugError('removeSelectedKeywords function not found');
        },
        quickAddToTracker(keyword) { 
            if (typeof quickAddToTracker === 'function') return quickAddToTracker.call(this, keyword); 
            window.debugError('quickAddToTracker function not found');
        },
        toggleKeywordSelection(keywordId) { 
            if (typeof toggleKeywordSelection === 'function') return toggleKeywordSelection.call(this, keywordId); 
            window.debugError('toggleKeywordSelection function not found');
        },
        toggleSelectAllKeywords() { 
            if (typeof toggleSelectAllKeywords === 'function') return toggleSelectAllKeywords.call(this); 
            window.debugError('toggleSelectAllKeywords function not found');
        },
        isKeywordSelected(keywordId) { 
            if (typeof isKeywordSelected === 'function') return isKeywordSelected.call(this, keywordId); 
            window.debugError('isKeywordSelected function not found');
            return false;
        },
        
        // Chart functions (defined in seo-charts.js)
        initCharts() { 
            if (typeof initSEOCharts === 'function') return initSEOCharts.call(this); 
            window.debugError('initSEOCharts function not found');
        },
        initTrafficChart() { 
            if (typeof initTrafficChart === 'function') return initTrafficChart.call(this); 
            window.debugError('initTrafficChart function not found');
        },
        initRankingsChart() { 
            if (typeof initRankingsChart === 'function') return initRankingsChart.call(this); 
            window.debugError('initRankingsChart function not found');
        },
        loadRankingTrends() { 
            if (typeof loadRankingTrends === 'function') return loadRankingTrends.call(this); 
            window.debugError('loadRankingTrends function not found');
        },
        updateChartsWithRealData(trendData) { 
            if (typeof updateChartsWithRealData === 'function') return updateChartsWithRealData.call(this, trendData); 
            window.debugError('updateChartsWithRealData function not found');
        },
        showChartError() { 
            if (typeof showChartError === 'function') return showChartError.call(this); 
            window.debugError('showChartError function not found');
        },
        destroyCharts() { 
            if (typeof destroyCharts === 'function') return destroyCharts.call(this); 
            window.debugError('destroyCharts function not found');
        },
        
        // Chart-related properties (managed by seo-charts.js)
        trafficChart: null,
        rankingsChart: null,
        chartsInitialized: false,
        
        // Data loading functions
        async loadMetrics() {
            try {
                window.debugLog('🔄 Loading unified SEO metrics...');
                
                // Primary: Try unified SerpBear + Google Ads + GSC data
                try {
                    const domainParam = this.selectedDomain ? `?domain=${encodeURIComponent(this.selectedDomain)}` : '';
                    const serpbearResponse = await fetch(`/api/serpbear/dashboard-metrics${domainParam}`, {
                        credentials: 'include'
                    });
                    const serpbearData = await serpbearResponse.json();
                    
                    if (serpbearData.success && serpbearData.data_source === 'unified') {
                        window.debugLog('✅ Using unified data (SerpBear + Google Ads + GSC)');
                        
                        // Update metrics with unified data
                        this.metrics = {
                            ...this.metrics,
                            ...serpbearData.metrics
                        };
                        
                        // Load unified keywords
                        const keywordsDomainParam = this.selectedDomain ? `&domain=${encodeURIComponent(this.selectedDomain)}` : '';
                        const keywordsResponse = await fetch(`/api/serpbear/keywords?limit=50${keywordsDomainParam}`, {
                            credentials: 'include'
                        });
                        const keywordsData = await keywordsResponse.json();
                        
                        if (keywordsData.success) {
                            this.keywords = keywordsData.keywords;
                            window.debugLog(`📊 Loaded ${this.keywords.length} keywords with unified data`);
                        }
                        
                        // Load trend data and competitor analysis
                        await this.loadRankingTrends();
                        await this.loadCompetitorAnalysis();
                        
                        // GSC connection status is determined by checkGSCConnection() 
                        // which is called during initialization
                        return;
                    }
                } catch (primaryError) {
                    window.debugWarn('Primary unified data failed:', primaryError);
                }
                
                // Fallback 1: Try SerpBear rankings + Google Ads (no GSC)
                try {
                    window.debugLog('📊 Trying SerpBear + Google Ads fallback...');
                    
                    // Load tracked keywords for Google Ads data
                    await this.loadTrackedKeywords();
                    
                    // Get SerpBear ranking data directly
                    const domainParam = this.selectedDomain ? `?domain=${encodeURIComponent(this.selectedDomain)}` : '';
                    const serpbearResponse = await fetch(`/api/serpbear/dashboard-metrics${domainParam}`, {
                        credentials: 'include'
                    });
                    const serpbearData = await serpbearResponse.json();
                    
                    if (serpbearData.success) {
                        window.debugLog('✅ Using SerpBear rankings + Google Ads data');
                        
                        // Update metrics with available data
                        this.metrics = {
                            ...this.metrics,
                            ...serpbearData.metrics
                        };
                        
                        // Load keywords with ranking data
                        const keywordsDomainParam = this.selectedDomain ? `&domain=${encodeURIComponent(this.selectedDomain)}` : '';
                        const keywordsResponse = await fetch(`/api/serpbear/keywords?limit=50${keywordsDomainParam}`, {
                            credentials: 'include'
                        });
                        const keywordsData = await keywordsResponse.json();
                        
                        if (keywordsData.success) {
                            this.keywords = keywordsData.keywords;
                            window.debugLog(`📊 Loaded ${this.keywords.length} keywords with SerpBear + Google Ads`);
                        }
                        
                        this.gscConnected = false; // No GSC data
                        
                        // Load competitor analysis for fallback 1
                        await this.loadCompetitorAnalysis();
                        return;
                    }
                } catch (fallback1Error) {
                    window.debugWarn('SerpBear + Google Ads fallback failed:', fallback1Error);
                }
                
                // Fallback 2: Pure SerpBear rankings only
                try {
                    window.debugLog('📊 Trying pure SerpBear rankings fallback...');
                    
                    // Direct SerpBear database access via our unified service
                    const domainParam = this.selectedDomain ? `domain=${encodeURIComponent(this.selectedDomain)}&` : '';
                    const serpbearResponse = await fetch(`/api/serpbear/dashboard-metrics?${domainParam}pure_rankings=true`, {
                        credentials: 'include'
                    });
                    const serpbearData = await serpbearResponse.json();
                    
                    if (serpbearData.success) {
                        window.debugLog('✅ Using pure SerpBear ranking data');
                        
                        // Update metrics with ranking data only
                        this.metrics = {
                            organic_traffic: serpbearData.metrics.organic_traffic || 0,
                            organic_traffic_change: 0,
                            top_10_keywords: serpbearData.metrics.top_10_keywords || 0,
                            top_10_keywords_change: 0,
                            avg_position: serpbearData.metrics.avg_position || 0,
                            avg_position_change: 0,
                            total_keywords: serpbearData.metrics.total_keywords || 0,
                            total_keywords_change: 0,
                            ctr: serpbearData.metrics.ctr || 0,
                            ctr_change: 0
                        };
                        
                        // Load keywords with position data only
                        const keywordsDomainParam = this.selectedDomain ? `&domain=${encodeURIComponent(this.selectedDomain)}` : '';
                        const keywordsResponse = await fetch(`/api/serpbear/keywords?limit=50${keywordsDomainParam}`, {
                            credentials: 'include'
                        });
                        const keywordsData = await keywordsResponse.json();
                        
                        if (keywordsData.success) {
                            this.keywords = keywordsData.keywords.map(kw => ({
                                ...kw,
                                search_volume: 0, // No volume data in this fallback
                                traffic: 0,       // No traffic data
                                difficulty: 0     // No difficulty data
                            }));
                            window.debugLog(`📊 Loaded ${this.keywords.length} keywords with positions only`);
                        }
                        
                        this.gscConnected = false;
                        
                        // Load competitor analysis for fallback 2
                        await this.loadCompetitorAnalysis();
                        return;
                    }
                } catch (fallback2Error) {
                    window.debugWarn('Pure SerpBear fallback failed:', fallback2Error);
                }
                
                // Final fallback: Show connection prompt
                window.debugLog('⚠️ All data sources failed - showing connection prompt');
                this.checkGSCConnection();
                
            } catch (error) {
                window.debugError('Failed to load SEO metrics:', error);
                this.checkGSCConnection();
            }
        },
        
        async loadTrackedKeywords() {
            try {
                const domainParam = this.selectedDomain ? `?domain=${encodeURIComponent(this.selectedDomain)}` : '';
                const response = await fetch(`/api/seo-monitor/tracked-keywords${domainParam}`, {
                    credentials: 'include'
                });
                const data = await response.json();
                
                if (data.success) {
                    this.trackedKeywords = data.keywords || [];
                    window.debugLog(`Loaded ${this.trackedKeywords.length} tracked keywords${this.selectedDomain ? ' for ' + this.selectedDomain : ''}`);
                } else {
                    window.debugLog('No tracked keywords found');
                }
                
            } catch (error) {
                window.debugError('Failed to load tracked keywords:', error);
            }
        },
        
        async loadAvailableDomains() {
            try {
                window.debugLog('🌐 Loading available domains from SerpBear...');
                
                const response = await fetch('/api/serpbear/connection-status', {
                    credentials: 'include'
                });
                const data = await response.json();
                
                if (data.success && data.domains) {
                    this.availableDomains = data.domains;
                    window.debugLog(`📋 Found ${this.availableDomains.length} available domains:`, this.availableDomains);
                    
                    // Set primary domain from environment or first available
                    if (this.availableDomains.length > 0 && !this.selectedDomain) {
                        this.selectedDomain = this.availableDomains[0];
                        window.debugLog(`🎯 Set primary domain to: ${this.selectedDomain}`);
                    }
                    
                    return this.availableDomains.length > 0;
                } else {
                    window.debugLog('No domains found in SerpBear');
                    return false;
                }
            } catch (error) {
                window.debugError('Failed to load available domains:', error);
                return false;
            }
        },
        
        
        async switchDomain(newDomain) {
            if (newDomain === this.selectedDomain) return;
            
            window.debugLog(`🔄 Switching from ${this.selectedDomain} to ${newDomain}`);
            console.log('🔄 Domain switching debug:', {
                oldDomain: this.selectedDomain,
                newDomain: newDomain,
                availableDomains: this.availableDomains,
                currentKeywordCount: this.keywords.length
            });
            
            this.selectedDomain = newDomain;
            
            // Reload all dashboard data for the new domain
            try {
                console.log('📊 Loading metrics for domain:', newDomain);
                await this.loadMetrics();
                
                console.log('🔑 Loading tracked keywords for domain:', newDomain);
                await this.loadTrackedKeywords();
                
                console.log('🥊 Loading competitor analysis for domain:', newDomain);
                await this.loadCompetitorAnalysis();
                
                console.log('✅ After domain switch - keyword count:', this.keywords.length);
                window.debugLog(`✅ Dashboard data refreshed for domain: ${newDomain}`);
            } catch (error) {
                console.error('❌ Domain switch error:', error);
                window.debugError(`Failed to reload data for domain ${newDomain}:`, error);
            }
        },
        
        // GSC Connection functions
        async checkGSCConnection() {
            try {
                // First, try to check if GSC is already connected by attempting to get sites
                const statusResponse = await fetch('/api/gsc/sites', {
                    credentials: 'include'
                });
                
                if (statusResponse.ok) {
                    const statusData = await statusResponse.json();
                    if (statusData.success && statusData.sites && statusData.sites.length > 0) {
                        // GSC is connected and has verified sites
                        this.gscConnected = true;
                        return;
                    }
                }
                
                // If not connected, get auth URL for connection
                const authResponse = await fetch('/api/gsc/auth/url', {
                    credentials: 'include'
                });
                
                if (authResponse.ok) {
                    const authData = await authResponse.json();
                    this.gscAuthUrl = authData.auth_url;
                    this.gscConnected = false;
                }
            } catch (error) {
                window.debugError('Failed to check GSC connection:', error);
                this.gscConnected = false;
            } finally {
                // Mark GSC status as checked to prevent panel flash
                this.gscStatusChecked = true;
            }
        },
        
        async connectGSC() {
            if (this.gscAuthUrl) {
                this.connectingGSC = true;
                // Open Google Search Console authorization in new window
                const authWindow = window.open(this.gscAuthUrl, 'gsc-auth', 'width=600,height=600');
                
                // Listen for authorization completion
                const checkClosed = setInterval(() => {
                    if (authWindow.closed) {
                        clearInterval(checkClosed);
                        this.connectingGSC = false;
                        // Reload metrics after authorization
                        setTimeout(() => this.loadMetrics(), 2000);
                    }
                }, 1000);
            } else {
                this.checkGSCConnection();
            }
        },
        
        // Notification system
        showNotification(type, message) {
            const notification = document.createElement('div');
            notification.className = `fixed top-4 right-4 px-4 py-2 rounded-lg shadow-lg z-50 ${
                type === 'success' ? 'bg-green-600 text-white' : 'bg-red-600 text-white'
            }`;
            notification.textContent = message;
            document.body.appendChild(notification);
            setTimeout(() => notification.remove(), 5000);
        },
        
        // Initialization
        initialized: false,
        
        init() {
            if (this.initialized) {
                window.debugLog('SEO dashboard component already initialized, skipping');
                return;
            }
            
            this.initialized = true;
            window.debugLog('SEO dashboard component initialized');
            
            // Load available domains first, then load metrics
            this.loadAvailableDomains().then((domainsLoaded) => {
                if (domainsLoaded) {
                    // Check GSC connection and load metrics
                    this.checkGSCConnection().then(() => {
                        this.loadMetrics();
                    });
                } else {
                    // Still check GSC connection even if no domains found
                    this.checkGSCConnection();
                }
            });
            
            // Initialize charts after a delay to ensure DOM is ready
            setTimeout(() => {
                this.initCharts();
            }, 500);
        }
    };
}

// Make function globally available
window.seoDashboardData = seoDashboardData;