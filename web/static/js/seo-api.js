/**
 * SEO Dashboard - API Functions
 * 
 * This file contains all API-related functionality including:
 * - Competitor analysis loading
 * - Tracked keywords management
 * - SerpBear synchronization
 * - Quick keyword operations
 */

// Use shared debug utilities (loaded from debug-utils.js)

/**
 * Load competitor analysis data
 */
async function loadCompetitorAnalysis() {
    try {
        this.competitorAnalysisLoaded = false;
        this.competitorAnalysisError = null;
        
        // Get the domain for competitor analysis
        const domain = this.selectedDomain || (this.availableDomains.length > 0 ? this.availableDomains[0] : null);
        
        if (!domain) {
            window.debugWarn('No domain available for competitor analysis');
            return;
        }
        
        const response = await fetch(`/api/serpbear/competitor-analysis?domain=${encodeURIComponent(domain)}`, {
            credentials: 'include'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            // Update competitor data arrays
            this.competitors = data.competitors || [];
            this.opportunities = data.opportunities || [];  
            this.contentGaps = data.content_gaps || [];
            
            window.debugLog(`Loaded competitor analysis for ${domain}:`, {
                competitors: this.competitors.length,
                opportunities: this.opportunities.length,
                content_gaps: this.contentGaps.length
            });
            
            this.competitorAnalysisLoaded = true;
        } else {
            throw new Error(data.message || 'Failed to load competitor analysis');
        }
        
    } catch (error) {
        window.debugError('Failed to load competitor analysis:', error);
        this.competitorAnalysisError = 'Failed to load competitor analysis';
    }
}

/**
 * Sync tracked keywords to SerpBear for ranking monitoring
 */
async function syncTrackedKeywordsToSerpBear() {
    try {
        this.syncingToSerpBear = true;
        window.debugLog('🔄 Syncing tracked keywords to SerpBear...');
        
        const response = await fetch('/api/seo-monitor/sync-to-serpbear', {
            method: 'POST',
            credentials: 'include'
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Show success message
            this.showNotification('success', `✅ Synced ${data.successful_syncs || 0} keywords to SerpBear`);
            // Reload data to show updated status
            await this.loadTrackedKeywords();
            await this.loadMetrics();
        } else {
            throw new Error(data.message || 'Sync failed');
        }
        
    } catch (error) {
        window.debugError('Failed to sync to SerpBear:', error);
        this.showNotification('error', `❌ Failed to sync keywords: ${error.message}`);
    } finally {
        this.syncingToSerpBear = false;
    }
}

/**
 * Add a new keyword to the tracking system
 */
async function addTrackedKeyword() {
    try {
        if (!this.newKeyword.keyword.trim()) {
            this.showNotification('error', 'Please enter a keyword');
            return;
        }
        
        if (!this.newKeyword.domain.trim()) {
            // Use selected domain as default
            this.newKeyword.domain = this.selectedDomain || '';
        }
        
        const response = await fetch('/api/seo-monitor/add-keyword', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            credentials: 'include',
            body: JSON.stringify(this.newKeyword)
        });
        
        const data = await response.json();
        
        if (data.success) {
            this.showNotification('success', `✅ Added "${this.newKeyword.keyword}" to tracker`);
            // Reset form and close modal
            this.newKeyword = { keyword: '', domain: '', target_url: '', notes: '' };
            this.showAddKeywordModal = false;
            // Reload tracked keywords
            await this.loadTrackedKeywords();
        } else {
            throw new Error(data.message || 'Failed to add keyword');
        }
        
    } catch (error) {
        window.debugError('Failed to add keyword:', error);
        this.showNotification('error', `❌ Failed to add keyword: ${error.message}`);
    }
}

/**
 * Remove a keyword from tracking
 * 
 * @param {string} keywordId - ID of the keyword to remove
 */
async function removeTrackedKeyword(keywordId) {
    try {
        if (!confirm('Are you sure you want to remove this keyword from tracking?')) {
            return;
        }
        
        const response = await fetch(`/api/seo-monitor/tracked-keywords/${keywordId}`, {
            method: 'DELETE',
            credentials: 'include'
        });
        
        const data = await response.json();
        
        if (data.success) {
            this.showNotification('success', '✅ Keyword removed from tracking');
            await this.loadTrackedKeywords();
        } else {
            throw new Error(data.message || 'Failed to remove keyword');
        }
        
    } catch (error) {
        window.debugError('Failed to remove keyword:', error);
        this.showNotification('error', `❌ Failed to remove keyword: ${error.message}`);
    }
}

/**
 * Remove multiple selected keywords from tracking
 */
async function removeSelectedKeywords() {
    try {
        if (this.selectedKeywords.length === 0) {
            this.showNotification('error', 'Please select keywords to remove');
            return;
        }
        
        if (!confirm(`Are you sure you want to remove ${this.selectedKeywords.length} selected keywords from tracking?`)) {
            return;
        }
        
        const response = await fetch('/api/seo-monitor/bulk-remove-keywords', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            credentials: 'include',
            body: JSON.stringify({
                keyword_ids: this.selectedKeywords
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            this.showNotification('success', `✅ Removed ${data.removed_count || this.selectedKeywords.length} keywords`);
            this.selectedKeywords = [];
            this.selectAllKeywords = false;
            await this.loadTrackedKeywords();
        } else {
            throw new Error(data.message || 'Failed to remove keywords');
        }
        
    } catch (error) {
        window.debugError('Failed to remove selected keywords:', error);
        this.showNotification('error', `❌ Failed to remove keywords: ${error.message}`);
    }
}

/**
 * Quick add a keyword from the main keywords table to tracker
 * 
 * @param {Object} keyword - Keyword object from the main table
 */
async function quickAddToTracker(keyword) {
    try {
        const keywordData = {
            keyword: keyword.keyword,
            domain: keyword.domain || this.selectedDomain || '',
            target_url: keyword.url || '',
            notes: `Added from SEO dashboard on ${new Date().toLocaleDateString()}`
        };
        
        const response = await fetch('/api/seo-monitor/add-keyword', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            credentials: 'include',
            body: JSON.stringify(keywordData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            this.showNotification('success', `✅ Added "${keyword.keyword}" to tracker`);
            // Reload tracked keywords to update UI
            await this.loadTrackedKeywords();
            // Update the keywords array to reflect tracking status
            const keywordIndex = this.keywords.findIndex(k => k.keyword === keyword.keyword);
            if (keywordIndex !== -1) {
                this.keywords[keywordIndex].is_tracked = true;
            }
        } else {
            throw new Error(data.message || 'Failed to add keyword');
        }
        
    } catch (error) {
        window.debugError('Failed to quick add keyword:', error);
        this.showNotification('error', `❌ Failed to add keyword: ${error.message}`);
    }
}

/**
 * Toggle keyword selection for bulk operations
 * 
 * @param {string} keywordId - ID of the keyword to toggle
 */
function toggleKeywordSelection(keywordId) {
    const index = this.selectedKeywords.indexOf(keywordId);
    if (index > -1) {
        this.selectedKeywords.splice(index, 1);
    } else {
        this.selectedKeywords.push(keywordId);
    }
    
    // Update select all checkbox state
    this.selectAllKeywords = this.selectedKeywords.length === this.trackedKeywords.length;
}

/**
 * Toggle selection of all keywords
 */
function toggleSelectAllKeywords() {
    if (this.selectAllKeywords) {
        this.selectedKeywords = this.trackedKeywords.map(k => k.id);
    } else {
        this.selectedKeywords = [];
    }
}

/**
 * Check if a keyword is selected for bulk operations
 * 
 * @param {string} keywordId - ID of the keyword to check
 * @returns {boolean} True if keyword is selected
 */
function isKeywordSelected(keywordId) {
    return this.selectedKeywords.includes(keywordId);
}

// Make functions globally available for use in AlpineJS components
window.loadCompetitorAnalysis = loadCompetitorAnalysis;
window.syncTrackedKeywordsToSerpBear = syncTrackedKeywordsToSerpBear;
window.addTrackedKeyword = addTrackedKeyword;
window.removeTrackedKeyword = removeTrackedKeyword;
window.removeSelectedKeywords = removeSelectedKeywords;
window.quickAddToTracker = quickAddToTracker;
window.toggleKeywordSelection = toggleKeywordSelection;
window.toggleSelectAllKeywords = toggleSelectAllKeywords;
window.isKeywordSelected = isKeywordSelected;