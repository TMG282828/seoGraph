/**
 * Shared Debug Utilities
 * 
 * Provides consistent debug logging across all SEO dashboard JavaScript files.
 * Only shows logs in localhost or when ?debug=true is in URL.
 */

// Global debug configuration - only declare once
if (typeof window.DEBUG_MODE === 'undefined') {
    window.DEBUG_MODE = window.location.hostname === 'localhost' || window.location.search.includes('debug=true');
}

// Global debug logging functions
if (typeof window.debugLog === 'undefined') {
    window.debugLog = window.DEBUG_MODE ? console.log : () => {};
    window.debugWarn = window.DEBUG_MODE ? console.warn : () => {};
    window.debugError = console.error; // Always show errors
}