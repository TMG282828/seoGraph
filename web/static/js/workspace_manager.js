/**
 * Workspace Manager - Alpine.js Store for Workspace State Management
 * 
 * Provides centralized workspace state management across the application
 * with support for workspace switching, creation, joining, and member management.
 */

// Alpine.js Workspace Store
document.addEventListener('alpine:init', () => {
    Alpine.store('workspace', {
        // State
        current: null,
        available: [],
        isLoading: false,
        error: null,
        
        // Modal states
        showModal: false,
        showSettingsModal: false,
        
        // Initialize workspace store
        async init() {
            console.log('🏢 Initializing workspace store');
            await this.loadWorkspaces();
            this.setupEventListeners();
        },
        
        // Load all available workspaces
        async loadWorkspaces() {
            try {
                this.isLoading = true;
                this.error = null;
                
                console.log('📡 Loading workspaces from API');
                
                const response = await fetch('/api/workspaces', {
                    headers: {
                        'Authorization': `Bearer ${this.getAuthToken()}`,
                        'Content-Type': 'application/json'
                    }
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                this.available = await response.json();
                
                // Set current workspace
                this.current = this.available.find(w => w.is_current) || this.available[0] || null;
                
                console.log(`✅ Loaded ${this.available.length} workspaces`, this.available);
                
                // Update page title with workspace name
                if (this.current) {
                    this.updatePageTitle(this.current.name);
                }
                
            } catch (error) {
                console.error('❌ Failed to load workspaces:', error);
                this.error = error.message;
                
                // Fallback to default workspace
                this.available = [{
                    id: 'demo-org',
                    name: 'My Workspace',
                    user_role: 'owner',
                    is_current: true,
                    member_count: 1,
                    created_at: new Date().toISOString()
                }];
                this.current = this.available[0];
                
            } finally {
                this.isLoading = false;
            }
        },
        
        // Switch to a different workspace
        async switchWorkspace(workspaceId) {
            if (!workspaceId || this.current?.id === workspaceId) return;
            
            const targetWorkspace = this.available.find(w => w.id === workspaceId);
            if (!targetWorkspace) {
                console.error('❌ Workspace not found:', workspaceId);
                return;
            }
            
            try {
                this.isLoading = true;
                console.log(`🔄 Switching to workspace: ${targetWorkspace.name}`);
                
                const response = await fetch(`/api/workspaces/${workspaceId}/switch`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${this.getAuthToken()}`,
                        'Content-Type': 'application/json'
                    }
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to switch workspace');
                }
                
                const result = await response.json();
                
                // Update auth token
                if (result.access_token) {
                    this.updateAuthToken(result.access_token);
                }
                
                // Update workspace state
                this.available.forEach(w => w.is_current = (w.id === workspaceId));
                this.current = targetWorkspace;
                this.current.is_current = true;
                
                // Update page title
                this.updatePageTitle(this.current.name);
                
                console.log(`✅ Switched to workspace: ${targetWorkspace.name}`);
                
                // Dispatch workspace changed event
                this.dispatchEvent('workspace-switched', { workspace: this.current });
                
                // Refresh page to update all workspace-scoped data
                setTimeout(() => {
                    window.location.reload();
                }, 500);
                
                return true;
                
            } catch (error) {
                console.error('❌ Failed to switch workspace:', error);
                this.error = error.message;
                this.showError('Failed to switch workspace: ' + error.message);
                return false;
            } finally {
                this.isLoading = false;
            }
        },
        
        // Create new workspace
        async createWorkspace(workspaceData) {
            try {
                this.isLoading = true;
                console.log('🏗️ Creating workspace:', workspaceData.name);
                
                const response = await fetch('/api/workspaces', {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${this.getAuthToken()}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(workspaceData)
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to create workspace');
                }
                
                const newWorkspace = await response.json();
                
                // Add to available workspaces
                this.available.push(newWorkspace);
                
                console.log('✅ Workspace created:', newWorkspace);
                
                // Dispatch event
                this.dispatchEvent('workspace-created', { workspace: newWorkspace });
                
                return newWorkspace;
                
            } catch (error) {
                console.error('❌ Failed to create workspace:', error);
                this.error = error.message;
                throw error;
            } finally {
                this.isLoading = false;
            }
        },
        
        // Join workspace via invite code
        async joinWorkspace(inviteCode) {
            try {
                this.isLoading = true;
                console.log('🤝 Joining workspace with code:', inviteCode);
                
                const response = await fetch('/api/workspaces/join', {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${this.getAuthToken()}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ invite_code: inviteCode })
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to join workspace');
                }
                
                const joinedWorkspace = await response.json();
                
                // Add to available workspaces
                this.available.push(joinedWorkspace);
                
                console.log('✅ Joined workspace:', joinedWorkspace);
                
                // Dispatch event
                this.dispatchEvent('workspace-joined', { workspace: joinedWorkspace });
                
                return joinedWorkspace;
                
            } catch (error) {
                console.error('❌ Failed to join workspace:', error);
                this.error = error.message;
                throw error;
            } finally {
                this.isLoading = false;
            }
        },
        
        // Get workspace by ID
        getWorkspace(workspaceId) {
            return this.available.find(w => w.id === workspaceId);
        },
        
        // Check if user can manage workspace
        canManageWorkspace(workspaceId) {
            const workspace = this.getWorkspace(workspaceId);
            if (!workspace) return false;
            
            const role = workspace.user_role;
            return role === 'owner' || role === 'admin';
        },
        
        // Setup event listeners
        setupEventListeners() {
            // Listen for workspace events
            window.addEventListener('workspace-created', (event) => {
                console.log('🎉 Workspace created event:', event.detail);
            });
            
            window.addEventListener('workspace-joined', (event) => {
                console.log('🎉 Workspace joined event:', event.detail);
            });
            
            window.addEventListener('workspace-switched', (event) => {
                console.log('🔄 Workspace switched event:', event.detail);
            });
            
            // Listen for auth token updates
            window.addEventListener('auth-token-updated', (event) => {
                console.log('🔑 Auth token updated');
                // Reload workspaces with new token
                this.loadWorkspaces();
            });
        },
        
        // Utility methods
        updatePageTitle(workspaceName) {
            if (workspaceName && workspaceName !== 'My Workspace') {
                document.title = `${workspaceName} - SEO Content AI`;
            } else {
                document.title = 'SEO Content AI';
            }
        },
        
        // Auth token management
        getAuthToken() {
            // Try cookie first
            const tokenFromCookie = this.getCookie('access_token');
            if (tokenFromCookie) return tokenFromCookie;
            
            // Fallback to localStorage
            return localStorage.getItem('access_token') || '';
        },
        
        updateAuthToken(token) {
            // Update cookie
            document.cookie = `access_token=${token}; path=/; secure; samesite=strict`;
            
            // Update localStorage as backup
            localStorage.setItem('access_token', token);
            
            // Dispatch event
            this.dispatchEvent('auth-token-updated', { token });
        },
        
        getCookie(name) {
            const value = `; ${document.cookie}`;
            const parts = value.split(`; ${name}=`);
            if (parts.length === 2) return parts.pop().split(';').shift();
            return null;
        },
        
        // Event dispatcher
        dispatchEvent(eventName, detail = {}) {
            window.dispatchEvent(new CustomEvent(eventName, { 
                detail,
                bubbles: true,
                cancelable: true 
            }));
        },
        
        // Error handling
        showError(message) {
            console.error('Workspace Error:', message);
            
            // Could integrate with toast notification system
            if (window.showToast) {
                window.showToast(message, 'error');
            } else {
                // Fallback to alert
                alert(`Error: ${message}`);
            }
        },
        
        showSuccess(message) {
            console.log('Workspace Success:', message);
            
            // Could integrate with toast notification system
            if (window.showToast) {
                window.showToast(message, 'success');
            } else {
                // Simple console log for now
                console.log(`✅ ${message}`);
            }
        },
        
        // Clear error
        clearError() {
            this.error = null;
        },
        
        // Modal management
        openModal() {
            this.showModal = true;
        },
        
        closeModal() {
            this.showModal = false;
        },
        
        openSettingsModal() {
            this.showSettingsModal = true;
        },
        
        closeSettingsModal() {
            this.showSettingsModal = false;
        },
        
        // Refresh workspaces
        async refresh() {
            await this.loadWorkspaces();
        }
    });
});

// Workspace Utilities - Global helper functions
window.WorkspaceUtils = {
    
    // Get workspace initial letters
    getWorkspaceInitial(name) {
        if (!name) return 'W';
        return name.split(' ')
            .map(word => word[0])
            .join('')
            .toUpperCase()
            .slice(0, 2);
    },
    
    // Generate consistent workspace color
    getWorkspaceColor(name) {
        if (!name) return '#3B82F6';
        
        const colors = [
            '#3B82F6', // Blue
            '#8B5CF6', // Purple  
            '#06B6D4', // Cyan
            '#10B981', // Emerald
            '#F59E0B', // Amber
            '#EF4444', // Red
            '#EC4899', // Pink
            '#6366F1'  // Indigo
        ];
        
        // Generate hash from name
        let hash = 0;
        for (let i = 0; i < name.length; i++) {
            hash = name.charCodeAt(i) + ((hash << 5) - hash);
        }
        
        return colors[Math.abs(hash) % colors.length];
    },
    
    // Format role display
    getRoleDisplay(role) {
        const roleMap = {
            'owner': 'Owner',
            'admin': 'Admin',
            'manager': 'Manager', 
            'member': 'Member',
            'viewer': 'Viewer',
            'guest': 'Guest'
        };
        return roleMap[role] || 'Member';
    },
    
    // Format date for display
    formatDate(dateString) {
        if (!dateString) return '';
        
        const date = new Date(dateString);
        const now = new Date();
        const diffMs = now - date;
        const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
        
        if (diffDays === 0) {
            return 'Today';
        } else if (diffDays === 1) {
            return 'Yesterday';
        } else if (diffDays < 7) {
            return `${diffDays} days ago`;
        } else {
            return date.toLocaleDateString();
        }
    },
    
    // Validate workspace name
    validateWorkspaceName(name) {
        if (!name || name.trim().length === 0) {
            return 'Workspace name is required';
        }
        
        if (name.trim().length > 200) {
            return 'Workspace name must be less than 200 characters';
        }
        
        // Check for valid characters
        if (!/^[a-zA-Z0-9\s\-_'".]+$/.test(name.trim())) {
            return 'Workspace name contains invalid characters';
        }
        
        return null; // Valid
    },
    
    // Validate invite code format
    validateInviteCode(code) {
        if (!code || code.trim().length === 0) {
            return 'Invite code is required';
        }
        
        if (code.trim().length < 6 || code.trim().length > 20) {
            return 'Invalid invite code format';
        }
        
        return null; // Valid
    }
};

// Initialize workspace store when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait for Alpine.js to be ready
    setTimeout(() => {
        if (window.Alpine && Alpine.store('workspace')) {
            Alpine.store('workspace').init();
        }
    }, 100);
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { WorkspaceUtils };
}