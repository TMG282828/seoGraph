/**
 * Interactive Graph Visualization for the SEO Content Knowledge Graph System.
 * 
 * This module provides D3.js and Cytoscape.js powered graph visualizations
 * for content relationships, keyword networks, and knowledge graphs.
 */

class GraphVisualization {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.options = {
            renderer: options.renderer || 'cytoscape', // 'cytoscape' or 'd3'
            width: options.width || 800,
            height: options.height || 600,
            layout: options.layout || 'cola',
            interactive: options.interactive !== false,
            showLabels: options.showLabels !== false,
            showLegend: options.showLegend !== false,
            theme: options.theme || 'light',
            ...options
        };
        
        this.data = { nodes: [], edges: [] };
        this.cy = null;
        this.d3Svg = null;
        this.selectedNodes = new Set();
        this.filteredNodeTypes = new Set();
        this.searchQuery = '';
        
        this.initialize();
    }
    
    initialize() {
        // Create container structure
        this.createContainer();
        
        // Initialize renderer
        if (this.options.renderer === 'cytoscape') {
            this.initializeCytoscape();
        } else {
            this.initializeD3();
        }
        
        // Add event listeners
        this.setupEventListeners();
        
        // Create controls
        if (this.options.interactive) {
            this.createControls();
        }
        
        // Create legend
        if (this.options.showLegend) {
            this.createLegend();
        }
    }
    
    createContainer() {
        this.container.innerHTML = `
            <div class="graph-container">
                <div class="graph-header">
                    <div class="graph-title">
                        <h3>Knowledge Graph</h3>
                    </div>
                    <div class="graph-controls" id="${this.containerId}-controls">
                        <!-- Controls will be added here -->
                    </div>
                </div>
                <div class="graph-content">
                    <div class="graph-sidebar" id="${this.containerId}-sidebar">
                        <!-- Sidebar content -->
                    </div>
                    <div class="graph-main" id="${this.containerId}-main">
                        <!-- Main graph area -->
                    </div>
                </div>
                <div class="graph-footer">
                    <div class="graph-stats" id="${this.containerId}-stats">
                        <!-- Statistics -->
                    </div>
                    <div class="graph-legend" id="${this.containerId}-legend">
                        <!-- Legend -->
                    </div>
                </div>
            </div>
        `;
        
        // Add CSS classes
        this.container.classList.add('graph-visualization', `theme-${this.options.theme}`);
    }
    
    initializeCytoscape() {
        const mainContainer = document.getElementById(`${this.containerId}-main`);
        
        // Cytoscape configuration
        const cyConfig = {
            container: mainContainer,
            
            style: this.getCytoscapeStyle(),
            
            layout: {
                name: this.options.layout,
                animate: true,
                animationDuration: 1000,
                fit: true,
                padding: 50,
                
                // Layout-specific options
                ...(this.options.layout === 'cola' && {
                    nodeSpacing: 10,
                    edgeLength: 100,
                    animate: true,
                    randomize: false,
                    maxSimulationTime: 4000
                }),
                
                ...(this.options.layout === 'cose' && {
                    nodeRepulsion: 400000,
                    nodeOverlap: 10,
                    idealEdgeLength: 50,
                    edgeElasticity: 100,
                    nestingFactor: 5,
                    gravity: 80,
                    numIter: 1000
                }),
                
                ...(this.options.layout === 'dagre' && {
                    nodeSep: 100,
                    edgeSep: 100,
                    rankSep: 200,
                    rankDir: 'TB',
                    ranker: 'longest-path'
                })
            },
            
            // Interaction options
            zoomingEnabled: true,
            userZoomingEnabled: true,
            panningEnabled: true,
            userPanningEnabled: true,
            boxSelectionEnabled: true,
            selectionType: 'single',
            
            // Rendering options
            textureOnViewport: false,
            motionBlur: true,
            motionBlurOpacity: 0.2,
            wheelSensitivity: 0.1,
            pixelRatio: 'auto'
        };
        
        this.cy = cytoscape(cyConfig);
        
        // Add event listeners for Cytoscape
        this.setupCytoscapeEvents();
    }
    
    initializeD3() {
        const mainContainer = document.getElementById(`${this.containerId}-main`);
        
        // Create SVG
        this.d3Svg = d3.select(mainContainer)
            .append('svg')
            .attr('width', this.options.width)
            .attr('height', this.options.height)
            .attr('viewBox', `0 0 ${this.options.width} ${this.options.height}`)
            .style('background-color', this.options.theme === 'dark' ? '#1a1a1a' : '#ffffff');
        
        // Create groups for different elements
        this.d3Groups = {
            links: this.d3Svg.append('g').attr('class', 'links'),
            nodes: this.d3Svg.append('g').attr('class', 'nodes'),
            labels: this.d3Svg.append('g').attr('class', 'labels')
        };
        
        // Initialize force simulation
        this.simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(this.options.width / 2, this.options.height / 2))
            .force('collision', d3.forceCollide().radius(30));
        
        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on('zoom', (event) => {
                this.d3Groups.links.attr('transform', event.transform);
                this.d3Groups.nodes.attr('transform', event.transform);
                this.d3Groups.labels.attr('transform', event.transform);
            });
        
        this.d3Svg.call(zoom);
        
        // Add event listeners for D3
        this.setupD3Events();
    }
    
    getCytoscapeStyle() {
        const baseStyle = [
            // Node styles
            {
                selector: 'node',
                style: {
                    'background-color': 'data(color)',
                    'border-color': '#333',
                    'border-width': 2,
                    'border-opacity': 0.8,
                    'width': 'data(size)',
                    'height': 'data(size)',
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'font-size': '12px',
                    'font-weight': 'bold',
                    'color': this.options.theme === 'dark' ? '#ffffff' : '#333333',
                    'text-outline-width': 2,
                    'text-outline-color': this.options.theme === 'dark' ? '#000000' : '#ffffff',
                    'text-outline-opacity': 0.8,
                    'overlay-padding': '6px',
                    'z-index': 10
                }
            },
            
            // Edge styles
            {
                selector: 'edge',
                style: {
                    'width': 'data(weight)',
                    'line-color': 'data(color)',
                    'target-arrow-color': 'data(color)',
                    'target-arrow-shape': 'triangle',
                    'arrow-scale': 1.5,
                    'curve-style': 'bezier',
                    'opacity': 0.8,
                    'label': 'data(label)',
                    'font-size': '10px',
                    'color': this.options.theme === 'dark' ? '#cccccc' : '#666666',
                    'text-outline-width': 1,
                    'text-outline-color': this.options.theme === 'dark' ? '#000000' : '#ffffff',
                    'text-rotation': 'autorotate',
                    'text-margin-y': -10
                }
            },
            
            // Node type specific styles
            {
                selector: 'node[type="content"]',
                style: {
                    'background-color': '#4CAF50',
                    'shape': 'rectangle'
                }
            },
            
            {
                selector: 'node[type="keyword"]',
                style: {
                    'background-color': '#2196F3',
                    'shape': 'ellipse'
                }
            },
            
            {
                selector: 'node[type="topic"]',
                style: {
                    'background-color': '#FF9800',
                    'shape': 'diamond'
                }
            },
            
            {
                selector: 'node[type="competitor"]',
                style: {
                    'background-color': '#F44336',
                    'shape': 'hexagon'
                }
            },
            
            // Selected node styles
            {
                selector: 'node:selected',
                style: {
                    'border-width': 4,
                    'border-color': '#FFD700',
                    'background-color': 'data(selectedColor)',
                    'z-index': 20
                }
            },
            
            // Highlighted node styles
            {
                selector: 'node.highlighted',
                style: {
                    'border-width': 3,
                    'border-color': '#FF4081',
                    'z-index': 15
                }
            },
            
            // Filtered node styles
            {
                selector: 'node.filtered',
                style: {
                    'opacity': 0.3,
                    'z-index': 1
                }
            },
            
            // Connected edge styles
            {
                selector: 'edge.connected',
                style: {
                    'width': 4,
                    'line-color': '#FF4081',
                    'target-arrow-color': '#FF4081',
                    'opacity': 1,
                    'z-index': 15
                }
            },
            
            // Hover styles
            {
                selector: 'node:active',
                style: {
                    'overlay-opacity': 0.2,
                    'overlay-color': '#FFD700'
                }
            }
        ];
        
        return baseStyle;
    }
    
    setupCytoscapeEvents() {
        // Node selection
        this.cy.on('select', 'node', (event) => {
            const node = event.target;
            this.selectedNodes.add(node.id());
            this.highlightConnectedNodes(node);
            this.updateNodeInfo(node);
        });
        
        this.cy.on('unselect', 'node', (event) => {
            const node = event.target;
            this.selectedNodes.delete(node.id());
            this.clearHighlights();
            this.updateNodeInfo(null);
        });
        
        // Node hover
        this.cy.on('mouseover', 'node', (event) => {
            const node = event.target;
            this.showNodeTooltip(node, event);
        });
        
        this.cy.on('mouseout', 'node', (event) => {
            this.hideNodeTooltip();
        });
        
        // Double-click to focus
        this.cy.on('dblclick', 'node', (event) => {
            const node = event.target;
            this.focusOnNode(node);
        });
        
        // Context menu
        this.cy.on('cxttap', 'node', (event) => {
            const node = event.target;
            this.showContextMenu(node, event);
        });
        
        // Layout finished
        this.cy.on('layoutstop', () => {
            this.updateStats();
        });
        
        // Graph ready
        this.cy.ready(() => {
            this.updateStats();
        });
    }
    
    setupD3Events() {
        // Add drag behavior
        const drag = d3.drag()
            .on('start', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });
        
        // Store drag behavior for later use
        this.d3Drag = drag;
    }
    
    setupEventListeners() {
        // Window resize
        window.addEventListener('resize', () => {
            this.handleResize();
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (event) => {
            if (event.target.closest(`#${this.containerId}`)) {
                this.handleKeyboardShortcuts(event);
            }
        });
    }
    
    createControls() {
        const controlsContainer = document.getElementById(`${this.containerId}-controls`);
        
        controlsContainer.innerHTML = `
            <div class="graph-controls-group">
                <div class="control-group">
                    <label for="${this.containerId}-layout">Layout:</label>
                    <select id="${this.containerId}-layout" class="control-select">
                        <option value="cola" ${this.options.layout === 'cola' ? 'selected' : ''}>Cola</option>
                        <option value="cose" ${this.options.layout === 'cose' ? 'selected' : ''}>COSE</option>
                        <option value="dagre" ${this.options.layout === 'dagre' ? 'selected' : ''}>Dagre</option>
                        <option value="circle" ${this.options.layout === 'circle' ? 'selected' : ''}>Circle</option>
                        <option value="grid" ${this.options.layout === 'grid' ? 'selected' : ''}>Grid</option>
                        <option value="concentric" ${this.options.layout === 'concentric' ? 'selected' : ''}>Concentric</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label for="${this.containerId}-search">Search:</label>
                    <input type="text" id="${this.containerId}-search" class="control-input" placeholder="Search nodes...">
                </div>
                
                <div class="control-group">
                    <label for="${this.containerId}-filter">Filter:</label>
                    <div class="filter-checkboxes" id="${this.containerId}-filters">
                        <!-- Filter checkboxes will be added dynamically -->
                    </div>
                </div>
                
                <div class="control-group">
                    <button id="${this.containerId}-fit" class="control-button">Fit</button>
                    <button id="${this.containerId}-center" class="control-button">Center</button>
                    <button id="${this.containerId}-reset" class="control-button">Reset</button>
                </div>
                
                <div class="control-group">
                    <button id="${this.containerId}-export" class="control-button">Export</button>
                    <button id="${this.containerId}-fullscreen" class="control-button">Fullscreen</button>
                </div>
            </div>
        `;
        
        // Add event listeners for controls
        this.setupControlEvents();
    }
    
    setupControlEvents() {
        // Layout change
        const layoutSelect = document.getElementById(`${this.containerId}-layout`);
        layoutSelect.addEventListener('change', (event) => {
            this.changeLayout(event.target.value);
        });
        
        // Search
        const searchInput = document.getElementById(`${this.containerId}-search`);
        searchInput.addEventListener('input', (event) => {
            this.searchNodes(event.target.value);
        });
        
        // Control buttons
        document.getElementById(`${this.containerId}-fit`).addEventListener('click', () => {
            this.fitToContent();
        });
        
        document.getElementById(`${this.containerId}-center`).addEventListener('click', () => {
            this.centerGraph();
        });
        
        document.getElementById(`${this.containerId}-reset`).addEventListener('click', () => {
            this.resetGraph();
        });
        
        document.getElementById(`${this.containerId}-export`).addEventListener('click', () => {
            this.exportGraph();
        });
        
        document.getElementById(`${this.containerId}-fullscreen`).addEventListener('click', () => {
            this.toggleFullscreen();
        });
    }
    
    createLegend() {
        const legendContainer = document.getElementById(`${this.containerId}-legend`);
        
        const nodeTypes = [
            { type: 'content', color: '#4CAF50', label: 'Content' },
            { type: 'keyword', color: '#2196F3', label: 'Keyword' },
            { type: 'topic', color: '#FF9800', label: 'Topic' },
            { type: 'competitor', color: '#F44336', label: 'Competitor' }
        ];
        
        const legendHTML = `
            <div class="legend-title">Node Types</div>
            <div class="legend-items">
                ${nodeTypes.map(type => `
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: ${type.color}"></div>
                        <span class="legend-label">${type.label}</span>
                    </div>
                `).join('')}
            </div>
        `;
        
        legendContainer.innerHTML = legendHTML;
    }
    
    // Data loading and management
    async loadData(dataSource) {
        try {
            let data;
            
            if (typeof dataSource === 'string') {
                // Load from URL
                const response = await fetch(dataSource);
                data = await response.json();
            } else {
                // Use provided data
                data = dataSource;
            }
            
            this.data = this.processData(data);
            this.renderGraph();
            this.updateFilterOptions();
            this.updateStats();
            
        } catch (error) {
            console.error('Error loading graph data:', error);
            this.showError('Failed to load graph data');
        }
    }
    
    processData(data) {
        // Normalize data structure
        const nodes = data.nodes || [];
        const edges = data.edges || data.links || [];
        
        // Process nodes
        const processedNodes = nodes.map(node => ({
            id: node.id || node.data?.id,
            label: node.label || node.data?.label || node.name || node.id,
            type: node.type || node.data?.type || 'default',
            size: node.size || node.data?.size || 40,
            color: node.color || node.data?.color || this.getNodeColor(node.type),
            ...node.data,
            ...node
        }));
        
        // Process edges
        const processedEdges = edges.map(edge => ({
            id: edge.id || `${edge.source}-${edge.target}`,
            source: edge.source,
            target: edge.target,
            label: edge.label || edge.type || '',
            type: edge.type || 'default',
            weight: edge.weight || edge.strength || 2,
            color: edge.color || this.getEdgeColor(edge.type),
            ...edge.data,
            ...edge
        }));
        
        return {
            nodes: processedNodes,
            edges: processedEdges
        };
    }
    
    getNodeColor(type) {
        const colors = {
            content: '#4CAF50',
            keyword: '#2196F3',
            topic: '#FF9800',
            competitor: '#F44336',
            default: '#9E9E9E'
        };
        return colors[type] || colors.default;
    }
    
    getEdgeColor(type) {
        const colors = {
            relates: '#666666',
            contains: '#2196F3',
            mentions: '#FF9800',
            competes: '#F44336',
            default: '#999999'
        };
        return colors[type] || colors.default;
    }
    
    renderGraph() {
        if (this.options.renderer === 'cytoscape') {
            this.renderCytoscapeGraph();
        } else {
            this.renderD3Graph();
        }
    }
    
    renderCytoscapeGraph() {
        // Convert data to Cytoscape format
        const elements = [];
        
        // Add nodes
        this.data.nodes.forEach(node => {
            elements.push({
                data: {
                    id: node.id,
                    label: node.label,
                    type: node.type,
                    size: node.size,
                    color: node.color,
                    selectedColor: this.lightenColor(node.color, 0.3),
                    ...node
                }
            });
        });
        
        // Add edges
        this.data.edges.forEach(edge => {
            elements.push({
                data: {
                    id: edge.id,
                    source: edge.source,
                    target: edge.target,
                    label: edge.label,
                    type: edge.type,
                    weight: edge.weight,
                    color: edge.color,
                    ...edge
                }
            });
        });
        
        // Add elements to graph
        this.cy.add(elements);
        
        // Run layout
        this.cy.layout({
            name: this.options.layout,
            animate: true,
            animationDuration: 1000,
            fit: true,
            padding: 50
        }).run();
    }
    
    renderD3Graph() {
        // Clear existing elements
        this.d3Groups.links.selectAll('*').remove();
        this.d3Groups.nodes.selectAll('*').remove();
        this.d3Groups.labels.selectAll('*').remove();
        
        // Create links
        const links = this.d3Groups.links
            .selectAll('line')
            .data(this.data.edges)
            .enter()
            .append('line')
            .attr('class', 'link')
            .attr('stroke', d => d.color)
            .attr('stroke-width', d => d.weight)
            .attr('stroke-opacity', 0.8);
        
        // Create nodes
        const nodes = this.d3Groups.nodes
            .selectAll('circle')
            .data(this.data.nodes)
            .enter()
            .append('circle')
            .attr('class', 'node')
            .attr('r', d => d.size / 2)
            .attr('fill', d => d.color)
            .attr('stroke', '#333')
            .attr('stroke-width', 2)
            .call(this.d3Drag);
        
        // Create labels
        const labels = this.d3Groups.labels
            .selectAll('text')
            .data(this.data.nodes)
            .enter()
            .append('text')
            .attr('class', 'label')
            .attr('text-anchor', 'middle')
            .attr('dy', '.35em')
            .attr('font-size', '12px')
            .attr('font-weight', 'bold')
            .attr('fill', this.options.theme === 'dark' ? '#ffffff' : '#333333')
            .text(d => d.label);
        
        // Add interaction events
        nodes
            .on('click', (event, d) => {
                this.selectNode(d);
            })
            .on('mouseover', (event, d) => {
                this.showD3NodeTooltip(d, event);
            })
            .on('mouseout', (event, d) => {
                this.hideNodeTooltip();
            });
        
        // Update simulation
        this.simulation
            .nodes(this.data.nodes)
            .on('tick', () => {
                links
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
                
                nodes
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);
                
                labels
                    .attr('x', d => d.x)
                    .attr('y', d => d.y);
            });
        
        this.simulation.force('link')
            .links(this.data.edges);
        
        this.simulation.alpha(1).restart();
    }
    
    // Interaction methods
    highlightConnectedNodes(node) {
        if (this.options.renderer === 'cytoscape') {
            const connectedNodes = node.connectedNodes();
            const connectedEdges = node.connectedEdges();
            
            // Highlight connected nodes
            connectedNodes.addClass('highlighted');
            connectedEdges.addClass('connected');
            
            // Fade other nodes
            this.cy.nodes().not(connectedNodes).not(node).addClass('filtered');
            this.cy.edges().not(connectedEdges).addClass('filtered');
        }
    }
    
    clearHighlights() {
        if (this.options.renderer === 'cytoscape') {
            this.cy.nodes().removeClass('highlighted filtered');
            this.cy.edges().removeClass('connected filtered');
        }
    }
    
    focusOnNode(node) {
        if (this.options.renderer === 'cytoscape') {
            this.cy.animate({
                center: {
                    eles: node
                },
                zoom: 2
            }, {
                duration: 1000
            });
        }
    }
    
    searchNodes(query) {
        this.searchQuery = query.toLowerCase();
        
        if (!query) {
            this.clearSearch();
            return;
        }
        
        if (this.options.renderer === 'cytoscape') {
            this.cy.nodes().forEach(node => {
                const label = node.data('label').toLowerCase();
                if (label.includes(this.searchQuery)) {
                    node.removeClass('filtered').addClass('highlighted');
                } else {
                    node.addClass('filtered').removeClass('highlighted');
                }
            });
        }
    }
    
    clearSearch() {
        this.searchQuery = '';
        if (this.options.renderer === 'cytoscape') {
            this.cy.nodes().removeClass('filtered highlighted');
        }
    }
    
    changeLayout(layoutName) {
        this.options.layout = layoutName;
        
        if (this.options.renderer === 'cytoscape') {
            this.cy.layout({
                name: layoutName,
                animate: true,
                animationDuration: 1000,
                fit: true,
                padding: 50
            }).run();
        }
    }
    
    fitToContent() {
        if (this.options.renderer === 'cytoscape') {
            this.cy.fit();
        } else {
            // D3 fit implementation
            const bounds = this.d3Groups.nodes.node().getBBox();
            const width = bounds.width;
            const height = bounds.height;
            const midX = bounds.x + width / 2;
            const midY = bounds.y + height / 2;
            
            if (width === 0 || height === 0) return;
            
            const scale = Math.min(
                this.options.width / width,
                this.options.height / height
            ) * 0.9;
            
            const transform = d3.zoomIdentity
                .translate(this.options.width / 2, this.options.height / 2)
                .scale(scale)
                .translate(-midX, -midY);
            
            this.d3Svg.transition()
                .duration(750)
                .call(d3.zoom().transform, transform);
        }
    }
    
    centerGraph() {
        if (this.options.renderer === 'cytoscape') {
            this.cy.center();
        } else {
            const transform = d3.zoomIdentity
                .translate(this.options.width / 2, this.options.height / 2)
                .scale(1);
            
            this.d3Svg.transition()
                .duration(750)
                .call(d3.zoom().transform, transform);
        }
    }
    
    resetGraph() {
        this.clearSearch();
        this.clearHighlights();
        this.selectedNodes.clear();
        this.filteredNodeTypes.clear();
        this.fitToContent();
        this.updateFilterOptions();
    }
    
    // Tooltip methods
    showNodeTooltip(node, event) {
        const tooltip = this.getOrCreateTooltip();
        const nodeData = node.data();
        
        tooltip.innerHTML = `
            <div class="tooltip-header">
                <strong>${nodeData.label}</strong>
                <span class="tooltip-type">${nodeData.type}</span>
            </div>
            <div class="tooltip-content">
                <div class="tooltip-row">
                    <span class="tooltip-label">ID:</span>
                    <span class="tooltip-value">${nodeData.id}</span>
                </div>
                <div class="tooltip-row">
                    <span class="tooltip-label">Connections:</span>
                    <span class="tooltip-value">${node.degree()}</span>
                </div>
                ${nodeData.description ? `
                    <div class="tooltip-row">
                        <span class="tooltip-label">Description:</span>
                        <span class="tooltip-value">${nodeData.description}</span>
                    </div>
                ` : ''}
            </div>
        `;
        
        tooltip.style.display = 'block';
        tooltip.style.left = event.renderedPosition.x + 10 + 'px';
        tooltip.style.top = event.renderedPosition.y - 10 + 'px';
    }
    
    showD3NodeTooltip(node, event) {
        const tooltip = this.getOrCreateTooltip();
        
        tooltip.innerHTML = `
            <div class="tooltip-header">
                <strong>${node.label}</strong>
                <span class="tooltip-type">${node.type}</span>
            </div>
            <div class="tooltip-content">
                <div class="tooltip-row">
                    <span class="tooltip-label">ID:</span>
                    <span class="tooltip-value">${node.id}</span>
                </div>
                ${node.description ? `
                    <div class="tooltip-row">
                        <span class="tooltip-label">Description:</span>
                        <span class="tooltip-value">${node.description}</span>
                    </div>
                ` : ''}
            </div>
        `;
        
        tooltip.style.display = 'block';
        tooltip.style.left = event.pageX + 10 + 'px';
        tooltip.style.top = event.pageY - 10 + 'px';
    }
    
    hideNodeTooltip() {
        const tooltip = document.getElementById('graph-tooltip');
        if (tooltip) {
            tooltip.style.display = 'none';
        }
    }
    
    getOrCreateTooltip() {
        let tooltip = document.getElementById('graph-tooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'graph-tooltip';
            tooltip.className = 'graph-tooltip';
            document.body.appendChild(tooltip);
        }
        return tooltip;
    }
    
    // Export methods
    exportGraph() {
        if (this.options.renderer === 'cytoscape') {
            const png = this.cy.png({
                output: 'blob',
                bg: this.options.theme === 'dark' ? '#1a1a1a' : '#ffffff',
                full: true,
                scale: 2
            });
            
            const link = document.createElement('a');
            link.download = `knowledge-graph-${Date.now()}.png`;
            link.href = URL.createObjectURL(png);
            link.click();
        } else {
            // D3 export implementation
            const svgData = new XMLSerializer().serializeToString(this.d3Svg.node());
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            
            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                
                const link = document.createElement('a');
                link.download = `knowledge-graph-${Date.now()}.png`;
                link.href = canvas.toDataURL();
                link.click();
            };
            
            img.src = 'data:image/svg+xml;base64,' + btoa(svgData);
        }
    }
    
    // Utility methods
    lightenColor(color, percent) {
        const num = parseInt(color.replace('#', ''), 16);
        const amt = Math.round(2.55 * percent);
        const R = (num >> 16) + amt;
        const G = (num >> 8 & 0x00FF) + amt;
        const B = (num & 0x0000FF) + amt;
        return `#${(0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
            (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
            (B < 255 ? B < 1 ? 0 : B : 255)).toString(16).slice(1)}`;
    }
    
    updateStats() {
        const statsContainer = document.getElementById(`${this.containerId}-stats`);
        const nodeCount = this.data.nodes.length;
        const edgeCount = this.data.edges.length;
        const selectedCount = this.selectedNodes.size;
        
        statsContainer.innerHTML = `
            <span>Nodes: ${nodeCount}</span>
            <span>Edges: ${edgeCount}</span>
            <span>Selected: ${selectedCount}</span>
        `;
    }
    
    updateFilterOptions() {
        const filtersContainer = document.getElementById(`${this.containerId}-filters`);
        const nodeTypes = [...new Set(this.data.nodes.map(node => node.type))];
        
        filtersContainer.innerHTML = nodeTypes.map(type => `
            <label class="filter-checkbox">
                <input type="checkbox" value="${type}" checked>
                <span>${type}</span>
            </label>
        `).join('');
        
        // Add event listeners for filter checkboxes
        filtersContainer.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', (event) => {
                const nodeType = event.target.value;
                if (event.target.checked) {
                    this.filteredNodeTypes.delete(nodeType);
                } else {
                    this.filteredNodeTypes.add(nodeType);
                }
                this.applyFilters();
            });
        });
    }
    
    applyFilters() {
        if (this.options.renderer === 'cytoscape') {
            this.cy.nodes().forEach(node => {
                const nodeType = node.data('type');
                if (this.filteredNodeTypes.has(nodeType)) {
                    node.addClass('filtered');
                } else {
                    node.removeClass('filtered');
                }
            });
        }
    }
    
    handleResize() {
        if (this.options.renderer === 'cytoscape') {
            this.cy.resize();
        } else {
            // D3 resize implementation
            const container = document.getElementById(`${this.containerId}-main`);
            const rect = container.getBoundingClientRect();
            
            this.options.width = rect.width;
            this.options.height = rect.height;
            
            this.d3Svg
                .attr('width', this.options.width)
                .attr('height', this.options.height)
                .attr('viewBox', `0 0 ${this.options.width} ${this.options.height}`);
            
            this.simulation
                .force('center', d3.forceCenter(this.options.width / 2, this.options.height / 2))
                .restart();
        }
    }
    
    handleKeyboardShortcuts(event) {
        if (event.ctrlKey || event.metaKey) {
            switch (event.key) {
                case 'f':
                    event.preventDefault();
                    this.fitToContent();
                    break;
                case 'c':
                    event.preventDefault();
                    this.centerGraph();
                    break;
                case 'r':
                    event.preventDefault();
                    this.resetGraph();
                    break;
                case 'e':
                    event.preventDefault();
                    this.exportGraph();
                    break;
            }
        }
    }
    
    showError(message) {
        const mainContainer = document.getElementById(`${this.containerId}-main`);
        mainContainer.innerHTML = `
            <div class="graph-error">
                <h4>Error</h4>
                <p>${message}</p>
            </div>
        `;
    }
    
    destroy() {
        if (this.cy) {
            this.cy.destroy();
        }
        if (this.simulation) {
            this.simulation.stop();
        }
        
        // Remove event listeners
        window.removeEventListener('resize', this.handleResize);
        
        // Clear container
        this.container.innerHTML = '';
        
        // Remove tooltip
        const tooltip = document.getElementById('graph-tooltip');
        if (tooltip) {
            tooltip.remove();
        }
    }
}

// Export for use in other modules
window.GraphVisualization = GraphVisualization;