// DOM Elements
const tileGrid = document.getElementById('tileGrid');
const modal = document.getElementById('modal');
const modalTitle = document.getElementById('modalTitle');
const modalContent = document.getElementById('modalContent');
const modalImage = document.getElementById('modalImage');
const modalSentimentIcon = document.getElementById('modalSentimentIcon');
const modalSources = document.getElementById('modalSources');
const closeBtn = document.getElementById('closeBtn');
const pairSelect = document.getElementById('pairSelect');
const datePicker = document.getElementById('datePicker');

const themeToggle = document.getElementById('themeToggle');
if (!themeToggle) {
    console.error('Theme toggle button not found!');
}

const sentimentConfig = {
    Positive: {
        color: '#32CD32',
        icon: '/bull2.png'
    },
    Bullish: {
        color: '#90EE90',
        icon: '/bull.png'
    },
    Neutral: {
        color: '#D3D3D3',
        icon: '/neutral.png'
    },
    Bearish: {
        color: '#FFA07A',
        icon: '/bear.png'
    },
    Negative: {
        color: '#FF4500',
        icon: '/bear2.png'
    }
};

async function fetchPredictions(date) {
    try {
        // Format date properly to match MMDDYYYY format with leading zeros
        const [year, month, day] = date.split('-');
        const formattedDate = `${month}${day}${year}`;
        
        // Convert pair format (e.g., "BTC/USDT" to "BTC-USDT")
        //const formattedPair = pair.replace('/', '-');
        
        // Construct the path to the predictions.yaml file
        const path = `/news/${formattedDate}/ainews.yaml`;
        console.log('Fetching from path:', path);
        
        const response = await fetch(path);
        if (!response.ok) {
            throw new Error(`Failed to fetch predictions: ${response.statusText}`);
        }
        
        const yamlText = await response.text();
        console.log('Received YAML:', yamlText);
        
        // Parse YAML and return predictions array
        const yamlData = jsyaml.load(yamlText);
        return yamlData.news.map(prediction => ({
            ...prediction,
            // Convert the single sourcelink to the sourceLinks array format
            sourceLinks: [{
                title: "Source",
                url: prediction.sourcelink
            }],
            // Convert single image to images array format
            images: [{
                path: prediction.image
            }]
        }));
        
    } catch (error) {
        console.error('Error fetching predictions:', error);
        return [];
    }
}

function createTile(prediction) {
    const tileElement = document.createElement('div');
    tileElement.className = 'tile';
    tileElement.style.borderColor = sentimentConfig[prediction.sentiment].color;
    
    const mainImage = prediction.images && prediction.images.length > 0 
        ? prediction.images[0].path 
        : '/placeholder.png';
    
    tileElement.innerHTML = `
        <h2>${prediction.source}</h2>
        <p>${prediction.preview}</p>
        <img src="${mainImage}" alt="${prediction.title}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>📊</text></svg>'">
    `;
    tileElement.addEventListener('click', () => openModal(prediction));
    return tileElement;
}

function openModal(prediction) {
    modalTitle.textContent = prediction.preview;
    modalContent.textContent = prediction.fullContent;
    
    const mainImage = prediction.images && prediction.images.length > 0 
        ? prediction.images[0].path 
        : '/placeholder.png';
    
    modalImage.src = mainImage;
    modalImage.alt = prediction.title;
    modalSentimentIcon.src = sentimentConfig[prediction.sentiment].icon;
    modalSentimentIcon.alt = prediction.sentiment;
    
    modal.querySelector('.modal-content').style.backgroundColor = 
        `${sentimentConfig[prediction.sentiment].color}22`;
    
    modalSources.innerHTML = prediction.sourceLinks
        .map(source => `<a href="${source.url}" target="_blank">${source.title || 'Source'}</a>`)
        .join('');
    
    // Use classList to add show class
    modal.classList.add('show');
    document.body.classList.add('modal-open');
}

function closeModal() {
    // Use classList to remove show class
    modal.classList.remove('show');
    document.body.classList.remove('modal-open');
}

// Update event listeners
closeBtn.onclick = closeModal;

window.onclick = function(event) {
    if (event.target === modal) {
        closeModal();
    }
};

// Add keyboard support for closing modal
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape' && modal.classList.contains('show')) {
        closeModal();
    }
});

async function renderTiles() {
    //const selectedPair = pairSelect.value;
    const selectedDate = datePicker.value;
    
    tileGrid.innerHTML = '';
    tileGrid.innerHTML = '<div class="loading">Loading predictions...</div>';
    
    try {
        const predictions = await fetchPredictions(selectedDate);
        
        tileGrid.innerHTML = '';
        
        if (predictions.length === 0) {
            tileGrid.innerHTML = '<div class="no-data">No News available for this date</div>';
            return;
        }
        
        predictions.forEach(prediction => {
            tileGrid.appendChild(createTile(prediction));
        });
    } catch (error) {
        console.error('Error rendering tiles:', error);
        tileGrid.innerHTML = '<div class="error">Failed to load predictions</div>';
    }
}


async function handleDataUpdate() {
    await renderTiles();
}

pairSelect.addEventListener('change', handleDataUpdate);
datePicker.addEventListener('change', handleDataUpdate);

// Initialize date picker with today's date
const now = new Date();
const year = now.getFullYear();
const month = String(now.getMonth() + 1).padStart(2, '0');
const day = String(now.getDate()).padStart(2, '0');
const today = `${year}-${month}-${day}`;
datePicker.value = today;

// Initial render
handleDataUpdate();

function setTheme(theme) {
    console.log(`Setting theme to: ${theme}`);
    try {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
        console.log('Theme applied successfully');
        console.log('Current data-theme attribute:', document.documentElement.getAttribute('data-theme'));
    } catch (error) {
        console.error('Error setting theme:', error);
    }
}

function toggleTheme() {
    console.log('Theme toggle clicked');
    const currentTheme = document.documentElement.getAttribute('data-theme');
    console.log('Current theme:', currentTheme);
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    console.log('Switching to:', newTheme);
    setTheme(newTheme);
}

function initializeTheme() {
    console.log('Initializing theme');
    const savedTheme = localStorage.getItem('theme') || 'light';
    console.log('Saved theme from localStorage:', savedTheme);
    setTheme(savedTheme);
}

// Make sure event listener is properly attached
if (themeToggle) {
    console.log('Adding click event listener to theme toggle button');
    themeToggle.addEventListener('click', (e) => {
        console.log('Theme toggle clicked with event:', e);
        toggleTheme();
    });
} else {
    console.error('Could not attach event listener - theme toggle button not found');
}

// Initialize theme when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing theme');
    initializeTheme();
});

// Log initial state
console.log('Initial HTML data-theme:', document.documentElement.getAttribute('data-theme'));
console.log('Initial localStorage theme:', localStorage.getItem('theme'));