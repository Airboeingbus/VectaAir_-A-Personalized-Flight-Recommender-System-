/**
 * Flight Recommendation System - Frontend JavaScript
 * Handles search, autocomplete, and preference-based recommendations
 */

// State
let originCode = null;
let destinationCode = null;
let autocompleteTimeout = null;

// DOM Elements
const originInput = document.getElementById('origin-input');
const destinationInput = document.getElementById('destination-input');
const originSuggestions = document.getElementById('origin-suggestions');
const destinationSuggestions = document.getElementById('destination-suggestions');
const reliabilitySelect = document.getElementById('reliability-select');
const priceSlider = document.getElementById('price-slider');
const timeSlider = document.getElementById('time-slider');
const reliabilitySlider = document.getElementById('reliability-slider');
const departureTime = document.getElementById('departure-time');
const priceLabel = document.getElementById('price-label');
const timeLabel = document.getElementById('time-label');
const reliabilityLabel = document.getElementById('reliability-label');
const preferenceSummary = document.getElementById('preference-summary');
const summaryText = document.getElementById('summary-text');
const resultsCountSelect = document.getElementById('results-count');
const searchBtn = document.getElementById('search-btn');
const logoutBtn = document.getElementById('logout-btn');

// Trip Planning Elements
const tripOneWay = document.getElementById('trip-one-way');
const tripRoundTrip = document.getElementById('trip-round-trip');
const departureDate = document.getElementById('departure-date');
const returnDate = document.getElementById('return-date');
const returnDateGroup = document.getElementById('return-date-group');
const tripInfo = document.getElementById('trip-info');
const tripDisplayText = document.getElementById('trip-display-text');

// Results DOM
const loadingEl = document.getElementById('loading');
const errorMessageEl = document.getElementById('error-message');
const resultsTable = document.getElementById('results-table');
const resultsTbody = document.getElementById('results-tbody');
const emptyStateEl = document.getElementById('empty-state');
const statusBadge = document.getElementById('status-badge');

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    searchBtn.addEventListener('click', handleSearch);
    logoutBtn.addEventListener('click', handleLogout);
    
    // Trip type selection
    tripOneWay.addEventListener('change', handleTripTypeChange);
    tripRoundTrip.addEventListener('change', handleTripTypeChange);
    
    // Date validation
    departureDate.addEventListener('change', validateDates);
    returnDate.addEventListener('change', validateDates);
    
    // Autocomplete
    originInput.addEventListener('input', function(e) {
        handleAutocomplete(e.target.value, 'origin');
    });
    
    destinationInput.addEventListener('input', function(e) {
        handleAutocomplete(e.target.value, 'destination');
    });
    
    // Slider listeners
    priceSlider.addEventListener('input', updatePreferenceSummary);
    timeSlider.addEventListener('input', updatePreferenceSummary);
    reliabilitySlider.addEventListener('input', updatePreferenceSummary);
    departureTime.addEventListener('change', updatePreferenceSummary);
});

/**
 * Handle logout
 */
function handleLogout() {
    window.location.href = '/logout';
}

/**
 * Handle autocomplete search with debouncing
 */
async function handleAutocomplete(query, type) {
    const suggestionsEl = type === 'origin' ? originSuggestions : destinationSuggestions;
    
    clearTimeout(autocompleteTimeout);
    
    if (!query || query.length < 1) {
        suggestionsEl.classList.add('hidden');
        return;
    }
    
    autocompleteTimeout = setTimeout(async () => {
        try {
            const response = await fetch(`/search_airports?q=${encodeURIComponent(query)}`);
            const results = await response.json();
            
            if (results.length === 0) {
                suggestionsEl.classList.add('hidden');
                return;
            }
            
            suggestionsEl.innerHTML = results.map(airport => 
                `<div class="suggestion" data-code="${airport.iata_code}">${airport.display}</div>`
            ).join('');
            
            suggestionsEl.classList.remove('hidden');
            
            // Click handler for suggestions
            suggestionsEl.querySelectorAll('.suggestion').forEach(item => {
                item.addEventListener('click', function() {
                    const code = this.getAttribute('data-code');
                    const display = this.textContent;
                    
                    if (type === 'origin') {
                        originInput.value = display;
                        originCode = code;
                    } else {
                        destinationInput.value = display;
                        destinationCode = code;
                    }
                    suggestionsEl.classList.add('hidden');
                });
            });
        } catch (error) {
            console.error('Autocomplete error:', error);
        }
    }, 300);
}

/**
 * Update preference summary dynamically
 */
function updatePreferenceSummary() {
    const priceVal = parseInt(priceSlider.value);
    const timeVal = parseInt(timeSlider.value);
    const reliabilityVal = parseInt(reliabilitySlider.value);
    
    // Update labels
    if (priceVal < 33) {
        priceLabel.textContent = 'Cheap';
    } else if (priceVal < 67) {
        priceLabel.textContent = 'Balanced';
    } else {
        priceLabel.textContent = 'Premium';
    }
    
    if (timeVal < 33) {
        timeLabel.textContent = 'Fastest';
    } else if (timeVal < 67) {
        timeLabel.textContent = 'Balanced';
    } else {
        timeLabel.textContent = 'Comfortable';
    }
    
    if (reliabilityVal < 33) {
        reliabilityLabel.textContent = 'Loose';
    } else if (reliabilityVal < 67) {
        reliabilityLabel.textContent = 'Balanced';
    } else {
        reliabilityLabel.textContent = 'Strict';
    }
    
    // Build summary
    const summaryParts = [];
    if (priceVal < 33) summaryParts.push('Cheap');
    else if (priceVal > 67) summaryParts.push('Premium');
    
    if (timeVal > 67) summaryParts.push('Comfortable');
    else if (timeVal < 33) summaryParts.push('Fast');
    
    if (reliabilityVal > 67) summaryParts.push('High Reliability');
    
    if (summaryParts.length > 0) {
        summaryText.textContent = summaryParts.join(' + ');
        preferenceSummary.style.display = 'block';
    } else {
        preferenceSummary.style.display = 'none';
    }
}

/**
 * Handle trip type change (one-way vs round-trip)
 */
function handleTripTypeChange() {
    const isRoundTrip = tripRoundTrip.checked;
    returnDateGroup.style.display = isRoundTrip ? 'block' : 'none';
    
    if (!isRoundTrip) {
        returnDate.value = '';
    }
    validateDates();
}

/**
 * Validate that return date >= departure date
 */
function validateDates() {
    if (!departureDate.value) return;
    
    if (tripRoundTrip.checked && returnDate.value) {
        if (new Date(returnDate.value) < new Date(departureDate.value)) {
            showError('Return date must be after departure date');
            returnDate.value = '';
        }
    }
}

/**
 * Handle search button click
 */
async function handleSearch() {
    if (!originCode || !destinationCode) {
        showError('Please select both departure and destination airports from the list');
        return;
    }
    
    if (originCode === destinationCode) {
        showError('Origin and destination must be different');
        return;
    }
    
    if (!departureDate.value) {
        showError('Please select a departure date');
        return;
    }
    
    if (tripRoundTrip.checked && !returnDate.value) {
        showError('Please select a return date for round-trip');
        return;
    }
    
    clearError();
    hideResults();
    showLoading(true);
    
    const topN = parseInt(resultsCountSelect.value);
    const tripType = tripOneWay.checked ? 'one-way' : 'round-trip';
    
    // Collect preferences including trip info
    const preferences = {
        origin: originCode,
        destination: destinationCode,
        price_pref: parseInt(priceSlider.value),
        time_pref: parseInt(timeSlider.value),
        reliability_pref: parseInt(reliabilitySlider.value),
        departure_time: departureTime.value,
        top_n: topN,
        trip_type: tripType,
        departure_date: departureDate.value,
        return_date: returnDate.value || null
    };
    
    // Store trip info for display
    window.currentTripInfo = {
        trip_type: tripType,
        departure_date: departureDate.value,
        return_date: returnDate.value
    };
    
    try {
        const response = await fetch('/recommend_with_preferences', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(preferences)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            showError(data.error || 'Error getting recommendations');
            return;
        }
        
        displayRecommendations(data);
        
    } catch (error) {
        console.error('Error:', error);
        showError('Connection error');
    } finally {
        showLoading(false);
    }
}

/**
 * Display recommendations with breakdown and explanations
 */
function displayRecommendations(data) {
    const recommendations = data.recommendations;
    const isFallback = data.fallback || false;
    
    if (!recommendations || recommendations.length === 0) {
        showError('No recommendations found');
        return;
    }
    
    resultsTbody.innerHTML = '';
    
    recommendations.forEach((rec, index) => {
        const row = document.createElement('tr');
        row.style.cursor = 'pointer';
        
        // Apply top-result class to first recommendation
        if (index === 0) {
            row.classList.add('top-result');
        }
        
        const rankCell = document.createElement('td');
        rankCell.className = 'col-rank';
        rankCell.textContent = index + 1;
        
        const flightCell = document.createElement('td');
        flightCell.className = 'col-flight';
        flightCell.textContent = rec.flight_id;
        
        const scoreCell = document.createElement('td');
        scoreCell.className = 'col-score';
        scoreCell.textContent = rec.score.toFixed(3);
        
        const breakdownCell = document.createElement('td');
        breakdownCell.className = 'col-breakdown';
        const breakdown = rec.breakdown || {};
        breakdownCell.textContent = `${(breakdown.collaborative || 0).toFixed(2)} | ${(breakdown.reliability || 0).toFixed(2)} | ${(breakdown.preference || 0).toFixed(2)} | ${(breakdown.bonus || 0).toFixed(2)}`;
        
        const ratingCell = document.createElement('td');
        ratingCell.className = 'col-rating';
        if (rec.score >= 0.90) {
            ratingCell.textContent = '⭐⭐⭐⭐⭐ Excellent';
        } else if (rec.score >= 0.75) {
            ratingCell.textContent = '⭐⭐⭐⭐ Good';
        } else if (rec.score >= 0.50) {
            ratingCell.textContent = '⭐⭐⭐ Fair';
        } else {
            ratingCell.textContent = '⭐⭐ Consider others';
        }
        
        row.appendChild(rankCell);
        row.appendChild(flightCell);
        row.appendChild(scoreCell);
        row.appendChild(breakdownCell);
        row.appendChild(ratingCell);
        
        // Show explanation when row is clicked
        row.addEventListener('click', function() {
            showExplanation(rec);
        });
        
        resultsTbody.appendChild(row);
    });
    
    // Show explanation for first recommendation
    showExplanation(recommendations[0]);
    displayTripInfo();
    showResults();
    
    if (isFallback) {
        showError(data.message);
    }
}

/**
 * UI Helper Functions
 */

function showExplanation(recommendation) {
    const explanationBox = document.getElementById('explanation-box');
    const explanationText = document.getElementById('explanation-text');
    
    if (explanationBox && explanationText) {
        explanationText.textContent = recommendation.explanation || 'Recommended based on your preferences.';
        explanationBox.classList.remove('hidden');
        explanationBox.classList.add('show');
    }
}

function displayTripInfo() {
    if (!window.currentTripInfo) return;
    
    const tripInfo = window.currentTripInfo;
    let tripText = '';
    
    if (tripInfo.trip_type === 'one-way') {
        const depDate = new Date(tripInfo.departure_date).toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric', 
            year: 'numeric' 
        });
        tripText = `One-way | ${depDate}`;
    } else {
        const depDate = new Date(tripInfo.departure_date).toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric' 
        });
        const retDate = new Date(tripInfo.return_date).toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric', 
            year: 'numeric' 
        });
        tripText = `Round-trip | ${depDate} → ${retDate}`;
    }
    
    tripDisplayText.textContent = tripText;
    tripInfo.classList.remove('hidden');
    tripInfo.classList.add('show');
}

function showLoading(show) {
    if (show) {
        loadingEl.classList.remove('hidden');
    } else {
        loadingEl.classList.add('hidden');
    }
}

function showError(message) {
    errorMessageEl.textContent = message;
    errorMessageEl.classList.remove('hidden');
}

function clearError() {
    errorMessageEl.textContent = '';
    errorMessageEl.classList.add('hidden');
}

function showResults() {
    resultsTable.classList.remove('hidden');
    emptyStateEl.classList.add('hidden');
}

function hideResults() {
    resultsTable.classList.add('hidden');
    emptyStateEl.classList.remove('hidden');
}

// Initial state
hideResults();
clearError();

// Initial state
hideResults();
clearError();

