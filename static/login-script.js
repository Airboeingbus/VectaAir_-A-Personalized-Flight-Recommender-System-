/**
 * Flight Recommendation System - Login Page JavaScript
 * Handles user authentication
 */

// DOM Elements
const loginForm = document.getElementById('login-form');
const userIdInput = document.getElementById('login-user-id');
const passwordInput = document.getElementById('login-password');
const loginBtn = document.getElementById('login-btn');
const errorMessageEl = document.getElementById('login-error');

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Auto-focus user ID input
    document.getElementById('login-user-id').focus();
    
    loginForm.addEventListener('submit', function(e) {
        e.preventDefault();
        handleLogin();
    });
    passwordInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            handleLogin();
        }
    });
});

/**
 * Handle login form submission
 */
async function handleLogin() {
    const userId = userIdInput.value.trim();
    const password = passwordInput.value.trim();
    
    if (!userId || !password) {
        showError('Please enter both User ID and password');
        return;
    }
    
    clearError();
    
    try {
        const response = await fetch('/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                user_id: userId, 
                password: password 
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Redirect to search page
            window.location.href = '/search';
        } else {
            showError(data.error || 'Login failed. Please check your credentials.');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Connection error. Please try again.');
    }
}

/**
 * UI Helper Functions
 */

function showError(message) {
    errorMessageEl.textContent = message;
    errorMessageEl.classList.remove('hidden');
}

function clearError() {
    errorMessageEl.textContent = '';
    errorMessageEl.classList.add('hidden');
}

// Initial state
clearError();
