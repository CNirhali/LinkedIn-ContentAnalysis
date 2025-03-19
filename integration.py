"""
// JavaScript for a browser extension (conceptual code)
// This would be implemented in a browser extension project

// Content script to analyze LinkedIn messages and posts
document.addEventListener('DOMContentLoaded', function() {
    // Scan for new content periodically
    setInterval(scanContent, 5000);
});

function scanContent() {
    // Select all posts and messages on the page
    const posts = document.querySelectorAll('.feed-shared-update-v2');
    const messages = document.querySelectorAll('.msg-conversation-card');

    // Process posts
    posts.forEach(post => {
        const content = post.querySelector('.feed-shared-text').textContent;

        // Send content to your API for analysis
        fetch('https://your-app.com/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content })
        })
        .then(response => response.json())
        .then(result => {
            if (result.is_harmful && result.severity_score > 70) {
                // Add warning overlay
                addWarningOverlay(post, result.severity_score);
            }
        });
    });

    // Process messages similarly
}

function addWarningOverlay(element, score) {
    const overlay = document.createElement('div');
    overlay.className = 'safety-warning';
    overlay.innerHTML = `
        <div class="warning-icon">⚠️</div>
        <div class="warning-text">
            This content may be harmful (Safety Score: ${score.toFixed(0)}%)
        </div>
        <button class="show-anyway">Show Anyway</button>
        <button class="report">Report</button>
    `;

    element.style.position = 'relative';
    element.appendChild(overlay);

    // Add event listeners for buttons
    overlay.querySelector('.show-anyway').addEventListener('click', function() {
        overlay.style.display = 'none';
    });

    overlay.querySelector('.report').addEventListener('click', function() {
        reportContent(element);
    });
}
"""
