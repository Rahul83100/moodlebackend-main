let currentSessionIdx = null;

async function startSession(idxFromUrl = null) {
    const idxInput = document.getElementById('index-input');
    // Use argument OR input value
    const idx = idxFromUrl || idxInput.value;
    const errorMsg = document.getElementById('error-msg');

    if (!idx) {
        showError("Please enter an Index ID.");
        return;
    }

    try {
        const response = await fetch('/api/session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index_id: parseInt(idx) })
        });

        if (!response.ok) {
            throw new Error('Index not found');
        }

        const data = await response.json();

        // Enter Session View
        currentSessionIdx = parseInt(idx);
        const sessionSpan = document.getElementById('session-id');
        sessionSpan.innerText = currentSessionIdx;
        sessionSpan.dataset.id = currentSessionIdx; // Store ID in DOM

        document.getElementById('initial-content').innerText = data.content;

        // Update URL to use query parameter
        const newUrl = `/?id=${currentSessionIdx}`;
        if (window.location.search !== `?id=${currentSessionIdx}`) {
            history.pushState({ id: currentSessionIdx }, '', newUrl);
        }

        // Switch Views
        document.getElementById('login-view').classList.add('hidden');
        document.getElementById('session-view').classList.remove('hidden');
        document.getElementById('session-view').classList.add('fade-in');

        // Clear chat
        document.getElementById('chat-box').innerHTML = '';
        errorMsg.style.display = 'none';

    } catch (err) {
        // ...
    }
}

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const msg = input.value;
    if (!msg) return;

    // Fetch ID from the HTML element itself
    const sessionSpan = document.getElementById('session-id');
    const currentId = sessionSpan.dataset.id;

    if (!currentId) {
        alert("Session ID not found in DOM");
        return;
    }

    // Add User Message
    addMessage(msg, 'user');
    input.value = '';

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index_id: parseInt(currentId), question: msg })
        });

        const data = await response.json();

        // Add Bot Message
        addMessage(data.answer, 'bot', data.relevance_score, data.breakdown, data.sources);

    } catch (err) {
        addMessage("Error processing your request.", 'bot');
    }
}

function addMessage(text, type, relevance = null, breakdown = null, sources = null) {
    const box = document.getElementById('chat-box');
    const div = document.createElement('div');
    div.className = `message ${type}-msg fade-in`;

    if (relevance) {
        const headerDiv = document.createElement('div');
        headerDiv.style.display = 'flex';
        headerDiv.style.flexDirection = 'column';
        headerDiv.style.marginBottom = '0.5rem';

        const relTag = document.createElement('div');
        relTag.className = 'relevance-tag';
        relTag.innerText = `Hybrid Score: ${relevance}`;
        headerDiv.appendChild(relTag);

        if (breakdown) {
            const breakTag = document.createElement('div');
            breakTag.style.fontSize = '0.7rem';
            breakTag.style.color = '#94a3b8';
            breakTag.style.marginTop = '2px';
            breakTag.innerText = breakdown;
            headerDiv.appendChild(breakTag);
        }
        div.appendChild(headerDiv);
    }

    // Strip out backend-fallback markdown sources if present to show stylized UI sources
    let cleanText = text;
    const sourceMarkerIndex = cleanText.indexOf("\n\n**Sources:**");
    if (sourceMarkerIndex !== -1) {
        cleanText = cleanText.substring(0, sourceMarkerIndex);
    }

    const content = document.createElement('div');
    content.innerText = cleanText;
    div.appendChild(content);

    // Extract sources if not passed as array but present in text fallback
    let displaySources = sources;
    if (!displaySources && sourceMarkerIndex !== -1) {
        const rawSourcesStr = text.substring(sourceMarkerIndex + 14).trim();
        if (rawSourcesStr) {
            displaySources = rawSourcesStr.split(",").map(s => s.trim());
        }
    }

    // Suppress sources display for fallback or greeting responses
    if (cleanText.includes("I don't have enough specific material") || 
        cleanText.includes("I am sorry, but I cannot engage") || 
        cleanText.includes("admin side issues")) {
        displaySources = null;
    }

    if (displaySources && displaySources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.style.marginTop = '10px';
        sourcesDiv.style.paddingTop = '8px';
        sourcesDiv.style.borderTop = '1px solid rgba(255, 255, 255, 0.1)';
        sourcesDiv.style.fontSize = '0.75rem';
        sourcesDiv.style.color = 'var(--text-secondary)';

        let sourcesHtml = "<strong>Sources:</strong>";
        displaySources.forEach(src => {
            sourcesHtml += ` <span style="display:inline-block; background:rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); padding:2px 8px; border-radius:12px; margin-left:6px; font-weight:500; font-size:0.7rem; color:var(--text-primary);">📄 ${src}</span>`;
        });
        sourcesDiv.innerHTML = sourcesHtml;
        div.appendChild(sourcesDiv);
    }

    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
}

function exitSession() {
    document.getElementById('session-view').classList.add('hidden');
    document.getElementById('login-view').classList.remove('hidden');
    document.getElementById('login-view').classList.add('fade-in');

    currentSessionIdx = null;
    history.pushState(null, '', '/');
    document.getElementById('index-input').value = '';
}

async function uploadFile() {
    const indexId = document.getElementById("upload-index-id").value;
    const fileInput = document.getElementById("upload-file");
    const statusEl = document.getElementById("upload-status");

    if (!indexId || !fileInput.files[0]) {
        statusEl.textContent = "Please enter an Index ID and select a file.";
        statusEl.style.color = "#ef4444";
        return;
    }

    statusEl.textContent = "Uploading & Ingesting...";
    statusEl.style.color = "var(--text-secondary)";

    const formData = new FormData();
    formData.append("index_id", indexId);
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch('/api/upload', {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }

        const result = await response.json();
        statusEl.textContent = "Success! Redirecting...";
        statusEl.style.color = "var(--success)";

        // Optional: Redirect to the new session after success
        setTimeout(() => {
            window.location.href = `/?id=${indexId}`;
        }, 1500);

    } catch (e) {
        console.error(e);
        statusEl.textContent = "Upload failed. See console.";
        statusEl.style.color = "#ef4444";
    }
}

function showError(msg) {
    const el = document.getElementById('error-msg');
    el.innerText = msg;
    el.style.display = 'block';
}

// Enter key support
document.getElementById('chat-input').addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    const params = new URLSearchParams(window.location.search);
    const id = params.get('id');

    // Also check path for backward compatibility
    if (id) {
        startSession(id);
    } else {
        const path = window.location.pathname;
        const match = path.match(/^\/(\d+)$/);
        if (match) {
            startSession(match[1]);
        }
    }
});

// Handle Back Button
window.addEventListener('popstate', (event) => {
    const params = new URLSearchParams(window.location.search);
    const id = params.get('id');

    if (id) {
        startSession(id);
    } else {
        // Check path fallback or exit
        const match = window.location.pathname.match(/^\/(\d+)$/);
        if (match) {
            startSession(match[1]);
        } else {
            exitSession();
        }
    }
});
