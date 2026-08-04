// Arslan.AI Voice RAG Assistant (Vapi.ai Engine)

document.addEventListener('DOMContentLoaded', async () => {
  const emptyState = document.getElementById('emptyState');
  const messagesDiv = document.getElementById('messages');
  const textInput = document.getElementById('textInput');
  const sendBtn = document.getElementById('sendBtn');
  const vapiCallBtn = document.getElementById('vapiCallBtn');
  const newChatBtn = document.getElementById('newChatBtn');
  const modelSelect = document.getElementById('modelSelect');
  const statusBar = document.getElementById('statusBar');

  const DEFAULT_PUBLIC_KEY = "93bc6cf3-0c28-4aa9-b264-1c9f0c775702";
  const DEFAULT_ASSISTANT_ID = "7b4cb4d9-b2a2-496b-ab0b-32b102602af2";

  let vapiPublicKey = DEFAULT_PUBLIC_KEY;
  let vapiAssistantId = DEFAULT_ASSISTANT_ID;
  let isVapiCallActive = false;
  let vapi = null;

  // Initial document fetch
  fetchDocuments();

  // Fetch Vapi Credentials from backend health check
  try {
    const healthRes = await fetch('/api/health');
    const healthData = await healthRes.json();
    if (healthData.vapi_public_key) vapiPublicKey = healthData.vapi_public_key;
    if (healthData.vapi_assistant_id) vapiAssistantId = healthData.vapi_assistant_id;
  } catch (err) {
    console.log('Health check note:', err);
  }

  function getVapiInstance() {
    if (vapi) return vapi;

    const VapiClass = window.Vapi || (window.vapiSDK && window.vapiSDK.Vapi);
    if (VapiClass && typeof VapiClass === 'function') {
      vapi = new VapiClass(vapiPublicKey);
      setupVapiListeners();
      return vapi;
    }
    return null;
  }

  function setupVapiListeners() {
    if (!vapi) return;

    vapi.on('call-start', () => {
      isVapiCallActive = true;
      vapiCallBtn.classList.add('active');
      statusBar.innerText = 'Vapi Voice Call Active! Speak naturally into your microphone.';
      emptyState.style.display = 'none';
      appendMessage('assistant', 'Vapi Real-Time Voice Session Connected! Ask me anything about your documents.');
    });

    vapi.on('call-end', () => {
      isVapiCallActive = false;
      vapiCallBtn.classList.remove('active');
      statusBar.innerText = 'Vapi voice call ended.';
    });

    vapi.on('speech-start', () => {
      statusBar.innerText = 'Vapi Assistant speaking...';
    });

    vapi.on('speech-end', () => {
      statusBar.innerText = 'Vapi Voice Call Active! Listening...';
    });

    vapi.on('message', (message) => {
      if (message.type === 'transcript' && message.transcriptType === 'final') {
        if (message.role === 'user') {
          appendMessage('user', message.transcript);
        } else if (message.role === 'assistant') {
          appendMessage('assistant', message.transcript);
        }
      }
    });

    vapi.on('error', (err) => {
      console.error('Vapi Web SDK error:', err);
      statusBar.innerText = 'Vapi Voice Error. Check browser microphone permissions.';
    });
  }

  // New Chat reset
  newChatBtn.addEventListener('click', () => {
    messagesDiv.innerHTML = '';
    emptyState.style.display = 'block';
    statusBar.innerText = 'Ready';
  });

  // Send text message
  sendBtn.addEventListener('click', sendMessage);
  textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendMessage();
  });

  // ---------- Vapi Single Call Button Click Handler ----------

  vapiCallBtn.addEventListener('click', async () => {
    if (isVapiCallActive) {
      if (vapi) vapi.stop();
      isVapiCallActive = false;
      vapiCallBtn.classList.remove('active');
      statusBar.innerText = 'Call ended.';
      return;
    }

    statusBar.innerText = 'Connecting Vapi.ai Voice Stream...';
    const instance = getVapiInstance();

    if (instance && vapiAssistantId) {
      try {
        await instance.start(vapiAssistantId);
      } catch (err) {
        console.error('Failed to start Vapi call:', err);
        statusBar.innerText = 'Error starting Vapi call.';
      }
    } else {
      // Fallback: If SDK fails to load from CDN, open Vapi Dashboard Call link or inform user
      statusBar.innerText = 'Vapi Web SDK initializing... Try clicking call again.';
    }
  });

  // ---------- Send Message & Execute LangChain RAG ----------

  async function sendMessage() {
    const text = textInput.value.trim();
    if (!text) return;

    emptyState.style.display = 'none';
    appendMessage('user', text);
    textInput.value = '';
    statusBar.innerText = 'Searching document knowledge & generating answer...';

    const selectedModel = modelSelect.value;

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, model: selectedModel })
      });

      const data = await response.json();
      statusBar.innerText = 'Ready';

      if (data.reply) {
        appendMessage('assistant', data.reply, data.sources);
      } else {
        appendMessage('assistant', data.detail || 'Sorry, I could not generate a response.');
      }
    } catch (err) {
      console.error('RAG Query Error:', err);
      statusBar.innerText = 'Error connecting to server.';
      appendMessage('assistant', 'Error executing RAG pipeline.');
    }
  }

  function appendMessage(role, text, sources = []) {
    const bubble = document.createElement('div');
    bubble.className = `bubble ${role}`;
    bubble.innerText = text;

    if (sources && sources.length > 0) {
      const chipsDiv = document.createElement('div');
      chipsDiv.className = 'sources-chips';
      sources.forEach(src => {
        const chip = document.createElement('span');
        chip.className = 'source-chip';
        chip.innerText = `📄 ${src}`;
        chipsDiv.appendChild(chip);
      });
      bubble.appendChild(chipsDiv);
    }

    messagesDiv.appendChild(bubble);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }
});

// ---------- RAG Document Management Functions ----------

async function fetchDocuments() {
  const docList = document.getElementById('docList');
  if (!docList) return;

  try {
    const res = await fetch('/api/documents');
    const data = await res.json();
    const docs = data.documents || [];

    if (docs.length === 0) {
      docList.innerHTML = '<div class="doc-loading">No documents uploaded yet.</div>';
      return;
    }

    docList.innerHTML = '';
    docs.forEach(doc => {
      const item = document.createElement('div');
      item.className = 'doc-item';
      item.innerHTML = `
        <span class="doc-name" title="${doc.source}">📄 ${doc.source}</span>
        <button class="delete-doc-btn" onclick="deleteDocument('${doc.source}')" title="Delete from knowledge base">🗑️</button>
      `;
      docList.appendChild(item);
    });
  } catch (err) {
    console.error('Error fetching documents:', err);
    docList.innerHTML = '<div class="doc-loading">Error loading document list.</div>';
  }
}

async function uploadSelectedFile(event) {
  const file = event.target.files[0];
  if (!file) return;

  const statusBar = document.getElementById('statusBar');
  statusBar.innerText = `Uploading and embedding '${file.name}' into Supabase Vector DB...`;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });

    const data = await res.json();
    if (res.ok) {
      statusBar.innerText = `Successfully ingested '${file.name}'!`;
      fetchDocuments();
    } else {
      alert(`Upload failed: ${data.detail || 'Unknown error'}`);
      statusBar.innerText = 'Upload failed.';
    }
  } catch (err) {
    console.error('Upload Error:', err);
    alert('Failed to upload file.');
    statusBar.innerText = 'Error uploading document.';
  }

  event.target.value = '';
}

async function deleteDocument(filename) {
  if (!confirm(`Are you sure you want to delete '${filename}' from the vector knowledge base?`)) {
    return;
  }

  const statusBar = document.getElementById('statusBar');
  statusBar.innerText = `Deleting '${filename}' from Supabase Vector DB...`;

  try {
    const res = await fetch(`/api/documents/${encodeURIComponent(filename)}`, {
      method: 'DELETE'
    });

    if (res.ok) {
      statusBar.innerText = `Deleted '${filename}' successfully!`;
      fetchDocuments();
    } else {
      const data = await res.json();
      alert(`Delete failed: ${data.detail || 'Unknown error'}`);
      statusBar.innerText = 'Delete failed.';
    }
  } catch (err) {
    console.error('Delete Error:', err);
    alert('Failed to delete document.');
    statusBar.innerText = 'Error deleting document.';
  }
}
