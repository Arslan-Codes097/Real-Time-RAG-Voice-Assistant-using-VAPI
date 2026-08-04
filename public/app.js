// Arslan.AI Voice RAG Assistant (Universal Vapi Engine)

document.addEventListener('DOMContentLoaded', async () => {
  const emptyState = document.getElementById('emptyState');
  const messagesDiv = document.getElementById('messages');
  const textInput = document.getElementById('textInput');
  const sendBtn = document.getElementById('sendBtn');
  const vapiCallBtn = document.getElementById('vapiCallBtn');
  const newChatBtn = document.getElementById('newChatBtn');
  const modelSelect = document.getElementById('modelSelect');
  const statusBar = document.getElementById('statusBar');

  let vapiPublicKey = "";
  let vapiAssistantId = "";
  let isVapiCallActive = false;
  let vapiInstance = null;

  // Initial document fetch
  fetchDocuments();

  // Load environment keys dynamically from backend
  async function loadConfig() {
    try {
      const res = await fetch('/api/health');
      const data = await res.json();
      vapiPublicKey = data.vapi_public_key || "";
      vapiAssistantId = data.vapi_assistant_id || "";
    } catch (err) {
      console.log('Health config fetch error:', err);
    }
  }

  loadConfig();

  function onCallStart() {
    isVapiCallActive = true;
    vapiCallBtn.classList.add('active');
    statusBar.innerText = 'Vapi Voice Call Active! Speak into your microphone.';
    emptyState.style.display = 'none';
    appendMessage('assistant', 'Vapi Real-Time Voice Session Active. How can I help you with your documents?');
  }

  function onCallEnd() {
    isVapiCallActive = false;
    vapiCallBtn.classList.remove('active');
    statusBar.innerText = 'Vapi voice call ended.';
  }

  // ---------- Universal Vapi Call Handler ----------

  vapiCallBtn.addEventListener('click', async () => {
    if (isVapiCallActive) {
      if (vapiInstance && vapiInstance.stop) vapiInstance.stop();
      onCallEnd();
      return;
    }

    if (!vapiPublicKey || !vapiAssistantId) {
      await loadConfig();
    }

    if (!vapiPublicKey || !vapiAssistantId) {
      statusBar.innerText = 'Error: VAPI_PUBLIC_KEY or VAPI_ASSISTANT_ID missing in .env file.';
      return;
    }

    statusBar.innerText = 'Connecting to Vapi Voice Server...';

    try {
      if (window.Vapi && typeof window.Vapi === 'function') {
        vapiInstance = new window.Vapi(vapiPublicKey);
        vapiInstance.on('call-start', onCallStart);
        vapiInstance.on('call-end', onCallEnd);
        vapiInstance.on('speech-start', () => { statusBar.innerText = 'Vapi Assistant speaking...'; });
        vapiInstance.on('speech-end', () => { statusBar.innerText = 'Listening...'; });
        vapiInstance.on('message', (msg) => {
          if (msg.type === 'transcript' && msg.transcriptType === 'final') {
            appendMessage(msg.role === 'user' ? 'user' : 'assistant', msg.transcript);
          }
        });
        await vapiInstance.start(vapiAssistantId);
      } else if (window.vapiSDK) {
        if (typeof window.vapiSDK.run === 'function') {
          vapiInstance = window.vapiSDK.run({
            apiKey: vapiPublicKey,
            assistant: vapiAssistantId
          });
          onCallStart();
        } else if (typeof window.vapiSDK === 'function') {
          vapiInstance = new window.vapiSDK(vapiPublicKey);
          await vapiInstance.start(vapiAssistantId);
          onCallStart();
        }
      } else {
        statusBar.innerText = 'Vapi Web SDK loading... Please click call again in 3 seconds.';
      }
    } catch (err) {
      console.error('Vapi connection error:', err);
      statusBar.innerText = 'Connection error. Click call again to retry.';
    }
  });

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
