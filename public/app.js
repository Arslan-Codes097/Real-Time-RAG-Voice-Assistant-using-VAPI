// Arslan.AI Voice RAG Assistant (Vapi.ai) Engine

document.addEventListener('DOMContentLoaded', async () => {
  const emptyState = document.getElementById('emptyState');
  const messagesDiv = document.getElementById('messages');
  const textInput = document.getElementById('textInput');
  const sendBtn = document.getElementById('sendBtn');
  const micBtn = document.getElementById('micBtn');
  const vapiCallBtn = document.getElementById('vapiCallBtn');
  const newChatBtn = document.getElementById('newChatBtn');
  const voiceReplyToggle = document.getElementById('voiceReplyToggle');
  const modelSelect = document.getElementById('modelSelect');
  const statusBar = document.getElementById('statusBar');
  const ttsPlayer = document.getElementById('ttsPlayer');

  let isVoiceReplyEnabled = true;
  let isRecording = false;
  let isVapiCallActive = false;
  let recognition = null;
  let vapiInstance = null;

  // Initial document fetch
  fetchDocuments();

  // Fetch Vapi Credentials from backend
  try {
    const healthRes = await fetch('/api/health');
    const healthData = await healthRes.json();
    if (healthData.vapi_public_key && window.vapiSDK) {
      vapiInstance = window.vapiSDK.run({
        apiKey: healthData.vapi_public_key,
        assistant: healthData.vapi_assistant_id
      });
    }
  } catch (err) {
    console.log('Vapi SDK init note:', err);
  }

  // Voice reply toggle
  voiceReplyToggle.addEventListener('click', () => {
    isVoiceReplyEnabled = !isVoiceReplyEnabled;
    voiceReplyToggle.setAttribute('data-enabled', isVoiceReplyEnabled);
    voiceReplyToggle.innerText = isVoiceReplyEnabled ? 'Enabled' : 'Disabled';
  });

  // New Chat reset
  newChatBtn.addEventListener('click', () => {
    messagesDiv.innerHTML = '';
    emptyState.style.display = 'block';
    statusBar.innerText = 'Ready';
  });

  // Send message
  sendBtn.addEventListener('click', sendMessage);
  textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendMessage();
  });

  // Web Speech API STT setup
  if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      isRecording = true;
      micBtn.classList.add('recording');
      statusBar.innerText = 'Listening... Speak your question now.';
    };

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      textInput.value = transcript;
      sendMessage();
    };

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      statusBar.innerText = 'Speech recognition error. Try again.';
      stopRecording();
    };

    recognition.onend = () => {
      stopRecording();
    };
  }

  micBtn.addEventListener('click', () => {
    if (!recognition) {
      alert('Speech Recognition is not supported in this browser. Please type your query.');
      return;
    }
    if (isRecording) {
      recognition.stop();
    } else {
      recognition.start();
    }
  });

  function stopRecording() {
    isRecording = false;
    micBtn.classList.remove('recording');
    statusBar.innerText = 'Ready';
  }

  // Vapi Voice Call Handler
  vapiCallBtn.addEventListener('click', async () => {
    if (isVapiCallActive) {
      isVapiCallActive = false;
      vapiCallBtn.classList.remove('active');
      statusBar.innerText = 'Vapi voice call ended.';
      if (vapiInstance && vapiInstance.stop) {
        vapiInstance.stop();
      }
      return;
    }

    statusBar.innerText = 'Connecting to Vapi.ai Real-Time Voice Server...';
    try {
      const healthRes = await fetch('/api/health');
      const healthData = await healthRes.json();

      if (!healthData.vapi_public_key || healthData.vapi_public_key.includes('your_vapi')) {
        alert('Vapi Public Key is not set in .env! Configure VAPI_PUBLIC_KEY and VAPI_ASSISTANT_ID from your https://dashboard.vapi.ai dashboard.');
        statusBar.innerText = 'Vapi API credentials missing in .env.';
        return;
      }

      isVapiCallActive = true;
      vapiCallBtn.classList.add('active');
      statusBar.innerText = 'Vapi Voice Call Active! Speak naturally to your assistant.';
      emptyState.style.display = 'none';
      appendMessage('assistant', 'Vapi Real-Time Voice Assistant active. Ask anything about your uploaded documents!');

      if (vapiInstance && vapiInstance.start) {
        vapiInstance.start();
      }
    } catch (err) {
      console.error('Vapi connection error:', err);
      statusBar.innerText = 'Error connecting to Vapi voice server.';
    }
  });

  // ---------- Send Message & Execute RAG ----------

  async function sendMessage() {
    const text = textInput.value.trim();
    if (!text) return;

    emptyState.style.display = 'none';
    appendMessage('user', text);
    textInput.value = '';
    statusBar.innerText = 'Retrieving document knowledge & generating answer...';

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

        if (isVoiceReplyEnabled) {
          playTTS(data.reply);
        }
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

  async function playTTS(text) {
    try {
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
      });
      const blob = await res.blob();
      const audioUrl = URL.createObjectURL(blob);
      ttsPlayer.src = audioUrl;
      ttsPlayer.play();
    } catch (err) {
      console.error('TTS Playback Error:', err);
    }
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
  statusBar.innerText = `Uploading and embedding '${file.name}' into vector database...`;

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
