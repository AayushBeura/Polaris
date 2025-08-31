# Polaris AI - RAG-Powered Meeting Assistant

## 📌 Overview
Polaris AI is an intelligent **meeting assistant** that integrates **Recall.ai**, **Cerebras**, and **Murf AI** to provide real-time meeting support.  
It can:
- Join virtual meetings via Recall.ai.
- Listen and respond contextually when invoked with its name (“Polaris”).
- Use **RAG (Retrieval-Augmented Generation)** to answer from uploaded documents (PDF, DOCX, TXT).
- Generate **MOM (Minutes of Meeting)** after each session (as PDF).
- Provide **real-time speech-to-speech responses** using Murf AI voices.
- Allow **mute/unmute**, **response mode switching**, and **voice customization**.

---

## ⚙️ Features
- 🔗 **Meeting Integration**: Connects to online meetings through Recall.ai bots.  
- 📝 **Transcript Handling**: Captures and processes live transcripts.  
- 🧠 **AI Assistant**: Context-aware responses using Cerebras LLM.  
- 📚 **RAG System**: Upload documents for contextual Q&A.  
- 🎤 **Voice Output**: Generates natural speech with Murf AI voices.  
- 📄 **MOM Generation**: Automatically creates concise minutes of meeting in PDF.  
- 🎛️ **UI Controls**:  
  - Start/Leave Meeting  
  - Upload/Delete Documents  
  - Change Voice  
  - Switch Modes: *Detailed, Quick, Document-based, General*  
  - Mute/Unmute  

---

## 🚀 Setup

### 1. Clone Repository
```bash
git clone <repo_url>
cd polaris-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. API Keys
Polaris requires three API keys. Store them in `user_config.json`:
```json
{
  "RECALL_API_KEY": "your_recall_api_key",
  "CEREBRAS_API_KEY": "your_cerebras_api_key",
  "MURF_API_KEY": "your_murf_api_key"
}
```

### 4. Run Application
```bash
python polaris.py
```
The app will start a Flask server with Socket.IO support.  
Ngrok tunneling is available for external meeting connections.

---

## 🖥️ Interface

- Web-based dashboard (Flask + Socket.IO).  
- Blurred background with animated UI elements.  
- Sections:  
  - Meeting Controls  
  - Document Management  
  - System Status  
  - Transcript & Logs  

---

## 🔄 Workflow
(Insert Flowchart Here)

---

## 📦 Project Structure
```
polaris.py             # Main Flask + Socket.IO backend
user_config.json       # Stores API keys
uploads/               # Uploaded documents & generated MOM PDFs
static/                # Assets (e.g., logo, backgrounds)
requirements.txt       # Python dependencies
```

---

## 🛠️ Tech Stack
- **Backend**: Flask, Flask-SocketIO  
- **AI APIs**: Cerebras LLM, Recall.ai, Murf AI  
- **Vector Search**: FAISS + Sentence Transformers  
- **Document Parsing**: PyPDF2, python-docx  
- **PDF Generation**: FPDF  
- **UI**: HTML, CSS, JavaScript (served by Flask)

---

## 📑 License
MIT License
