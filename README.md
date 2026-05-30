# 🏥 Medical Chatbot — RAG-based AI Assistant

A medical question-answering chatbot built with LangChain, Pinecone, Flask, and OpenRouter (free LLM).

## Tech Stack
- **LLM:** OpenRouter (Free models — Llama, GPT-OSS etc.)
- **Embeddings:** HuggingFace sentence-transformers/all-MiniLM-L6-v2
- **Vector DB:** Pinecone
- **Framework:** Flask
- **PDF Source:** Medical Encyclopedia (Gale)

## How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/medical-chatbot.git
cd medical-chatbot
```

### 2. Create conda environment
```bash
conda create -n medibot python=3.10 -y
conda activate medibot
pip install -r requirements.txt
```

### 3. Create `.env` file
```
PINECONE_API_KEY=your_pinecone_key
OPENROUTER_API_KEY=your_openrouter_key
```

### 4. Add your medical PDF to `data/` folder

### 5. Upload to Pinecone (one time only)
```bash
python store_index.py
```

### 6. Run the app
```bash
python app.py
```

Open `http://localhost:8080` in your browser.

## Project Structure
```
├── app.py              # Flask web app
├── store_index.py      # PDF → Pinecone uploader
├── src/
│   ├── helper.py       # PDF loader + embeddings
│   └── prompt.py       # System prompt
├── templates/
│   └── chat.html       # Chat UI
├── static/
│   └── style.css       # Styling
└── data/               # Put your medical PDF here
```

## Made by
B.Tech EEE Student — Lendi College 
