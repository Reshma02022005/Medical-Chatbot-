import os
from dotenv import load_dotenv
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

print("Step 1: Loading PDF...")
extracted_data = load_pdf_file(data='data/')
print(f"  Loaded {len(extracted_data)} pages")

print("Step 2: Splitting text into chunks...")
text_chunks = text_split(extracted_data)
print(f"  Created {len(text_chunks)} chunks")

print("Step 3: Loading embedding model...")
embeddings = download_hugging_face_embeddings()
print("  Embeddings ready")

print("Step 4: Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    print(f"  Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("  Index created!")
else:
    print(f"  Index '{index_name}' already exists")

print("Step 5: Uploading vectors to Pinecone (this takes 5-10 mins)...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
print("DONE! All vectors uploaded successfully!")