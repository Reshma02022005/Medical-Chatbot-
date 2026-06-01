from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore as PineconeStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from src.prompt import *
import os
import time

app = Flask(__name__, template_folder='templetes')
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"

docsearch = PineconeStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)

chatModel = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | chatModel
    | StrOutputParser()
)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        print("User:", msg)
        for attempt in range(3):
            try:
                answer = rag_chain.invoke(msg)
                print("Bot:", answer)
                return str(answer)
            except Exception as e:
                if "429" in str(e):
                    print(f"Rate limited, retrying in 10s... ({attempt+1}/3)")
                    time.sleep(10)
                else:
                    raise e
        return "I'm busy right now, please ask again!"
    except Exception as e:
        print("ERROR:", str(e))
        return str(e)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
