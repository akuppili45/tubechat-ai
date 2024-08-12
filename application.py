from flask import Flask, jsonify
import os
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.docstore.document import Document
import boto3
from flask_cors import CORS

application = app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return "ai"

@app.route("/vector-store/<video_id>")
def create_vector_store(video_id):
    # create a vector store
    # store the vector store in s3
    # return the vector store
    return jsonify({"message":"vector store"} )

# example: https://www.youtube.com/watch?v=4VDZRR07Eqw
# the v= part is the video id that can be used in cloud storage/DB
s3 = boto3.client('s3')
@app.route("/invoke/<video_id>/<chat>")
def invoke(video_id, chat):
    # Load the text content from the s3 bucket
    try:
        response = s3.get_object(Bucket='tubechat-contents', Key=f'{video_id}/captions.txt')
        full_text = response['Body'].read().decode('utf-8')
        docs = [Document(page_content=full_text)]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory=f"./chroma/{video_id}")

        # Retrieve and generate using the relevant snippets of the blog.
        retriever = vectorstore.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return jsonify({"chatResponse": rag_chain.invoke(chat)})
    except Exception as e:
        return jsonify({"error": str(e) })



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.run()