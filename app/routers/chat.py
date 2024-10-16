import logging
import chromadb
from typing import List
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate)
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq
from pydantic import BaseModel
import uuid

chat_router = r = APIRouter()

logger = logging.getLogger("uvicorn")


class Query(BaseModel):
    query: str


def init_index():
    print("loading postman collections")

    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    documents = reader.load_data()

    print("index creating with `%d` documents", len(documents))

    try:
        chroma_client = chromadb.EphemeralClient()
        collection_name = str(uuid.uuid4())
        if collection_name in chroma_client.list_collections():
            chroma_client.delete_collection(collection_name)
        chroma_collection = chroma_client.create_collection(collection_name)

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context)
        return index
    except Exception as e:
        print("an error occurred:", e)


def init_query_engine(index):
    template = (
        "Imagine you are an expert software engineer named Tigo Mamba, with access to all current and relevant software and API design, "
        "as well as the ability to search Postman collections and make cURL requests if necessary.\n"
        "Here is some context related to the query:\n"
        "<pre><code>-----------------------------------------\n"
        "{context_str}\n"
        "-----------------------------------------\n"
        "</code></pre>\n"
        "Considering the above information, please respond in Spanish.\n"
        "If the user requests Node.js code in any form, please generate the appropriate Node.js code. Include the code within `<pre><code class=\"language-javascript\">...</code></pre>` tags.\n"
        "If the user requests a cURL command in any form, please search for the relevant endpoint and generate the appropriate cURL command. Include the cURL command within `<pre><code class=\"language-bash\">...</code></pre>` tags.\n"
        "If the user requests a list, please return the information in an HTML list format, for example:\n"
        "<ul>\n"
        "  <li>Item 1</li>\n"
        "  <li>Item 2</li>\n"
        "  <li>Item 3</li>\n"
        "</ul>\n"
        "If no relevant information is found or context is empty, please respond carefully that no information was found.\n"
        "Ensure the response is accurate, helpful, and safe. Avoid providing sensitive or confidential information.\n"
        "Do not make assumptions or provide speculative information. Only provide information based on the context given.\n"
        "Understand and search all endpoints or collections that are related to the user's query.\n"
        "Structure the response in HTML format for better presentation, including appropriate HTML tags for headings, paragraphs, lists, and code blocks. Use `<br/>` for line breaks.\n"
        "Question: {query_str}\n\n"
        "Answer:"
    )
    qa_template = PromptTemplate(template)
    query_engine = index.as_query_engine(
        text_qa_template=qa_template, similarity_top_k=3)

    return query_engine


@r.post("/")
async def chat(
    request: Request,
    query: Query
):
    try:
        # llm = Ollama(model="mistral:latest", request_timeout=300.0)
        llm = Groq(model="llama-3.1-70b-versatile",
                   api_key="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")
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        Settings.llm = llm
        Settings.embed_model = embed_model

        index = init_index()
        query_engine = init_query_engine(index)

        response = query_engine.query(query.query)
        return response
    except Exception as e:
        logger.exception("Error in recommended engine", exc_info=True)
        logger.info(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in recommended: {e}",
        ) from e
