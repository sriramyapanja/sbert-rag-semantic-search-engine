import os
import json
import faiss
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, Response, request
from flasgger import Swagger
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
Swagger(app)


class SearchRagApi:
    """RAG search engine API"""

    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="conversational",
            temperature=0.1,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        self.llm_model = ChatHuggingFace(llm=llm)

    def load_corpus_datasource(self):
        """Load corpus data source"""
        df_contracted_services = pd.read_csv(
            os.path.join(BASE_DIR, "data", "contracted_services.csv")
        )
        list_documents = df_contracted_services['contracted_services'].to_list()
        return list_documents

    def generate_corpus_embeddings(self, list_documents):
        """Convert documents to vector embeddings"""
        documents_embeddings = self.embedding_model.encode(
            list_documents,
            convert_to_numpy=True
        )
        return documents_embeddings

    def generate_query_embeddings(self, text_query):
        """Generate embeddings from query"""
        query_embedding = self.embedding_model.encode(
            [text_query],
            convert_to_numpy=True
        )
        return query_embedding

    def retrieve_documents(self, list_documents, documents_embeddings, query_embedding, top_k=2):
        """Search documents in embeddings source"""
        dimension = documents_embeddings.shape[1]
        similar_documents_embeddings = faiss.IndexFlatL2(dimension)
        similar_documents_embeddings.add(documents_embeddings)
        distances, indexes = similar_documents_embeddings.search(query_embedding, top_k)
        list_results_found = [list_documents[i] for i in indexes[0]]
        return list_results_found

    def generate_augmented_response(self, text_query, list_retrieved_documents):
        """Generate augmented response"""
        chat_prompt_template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:"""
        prompt = ChatPromptTemplate.from_template(chat_prompt_template)
        rag_chain = (
            {
                "context": lambda x: list_retrieved_documents,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm_model
            | StrOutputParser()
        )
        result = rag_chain.invoke(text_query)
        return result

    def search_text(self, text_query):
        """Search RAG text"""
        list_documents = self.load_corpus_datasource()
        documents_embeddings = self.generate_corpus_embeddings(list_documents)
        query_embedding = self.generate_query_embeddings(text_query)
        list_retrieved_documents = self.retrieve_documents(
            list_documents,
            documents_embeddings,
            query_embedding
        )
        answer = self.generate_augmented_response(text_query, list_retrieved_documents)
        return answer

    @staticmethod
    @app.route('/service/search', methods=['POST'])
    def search_text_api():
        """Search using RAG - Endpoint to contextual search
        ---
        tags:
          - Contextual Search
        parameters:
          - name: body
            in: body
            required: true
        responses:
          200:
            description: Success to find records.
          500:
            description: Failure to search.
        """
        try:
            json_query = json.loads(request.data)
            searchRagApi = SearchRagApi()
            json_result = searchRagApi.search_text(json_query['sentence_query'])
            json_result = '{ "result" : "' + json_result + '"}'
            return Response(json_result, mimetype='application/json'), 200
        except Exception as e:
            print("FULL ERROR:", str(e))
            import traceback
            traceback.print_exc()
            return Response(
                '{"result": "Failure to search."}',
                mimetype='application/json'
            ), 500


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")