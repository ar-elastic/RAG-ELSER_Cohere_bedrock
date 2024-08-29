# index.py

from getpass import getpass
from urllib.request import urlopen
from langchain_elasticsearch import ElasticsearchStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.chains import RetrievalQA
import boto3
import json

# Initialize Amazon Bedrock client using the following code

default_region = "us-west-2"
AWS_ACCESS_KEY = getpass("AWS Access key: ")
AWS_SECRET_KEY = getpass("AWS Secret key: ")
AWS_REGION = input(f"AWS Region [default: {default_region}]: ") or default_region

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

# Connect to Elasticsearch

CLOUD_ID = getpass("Elastic deployment Cloud ID: ")
CLOUD_API_KEY = getpass("Elastic API Key")

vector_store = ElasticsearchStore(
   es_cloud_id=CLOUD_ID,
   es_api_key=CLOUD_API_KEY,
   index_name= "workplace_index",
   strategy=ElasticsearchStore.SparseVectorRetrievalStrategy(model_id=".elser_model_2_linux-x86_64")
)

# Download the dataset, deserialize the document and split the document into passages. We’ll chunk the documents into passages

url = "https://raw.githubusercontent.com/elastic/elasticsearch-labs/main/example-apps/chatbot-rag-app/data/data.json"

response = urlopen(url)

workplace_docs = json.loads(response.read())

metadata = []
content = []

for doc in workplace_docs:
  content.append(doc["content"])
  metadata.append({
      "name": doc["name"],
      "summary": doc["summary"],
      "rolePermissions":doc["rolePermissions"]
})

text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=400)
docs = text_splitter.create_documents(content, metadatas=metadata)

# Index data to Elasticsearch using ElasticsearchStore.from_documents.

documents = vector_store.from_documents(
    docs,
    es_cloud_id=CLOUD_ID,
    es_api_key=CLOUD_API_KEY,
    index_name="workplace_index",
    strategy=ElasticsearchStore.SparseVectorRetrievalStrategy(model_id=".elser_model_2_linux-x86_64")
)

default_model_id = "cohere.command-text-v14"
AWS_MODEL_ID = input(f"AWS model [default: {default_model_id}]: ") or default_model_id
llm = Bedrock(client=bedrock_client, model_id=AWS_MODEL_ID)

def create_bedrock_llm(bedrock_client, model_version_id):
    bedrock_lIm=Bedrock(
        model_id=model_version_id,
        client=bedrock_client,
        model_kwargs={'temperature': 0}
        )
    return bedrock_lIm

retriever = vector_store.as_retriever()

qa = RetrievalQA.from_llm(llm=llm, retriever=retriever, return_source_documents=True)

questions = [
    "What is the nasa sales team?",
    "What is our work from home policy?",
    "Does the company own my personal project?",
    "What job openings do we have?",
    "How does compensation work?",
]
question = questions[1]
print(f"Question: {question}")

ans = qa({"query": question})

print("\033[92m ---- Answer ---- \033[0m")
print(ans["result"] + "\n")
print("\033[94m ---- Sources ---- \033[0m")
for doc in ans["source_documents"]:
    print("Name: " + doc.metadata["name"])
    print("Content: " + doc.page_content)
    print("-------")


