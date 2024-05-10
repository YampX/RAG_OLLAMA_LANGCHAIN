import { TextLoader } from "langchain/document_loaders/fs/text";
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChatGroq } from "@langchain/groq";
import { Ollama } from "@langchain/community/llms/ollama";

const genModel = new Ollama({
  baseUrl: "http://localhost:11434", // Default value
  model: "llama3:8b",
});

const groqModel = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
});

const embeddings = new OllamaEmbeddings({
  model: "mxbai-embed-large",
  maxConcurrency: 5,
});

const chatModel = new ChatOllama({
  baseUrl: "http://localhost:11434", // Default value
  model: "llama3:8b",
});

const prompt =
  ChatPromptTemplate.fromTemplate(`Based solely on the context provided, answer the following question in Spanish:

<context>
{context}
</context>

Question: {input}`);

// Función para cargar un texto desde un archivo
export async function loadTextFromFile(filePath) {
  const loader = new TextLoader(filePath);
  const docs = await loader.load();
  return docs;
}

export async function saveEmbeddings(docs) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 800,
    chunkOverlap: 100,
  });
  
  const splitDocs = await splitter.splitDocuments(docs);

  const vectorStore = await Chroma.fromDocuments(splitDocs, embeddings, {
    collectionName: "a-test-collection",
    url: "http://localhost:8000",
  });

  const documentChain = await createStuffDocumentsChain({
    llm: groqModel,
    prompt,
  });
  const retriever = vectorStore.asRetriever();
  const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever,
  });
  return retrievalChain;
}

export async function questionsChroma() {
  const vectorStore = new Chroma(embeddings, {
    collectionName: "a-test-collection",
  });

  const documentChain = await createStuffDocumentsChain({
    llm: groqModel,
    prompt,
  });
  const retriever = vectorStore.asRetriever();
  const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever,
  });

  return retrievalChain;
}

export async function traducer(input, language) {
  const promptTraducer = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are an expert translator.`
    ],
    ["human", `Response translate "{input}" into {language}.`],
  ]);
  const chain = promptTraducer.pipe(groqModel);

  const result = await chain.invoke({
    input,
    language,
  });

  
  return result.lc_kwargs.content;
  // const res = await groqModel.invoke(`Response only ${text} in Spanish`);
  // console.log(res.lc_kwargs.content);
}
