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

const groqModel = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
});

const genModel = new Ollama({
  baseUrl: "http://localhost:11434",
  model: "llama3:8b",
});

const chatModel = new ChatOllama({
  baseUrl: "http://localhost:11434",
  model: "llama3:8b",
});

const embeddings = new OllamaEmbeddings({
  model: "mxbai-embed-large",
  maxConcurrency: 5,
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

export async function saveEmbeddings(docs, collectionName) {
  try {
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
      separators: ["\n\n", "\n", " ", ""],
    });
    const splitDocs = await splitter.splitDocuments(docs);
    const vectorStore = await Chroma.fromDocuments(splitDocs, embeddings, {
      collectionName,
      url: "http://localhost:8000",
    });
    return vectorStore;
  } catch (error) {
    console.error('Error al agregar datos a ChromaDB:', error);
    throw error; // Propago el error para que sea manejado por el código que llama a esta función
  }
}

export async function questionsChroma() {
  const vectorStore = new Chroma(embeddings, {
    collectionName: "Test_Data1",
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

// export async function traducer(input, language) {
//   const promptTraducer = ChatPromptTemplate.fromMessages([
//     ["system", `You are an expert translator.`],
//     ["human", `Response translate "{input}" into {language}.`],
//   ]);
//   const chain = promptTraducer.pipe(groqModel);

//   const result = await chain.invoke({
//     input,
//     language,
//   });

//   return result.lc_kwargs.content;
//   // const res = await groqModel.invoke(`Response only ${text} in Spanish`);
//   // console.log(res.lc_kwargs.content);
// }
