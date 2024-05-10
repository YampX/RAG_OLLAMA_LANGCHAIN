import { loadTextFromFile, saveEmbeddings, questionsChroma, traducer } from "./langchainModule.js";
import { deleteCollection, listCollections } from "./chromadbModule.js";

// Cargar el texto desde un archivo
const filePath = "./data/data.txt";
const text = await loadTextFromFile(filePath);

/*************************************************************/

// const retrievalChain = await saveEmbeddings(text);

// const result = await retrievalChain.invoke({
//     input: "Cual era la unica felicidad del personaje?",
//   });

//   console.log(result.answer);

/*************************************************************/

const retrievalChain = await questionsChroma();

const result = await retrievalChain.invoke({
  input: "Cual era la unica felicidad del personaje?",
});

console.log(result.answer);

/*************************************************************/

// await listCollections();
// await deleteCollection('a-test-collection')
