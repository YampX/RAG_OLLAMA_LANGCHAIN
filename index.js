import { loadTextFromFile, saveEmbeddings, questionsChroma } from "./langchainModule.js";
import { deleteCollection, listCollections } from "./chromadbModule.js";

// Cargar el texto desde un archivo y guardarlo
// const filePath = "./data/data1.txt";
// const text = await loadTextFromFile(filePath);
// console.log(text);
// await saveEmbeddings(text, 'Test_Data1');

/*************************************************************/

const retrievalChain = await questionsChroma();
const result = await retrievalChain.invoke({
  input: "Quien es RustiK?",
});

console.log(result.answer);

/*************************************************************/

// await listCollections();
// await deleteCollection('Test_Data')
