import { ChromaClient } from "chromadb";

const client = new ChromaClient({
    path: "http://localhost:8000" 
   });

export async function listCollections() {
    // list all collections
    const result = await client.listCollections();
    console.log(result);
  }
  
  export async function deleteCollection(collectionName) {
    const result = await client.deleteCollection({ name: `${collectionName}` });
    console.log(result);
  }