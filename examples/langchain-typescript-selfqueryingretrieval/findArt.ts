import { Chroma } from "langchain/vectorstores/chroma";
import { ChromaTranslator } from "langchain/retrievers/self_query/chroma";
import { Ollama } from "langchain/llms/ollama"
import { AttributeInfo } from "langchain/schema/query_constructor";
import { HuggingFaceTransformersEmbeddings } from "langchain/embeddings/hf_transformers";
import { SelfQueryRetriever } from "langchain/retrievers/self_query";

const modelName = "codellama";

// Define the attributes of the schema so that the model will know what to look for
const attributeInfo: AttributeInfo[] = [
  {
    name: "title",
    type: "string",
    description: "The title of the painting"
  },
  {
    name: "date",
    type: "integer",
    description: "The four digit year when the painting was created"
  },
  {
    name: "artistName",
    type: "string",
    description: "The first name and last name of the artist who created the painting. Always use the full name in the filter, even if it isn't included. If the query is 'van Gogh', the filter should be 'Vincent van Gogh'. Use Pierre-Auguste Renoir instead of just Renoir."
  }
]


// Define the model used to generate embeddings, these capture the context of the input data
const embeddings = new HuggingFaceTransformersEmbeddings({
  modelName: "Xenova/all-MiniLM-L6-v2",
});

// Run the model using Ollama
const llm = new Ollama({
  model: modelName
})

const documentContents = "Description of the art";

const findArt = async () => {
  // Load the saved vector store
  const vectorStore = await Chroma.fromExistingCollection(embeddings, {
    collectionName: "artcollection",
  });

  const retriever = SelfQueryRetriever.fromLLM({
    llm, vectorStore, documentContents, attributeInfo, verbose: false, useOriginalQuery: true, structuredQueryTranslator: new ChromaTranslator()
  });

  // Get the query from the command line
  const query = process.argv[2];

  try {
    const newquery = await retriever.getRelevantDocuments(query, [
      // You can add callbacks to the retriever to get information about the process. In this case, show the output 
      // query from the LLM used to retrieve the documents
      {
        handleLLMEnd(output) {
          console.log("This is the output from the LLM after it has come up with a filter")
          const llmEndOutput = output.generations[0][0].text.replace(/\\"/gm, "'").replace(/\n/gm, "")
          console.log(`output - ${JSON.stringify(llmEndOutput, null, 2)}`)
        }
      },
    ]);
    console.log(newquery);
  } catch (error) {
    console.log(`There was an error getting the values: ${error}`);
  }
}

findArt();