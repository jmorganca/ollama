import { Artwork, RawArtwork } from './types';
import { HuggingFaceTransformersEmbeddings } from 'langchain/embeddings/hf_transformers';
import { Chroma } from "langchain/vectorstores/chroma";
import { Document } from "langchain/document";
import { ChromaClient } from "chromadb";
const numberOfArtworks = 10;

// list of artists we are going to pull from the API
const artists = ["van Gogh", "Renoir", "Monet", "Picasso"]

const generateSource = async () => {
  // Delete the existing vector store so that we don't get duplicate documents
  await new ChromaClient().deleteCollection({
    name: "artcollection",
  });
  const allartworkdocs = await getArt(artists);

  // Create the vector store
  const vectorStore = await Chroma.fromDocuments(allartworkdocs, embedding, { collectionName: "artcollection" });
  console.log(`Created vector store with ${await vectorStore.collection?.count()} documents`);
}

const getArt = async (artists: string[]) => {
  const artworks: Artwork[] = [];
  const artistsWorkIds: number[] = []

  for (const artist of artists) {
    // First get the ids of the works by each artist
    const thisIds = await fetchArtistWorkIds(artist);
    console.log(`Fetching ${artist}`);
    await (new Promise(r => setTimeout(r, 1000)));
    artistsWorkIds.push(...thisIds);
  };
  // now get the actual artwork
  const artwork = await fetchArtwork(artistsWorkIds);
  return artwork
}

const fetchArtistWorkIds = async (artist: string): Promise<number[]> => {
  const artistURL = `https://api.artic.edu/api/v1/artworks/search?q=${artist}&limit=${numberOfArtworks}`;
  const response = await fetch(artistURL);
  const json = await response.json();
  const artistWorks: { id: number }[] = json.data;
  return artistWorks.map((work) => work.id);
}
const embedding = new HuggingFaceTransformersEmbeddings({
  modelName: "Xenova/all-MiniLM-L6-v2",
});

//Turns out there are some weird characters in the descriptions
const sanitize = (badstring: string): string => {
  let goodstring = " ";
  if (badstring !== null) {
    goodstring = badstring
      .replace(/<\s*a\s+[^>]*href\s*=\s*[\"']?([^\"' >]+)[\"' >]>/gm, "")
      .replace(/<\/a>/gm, "")
      .replace(/<\/?em>/gm, "")
      .replace(/[\u2018\u2019]/gm, "")
      .replace(/[\u201C\u201D]/gm, "")
      .replace(/[\u2013\u2014]/gm, "-")
      .replace(/[\u2026]/gm, "...")
      .replace(/[\u00A0]/gm, " ")
      .replace(/[\u00AD]/gm, "-")
      .replace(/[\u00B0]/gm, " degrees ")
      .replace(/[\u00B1]/gm, " plus or minus ")
      .replace(/[\u00B2]/gm, " squared ")
      .replace(/[\u00B3]/gm, " cubed ")
      .replace(/[\u00B4]/gm, "'")
      .replace(/[\u00B5]/gm, " micro ")
      .replace(/[\u00B6]/gm, " paragraph ")
      .replace(/[\u00B7]/gm, " dot ")
      .replace(/[\u00B8]/gm, ",")
      .replace(/[\u00B9]/gm, " first ")
      .replace(/[\u00BA]/gm, " degrees ")
      .replace(/[\u00BB]/gm, ">>")
      .replace(/[\u00BC]/gm, " 1/4 ")
      .replace(/[\u00BD]/gm, " 1/2 ")
      .replace(/[\uFB01]/gm, "fi")
      .replace(/[\uFB02]/gm, "fl")
      .replace(/[\uFB03]/gm, "ffi")
      .replace(/[\uFB04]/gm, "ffl")
      .replace(/[\uFB05]/gm, "ft")
      .replace(/[\uFB06\uFB07\uFB08]/gm, "st")
      .replace(/[\u00D7]/gm, "x")
      .replace(/[\u00E8\u00E9]/gm, "e")
      .replace(/[\u00F1]/gm, "n")
      .replace(/[\u00F6]/gm, "o")
      .replace(/[\u00F8]/gm, "o")
      .replace(/[\u00FC]/gm, "u")
      .replace(/[\u00FF]/gm, "y")
      .replace(/[\u0101\u0103\u00E0]/gm, "a")
      .replace(/[\u00C9]/gm, "E")
      .replace(/<p>/gm, "")
      .replace(/<\/p>/gm, "")
      .replace(/\n/gm, "");
  };
  return goodstring;
}

const fetchArtwork = async (workids: number[]) => {
  const docsarray = [];
  const artworks: Artwork[] = [];

  for await (const workid of workids) {
    const artworkURL = `https://api.artic.edu/api/v1/artworks/${workid}`;
    const response = await fetch(artworkURL);
    const json = await response.json();
    const artworkraw: RawArtwork = await json.data as RawArtwork;
    const description = sanitize(artworkraw.description)
    if (description !== " ") {
      const doc = new Document({
        pageContent: description,
        metadata: {
          title: sanitize(artworkraw.title),
          date: artworkraw.date_end,
          artistName:  artworkraw.artist_title,
        }
      });
      docsarray.push(doc);
      console.log("------------------")
      console.log(`${artworkraw.title} - ${artworkraw.artist_title}`);
    }
  }

  return docsarray;
}

generateSource();
