import OpenAI from "openai";
import { createClient } from "@supabase/supabase-js";
import fs from "fs";
import dotenv from "dotenv";

dotenv.config();

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

const knowledge = JSON.parse(fs.readFileSync("knowledge.json"));

for (let item of knowledge) {
  const embedding = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: item.content
  });

  await supabase.from("documents").insert({
    content: item.content,
    embedding: embedding.data[0].embedding
  });
}

console.log("Knowledge uploaded.");