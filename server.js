import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";
import { createClient } from "@supabase/supabase-js";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// ✅ OpenAI Setup
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// ✅ Supabase Setup
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

// ✅ Chat Endpoint (RAG Enabled)
app.post("/chat", async (req, res) => {
  try {
    const userMessage = req.body.message;

    if (!userMessage) {
      return res.status(400).json({ error: "Message is required." });
    }

    // 1️⃣ Create embedding for user query
    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: userMessage,
    });

    const queryEmbedding = embeddingResponse.data[0].embedding;

    // 2️⃣ Retrieve similar documents from Supabase
    const { data: documents, error } = await supabase.rpc(
      "match_documents",
      {
        query_embedding: queryEmbedding,
        match_threshold: 0.6,
        match_count: 6,
      }
    );

    if (error) {
      console.error("Supabase RPC Error:", error);
      return res.status(500).json({ error: "Vector search failed." });
    }

    // 3️⃣ Format context for GPT (include category + title)
    const context =
      documents && documents.length > 0
        ? documents
            .map(
              (doc) =>
                `[${doc.category}] ${doc.title}\n${doc.content}`
            )
            .join("\n\n")
        : "";

    // 4️⃣ Generate response using GPT
    const completion = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: `
You are Vizibl Chat AI.

You help website visitors understand Vizibl — a meta-DSP for programmatic advertising.

Always assume the user is asking about Vizibl unless clearly stated otherwise.

Guidelines:
- Use retrieved context when available.
- If context is limited, answer based on general knowledge of Vizibl.
- Never say you are unsure unless absolutely no relevant information exists.
- Keep answers concise but informative.
- Avoid overly long responses.
- Be confident and consultative.
- When relevant, suggest requesting a demo or contacting sales@vizibl.ai.
- If a question is broad like "tell me about the product", provide a structured overview with 3–5 bullet points.
- Never fabricate technical claims not in context.

Context:
${context}
`
        },
        {
          role: "user",
          content: userMessage,
        },
      ],
      temperature: 0.3,
    });

    const reply = completion.choices[0].message.content;

    res.json({ reply });

  } catch (error) {
    console.error("Server Error:", error);
    res.status(500).json({ error: "Something went wrong." });
  }
});

// ✅ Start Server
app.listen(3000, () => {
  console.log("🚀 Vizibl Chat AI Backend running on port 3000");
});