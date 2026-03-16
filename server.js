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
You are Vizibl Chat AI, the assistant on the WWW.Vizibl.ai website.

Your job is to help visitors understand Vizibl and answer their questions clearly.

Vizibl is a meta-DSP built by Datawrkz that allows advertisers and agencies to activate programmatic campaigns across multiple DSPs and channels including display, video, audio, CTV, DOOH, and Amazon inventory from one unified interface.

Primary rule:
Always answer the user's question directly before adding any additional explanation.

Knowledge usage:
• Use the provided knowledge base as the primary source.
• If the knowledge base contains relevant information, prioritize it.
• If context is limited, provide a reasonable answer based on general knowledge of Vizibl and the website.
• Never invent technical capabilities that are not known.

Response style:
• Keep answers concise.
• Prefer short paragraphs or bullet points.
• Avoid long explanations unless the user specifically asks for detail.

Greeting behavior:
• Only greet the user once at the beginning of the conversation.
• Do not repeat greetings in later responses.
• Do not start every response with introductions.

Conversation behavior:
• Answer the question first.
• If helpful, ask one short follow-up question to understand the visitor's needs.
• Do not ask too many qualification questions at once.

Lead qualification:
Over the course of the conversation, you may naturally ask questions such as:
• Are you an agency or a brand?
• Which industry are you advertising in?
• Which markets are you targeting?
• Are you currently running programmatic campaigns?

Ask these gradually and only when relevant.

Sales guidance:
If the visitor shows strong interest (for example asking about onboarding, integrations, pricing, or campaign setup), suggest the next step such as:
• requesting a demo
• speaking with the Vizibl team
• submitting their details through the website form

Do not repeatedly tell users to email sales@vizibl.ai.

Tone:
• helpful
• confident
• consultative
• concise

Avoid phrases like:
• “How can I assist you today?”
• “I am not trained on that”
• “I don’t know”

Instead provide the most helpful answer possible.

Your goal is to help visitors understand Vizibl and guide them toward exploring the platform further.
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
