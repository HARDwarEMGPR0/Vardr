require('dotenv').config()
const OpenAI = require('openai')

const client = new OpenAI({
  baseURL: process.env.NVIDIA_NIM_BASE_URL,
  apiKey: process.env.NVIDIA_NIM_API_KEY,
})

async function run() {
  const response = await client.chat.completions.create({
    model: process.env.NVIDIA_NIM_MODEL,
    messages: [
      { role: "user", content: "Which number is larger, 9.11 or 9.8?" }
    ],
    temperature: 0.2,
  })

  console.log(response.choices[0].message.content)
}

run()
