from openai import OpenAI

from article_analyzer.answer import LLM_MODEL

REWRITE_PROMPT = """You turn questions about 21 cm cosmology into search queries.
Rewrite the question using the technical terms a research paper on this topic would use.
Keep the original meaning and do not add new assumptions.
Do not answer the question. Return only the rewritten question as one sentence."""


def rewrite(client: OpenAI, question: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content.strip()