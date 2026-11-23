import os
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )


def split_transcript(transcript: str) -> list[str]:
    """
    Split long transcripts into manageable chunks.
    Larger chunks reduce the number of Gemini API calls.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=30000,
        chunk_overlap=1000
    )

    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()

    # Summarize individual chunks
    map_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Summarize this portion of a video transcript concisely "
            "and capture the important concepts."
        ),
        (
            "human",
            "{text}"
        ),
    ])

    map_chain = map_prompt | llm | StrOutputParser()

    chunk_summaries = []

    for chunk in split_transcript(transcript):
        summary = map_chain.invoke({
            "text": chunk
        })

        chunk_summaries.append(summary)


    # Combine all chunk summaries
    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert video summarizer. "
            "Combine these partial summaries into one professional summary "
            "using clear bullet points. Include important concepts, "
            "examples, and key takeaways."
        ),
        (
            "human",
            "{text}"
        ),
    ])

    combined_chain = combined_prompt | llm | StrOutputParser()

    final_summary = combined_chain.invoke({
        "text": "\n\n".join(chunk_summaries)
    })

    return final_summary



def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Based on the transcript, generate a short professional title "
            "with maximum 8 words. Return only the title."
        ),
        (
            "human",
            "{text}"
        ),
    ])

    title_chain = title_prompt | llm | StrOutputParser()

    title = title_chain.invoke({
        "text": transcript[:2000]
    })

    return title