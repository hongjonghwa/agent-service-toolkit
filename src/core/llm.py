from functools import cache
from typing import TypeAlias

from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_community.chat_models import FakeListChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from core.settings import settings
from schema.models import (
    AllModelEnum,
    AnthropicModelName,
    AWSModelName,
    AzureOpenAIModelName,
    DeepseekModelName,
    FakeModelName,
    GoogleModelName,
    GroqModelName,
    OllamaModelName,
    OpenAICompatibleName,
    OpenAIModelName,
)

_MODEL_TABLE = {
    OpenAIModelName.GPT_4O_MINI: "gpt-4o-mini",
    OpenAIModelName.GPT_4O: "gpt-4o",
    OpenAICompatibleName.OPENAI_COMPATIBLE: settings.COMPATIBLE_MODEL,
    AzureOpenAIModelName.AZURE_GPT_4O_MINI: settings.AZURE_OPENAI_DEPLOYMENT_MAP.get(
        "gpt-4o-mini", ""
    ),
    AzureOpenAIModelName.AZURE_GPT_4O: settings.AZURE_OPENAI_DEPLOYMENT_MAP.get("gpt-4o", ""),
    DeepseekModelName.DEEPSEEK_CHAT: "deepseek-chat",
    # AnthropicModelName.HAIKU_3: "claude-3-haiku-20240307",
    AnthropicModelName.HAIKU_35: "claude-3-5-haiku-latest",
    # AnthropicModelName.SONNET_35: "claude-3-5-sonnet-latest",
    AnthropicModelName.SONNET_37: "claude-3-7-sonnet-latest",
    # GoogleModelName.GEMINI_15_FLASH: "gemini-1.5-flash",
    GoogleModelName.GEMINI_20_FLASH: "gemini-2.0-flash",
    GoogleModelName.GEMINI_25_PRO: "gemini-2.5-pro-exp-03-25",
    GroqModelName.LLAMA_31_8B: "llama-3.1-8b-instant",
    GroqModelName.LLAMA_33_70B: "llama-3.3-70b-versatile",
    GroqModelName.LLAMA_GUARD_3_8B: "llama-guard-3-8b",
    AWSModelName.BEDROCK_HAIKU: "anthropic.claude-3-5-haiku-20241022-v1:0",
    AWSModelName.BEDROCK_SONNET: "anthropic.claude-3-5-sonnet-20240620-v1:0",
    OllamaModelName.OLLAMA_GENERIC: "ollama",
    FakeModelName.FAKE: "fake",
}

ModelT: TypeAlias = (
    ChatOpenAI | ChatAnthropic | ChatGoogleGenerativeAI | ChatGroq | ChatBedrock | ChatOllama
)


class FakeToolModel(FakeListChatModel):
    def __init__(self, responses: list[str]):
        super().__init__(responses=responses)

    def bind_tools(self, tools):
        return self


@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    # NOTE: models with streaming=True will send tokens as they are generated
    # if the /stream endpoint is called with stream_tokens=True (the default)
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name in OpenAIModelName:
        return ChatOpenAI(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in OpenAICompatibleName:
        if not settings.COMPATIBLE_BASE_URL or not settings.COMPATIBLE_MODEL:
            raise ValueError("OpenAICompatible base url and endpoint must be configured")

        return ChatOpenAI(
            model=settings.COMPATIBLE_MODEL,
            temperature=0.5,
            streaming=True,
            openai_api_base=settings.COMPATIBLE_BASE_URL,
            openai_api_key=settings.COMPATIBLE_API_KEY,
        )
    if model_name in AzureOpenAIModelName:
        if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
            raise ValueError("Azure OpenAI API key and endpoint must be configured")

        return AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            deployment_name=api_model_name,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=0.5,
            streaming=True,
            timeout=60,
            max_retries=3,
        )
    if model_name in DeepseekModelName:
        return ChatOpenAI(
            model=api_model_name,
            temperature=0.5,
            streaming=True,
            openai_api_base="https://api.deepseek.com",
            openai_api_key=settings.DEEPSEEK_API_KEY,
        )
    if model_name in AnthropicModelName:
        return ChatAnthropic(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in GoogleModelName:
        return ChatGoogleGenerativeAI(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in GroqModelName:
        if model_name == GroqModelName.LLAMA_GUARD_3_8B:
            return ChatGroq(model=api_model_name, temperature=0.0)
        return ChatGroq(model=api_model_name, temperature=0.5)
    if model_name in AWSModelName:
        return ChatBedrock(model_id=api_model_name, temperature=0.5)
    if model_name in OllamaModelName:
        if settings.OLLAMA_BASE_URL:
            chat_ollama = ChatOllama(
                model=settings.OLLAMA_MODEL, temperature=0.5, base_url=settings.OLLAMA_BASE_URL
            )
        else:
            chat_ollama = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0.5)
        return chat_ollama
    if model_name in FakeModelName:
        return FakeToolModel(responses=["This is a test response from the fake model."])
