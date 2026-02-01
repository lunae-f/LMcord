import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import discord
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-3-flash-preview"


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Settings:
    platform: str
    base_url: str
    model: str
    channel_history_limit: int
    reply_chain_limit: int
    enable_web_search: bool
    search_provider: str


def load_settings() -> Settings:
    platform = os.getenv("OPENAI_PLATFORM", "google")
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    channel_history_limit = int(os.getenv("CHANNEL_HISTORY_LIMIT", "50"))
    reply_chain_limit = int(os.getenv("REPLY_CHAIN_LIMIT", "50"))
    enable_web_search = env_bool("ENABLE_WEB_SEARCH", True)
    search_provider = os.getenv("SEARCH_PROVIDER", "tavily")
    return Settings(
        platform=platform,
        base_url=base_url,
        model=model,
        channel_history_limit=channel_history_limit,
        reply_chain_limit=reply_chain_limit,
        enable_web_search=enable_web_search,
        search_provider=search_provider,
    )


SETTINGS = load_settings()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN is required")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=SETTINGS.base_url)

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

client = discord.Client(intents=intents)


def build_system_prompt() -> str:
    return (
        "あなたはDiscordのアシスタントです。日本語で、丁寧かつ正確さ重視で回答してください。"
        "必要に応じてウェブ検索ツールを使ってください。検索を使った場合、必ず参照URLを含めてください。"
        "推測は避け、分からない場合は分からないと伝えてください。"
    )


def format_message_content(message: discord.Message) -> str:
    parts = [message.content.strip()] if message.content else []
    if message.attachments:
        parts.extend(a.url for a in message.attachments)
    content = "\n".join(p for p in parts if p)
    return content


def message_to_chat(message: discord.Message) -> Optional[dict]:
    content = format_message_content(message)
    if not content:
        return None
    role = "assistant" if message.author == client.user else "user"
    author = message.author.display_name
    return {
        "role": role,
        "content": f"{author}: {content}",
    }


async def fetch_reply_chain(message: discord.Message, limit: int) -> List[discord.Message]:
    chain: List[discord.Message] = []
    current = message
    while len(chain) < limit and current.reference is not None:
        try:
            if current.reference.resolved:
                referenced = current.reference.resolved
            else:
                referenced = await current.channel.fetch_message(current.reference.message_id)
        except Exception:
            break
        if referenced is None:
            break
        chain.append(referenced)
        current = referenced
    return list(reversed(chain))


def build_help_message() -> str:
    settings_lines = [
        "現在の設定:",
        f"- プラットフォーム: {SETTINGS.platform}",
        f"- エンドポイント: {SETTINGS.base_url}",
        f"- モデル: {SETTINGS.model}",
        f"- チャンネル履歴参照: {SETTINGS.channel_history_limit}件",
        f"- 返信参照チェーン: {SETTINGS.reply_chain_limit}件",
        f"- Web検索: {'有効' if SETTINGS.enable_web_search and TAVILY_API_KEY else '無効'}",
        f"- 検索プロバイダ: {SETTINGS.search_provider}",
        "- スタイル: 日本語/丁寧/正確さ重視",
        "",
        "使い方:",
        "- @ボット名 質問内容 の形式で呼び出してください。",
        "- 必要に応じてボットがウェブ検索を使います（検索した場合は参照URLを表示）。",
    ]
    return "\n".join(settings_lines)


async def tavily_search(query: str, max_results: int = 5) -> str:
    if not TAVILY_API_KEY:
        return "Tavily APIキーが設定されていません。"
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": max_results,
            "include_answer": True,
            "include_raw_content": False,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    answer = data.get("answer") or ""
    results = data.get("results") or []
    lines = []
    if answer:
        lines.append(f"要約: {answer}")
    lines.append("参照URL:")
    for item in results:
        title = item.get("title") or "(no title)"
        url = item.get("url") or ""
        lines.append(f"- {title}: {url}")
    return "\n".join(lines)


def build_tool_spec() -> List[dict]:
    if not (SETTINGS.enable_web_search and TAVILY_API_KEY):
        return []
    return [
        {
            "type": "function",
            "function": {
                "name": "tavily_search",
                "description": "必要に応じてウェブ検索を行い、要約と参照URLを返します。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "検索クエリ",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "取得する結果数",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]


def extract_mention_prompt(content: str, bot_id: int) -> Optional[str]:
    pattern = rf"^<@!?{bot_id}>\s*"
    if not re.match(pattern, content.strip()):
        return None
    return re.sub(pattern, "", content.strip(), count=1).strip()


async def build_messages(message: discord.Message, prompt: str) -> List[dict]:
    messages: List[dict] = [{"role": "system", "content": build_system_prompt()}]

    reply_chain = await fetch_reply_chain(message, SETTINGS.reply_chain_limit)
    if reply_chain:
        lines = ["返信チェーン参照:"]
        for m in reply_chain:
            content = format_message_content(m)
            if content:
                lines.append(f"- {m.author.display_name}: {content}")
        messages.append({"role": "system", "content": "\n".join(lines)})

    history = [
        m
        async for m in message.channel.history(
            limit=SETTINGS.channel_history_limit, before=message
        )
    ]
    history.reverse()
    for m in history:
        chat_message = message_to_chat(m)
        if chat_message:
            messages.append(chat_message)

    messages.append({"role": "user", "content": f"{message.author.display_name}: {prompt}"})
    return messages


async def run_llm(messages: List[dict]) -> str:
    tools = build_tool_spec()
    response = openai_client.chat.completions.create(
        model=SETTINGS.model,
        messages=messages,
        tools=tools or None,
        tool_choice="auto" if tools else None,
    )
    message = response.choices[0].message

    if message.tool_calls:
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ],
        })
        for tool_call in message.tool_calls:
            if tool_call.function.name == "tavily_search":
                args = json.loads(tool_call.function.arguments or "{}")
                query = args.get("query", "")
                max_results = int(args.get("max_results", 5))
                tool_result = await tavily_search(query, max_results=max_results)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )
        follow_up = openai_client.chat.completions.create(
            model=SETTINGS.model,
            messages=messages,
        )
        return follow_up.choices[0].message.content or ""

    return message.content or ""


@client.event
async def on_ready() -> None:
    print(f"Logged in as {client.user}")


@client.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    if not client.user:
        return

    prompt = extract_mention_prompt(message.content, client.user.id)
    if prompt is None:
        return

    if not prompt:
        await message.channel.send(build_help_message())
        return

    await message.channel.typing()
    try:
        messages = await build_messages(message, prompt)
        reply = await run_llm(messages)
        if not reply:
            reply = "申し訳ありません、応答の生成に失敗しました。"
        await message.channel.send(reply, reference=message)
    except Exception as exc:
        await message.channel.send(
            f"エラーが発生しました: {exc}", reference=message
        )


async def main() -> None:
    await client.start(DISCORD_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
