import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Union

import discord
from discord import app_commands
from discord.ext import commands
import requests
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    genai = None
    types = None

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_GOOGLE = "gemini-2.0-flash-exp"
DEFAULT_MODEL_OPENAI = "gpt-4o"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Settings:
    platform: str
    model: str
    base_url: Optional[str]
    channel_history_limit: int
    reply_chain_limit: int
    enable_web_search: bool
    search_provider: str
    enable_google_grounding: bool
    persona: Optional[str]
    enable_citation_links: bool
    embed_footer: str


def load_settings() -> Settings:
    platform = os.getenv("PLATFORM", "google").lower()
    
    if platform == "google":
        default_model = DEFAULT_MODEL_GOOGLE
    else:
        default_model = DEFAULT_MODEL_OPENAI
    
    model = os.getenv("MODEL", default_model)
    base_url = os.getenv("OPENAI_BASE_URL") if platform == "openai" else None
    channel_history_limit = int(os.getenv("CHANNEL_HISTORY_LIMIT", "15"))
    reply_chain_limit = int(os.getenv("REPLY_CHAIN_LIMIT", "15"))
    enable_web_search = env_bool("ENABLE_WEB_SEARCH", True)
    search_provider = os.getenv("SEARCH_PROVIDER", "tavily")
    enable_google_grounding = env_bool("ENABLE_GOOGLE_GROUNDING", True)
    persona = os.getenv("PERSONA")
    enable_citation_links = env_bool("ENABLE_CITATION_LINKS", False)
    embed_footer = os.getenv("EMBED_FOOTER", "[LMcord](https://github.com/lunae-f/LMcord), made with ❤️‍🔥 by Lunae. | MIT License")
    return Settings(
        platform=platform,
        model=model,
        base_url=base_url,
        channel_history_limit=channel_history_limit,
        reply_chain_limit=reply_chain_limit,
        enable_web_search=enable_web_search,
        search_provider=search_provider,
        enable_google_grounding=enable_google_grounding,
        persona=persona,
        enable_citation_links=enable_citation_links,
        embed_footer=embed_footer,
    )


SETTINGS = load_settings()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN is required")

# Initialize API clients based on platform
genai_client = None
openai_client = None

if SETTINGS.platform == "google":
    if not GOOGLE_AVAILABLE:
        raise RuntimeError("google-genai package is not installed. Install with: pip install google-genai")
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is required for Google platform")
    genai_client = genai.Client(api_key=GOOGLE_API_KEY)
elif SETTINGS.platform == "openai":
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package is not installed. Install with: pip install openai")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI platform")
    openai_client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=SETTINGS.base_url or DEFAULT_BASE_URL
    )
else:
    raise RuntimeError(f"Unsupported platform: {SETTINGS.platform}. Use 'google' or 'openai'")

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(command_prefix="!", intents=intents)
client = bot  # For backward compatibility

TOKENS_FILE = "data/tokens_monthly.json"


def ensure_data_dir() -> None:
    """Ensure data directory exists."""
    os.makedirs("data", exist_ok=True)


def load_monthly_tokens() -> dict:
    """Load monthly token counts from file."""
    ensure_data_dir()
    if not os.path.exists(TOKENS_FILE):
        return {"month": "", "input": 0, "output": 0}
    try:
        with open(TOKENS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"month": "", "input": 0, "output": 0}


def save_monthly_tokens(data: dict) -> None:
    """Save monthly token counts to file."""
    ensure_data_dir()
    try:
        with open(TOKENS_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def update_monthly_tokens(input_tokens: int, output_tokens: int) -> dict:
    """Update monthly token counts and reset if month changed."""
    current_month = datetime.now().strftime("%Y-%m")
    tokens_data = load_monthly_tokens()
    
    # Reset if month changed
    if tokens_data["month"] != current_month:
        tokens_data = {"month": current_month, "input": 0, "output": 0}
    
    tokens_data["input"] += input_tokens
    tokens_data["output"] += output_tokens
    save_monthly_tokens(tokens_data)
    return tokens_data


async def update_bot_status(tokens_data: dict) -> None:
    """Update bot status with monthly token counts."""
    if not client.user:
        return
    activity = discord.Activity(
        type=discord.ActivityType.playing,
        name=f"💎 {tokens_data['input']:,} IN / {tokens_data['output']:,} OUT in this month."
    )
    await client.change_presence(activity=activity)


def build_system_prompt() -> str:
    # If persona is set, use only the persona
    if SETTINGS.persona:
        return SETTINGS.persona
    
    # Otherwise, use default assistant prompt
    base_prompt = "あなたはDiscordのアシスタントです。日本語で、丁寧かつ正確さ重視で回答してください。"
    if SETTINGS.enable_web_search or SETTINGS.enable_google_grounding:
        base_prompt += "\n必要に応じてウェブ検索ツールを使用し、最新の情報を取得してください。"
    base_prompt += "\n推測を避け、わからない場合はわかりないと伝えてください。"
    
    return base_prompt


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
        f"- モデル: {SETTINGS.model}",
    ]
    if SETTINGS.base_url:
        settings_lines.append(f"- エンドポイント: {SETTINGS.base_url}")
    
    # Google Grounding表示（google使用時のみ）
    if SETTINGS.platform == "google":
        grounding_status = "有効" if SETTINGS.enable_google_grounding else "無効"
        settings_lines.append(f"- Google Grounding with Google Search: {grounding_status}")
    
    settings_lines.extend([
        f"- チャンネル履歴参照: {SETTINGS.channel_history_limit}件",
        f"- 返信参照チェーン: {SETTINGS.reply_chain_limit}件",
        f"- Web検索: {'有効' if SETTINGS.enable_web_search and TAVILY_API_KEY else '無効'}",
        f"- 検索プロバイダ: {SETTINGS.search_provider}",
        f"- 出典リンク表示: {'有効' if SETTINGS.enable_citation_links else '無効'}",
    ])
    
    # ペルソナを全文表示
    if SETTINGS.persona:
        settings_lines.extend([
            "",
            "【ペルソナ設定】",
            SETTINGS.persona,
        ])
    
    settings_lines.extend([
        "",
        "使い方:",
        "- @ボット名 質問内容 の形式で呼び出してください。",
        "- 必要に応じてボットがウェブ検索を使います（検索した場合は参照URLを表示）。",
    ])
    return "\n".join(settings_lines)


def split_discord_message(text: str, limit: int = 2000) -> List[str]:
    if len(text) <= limit:
        return [text]
    chunks: List[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit)
        if split_at == -1 or split_at < limit * 0.3:
            split_at = limit
        chunk = remaining[:split_at].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


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
    if results and SETTINGS.enable_citation_links:
        lines.append("参照:")
        for item in results:
            title = item.get("title") or "(no title)"
            url = item.get("url") or ""
            lines.append(f"[{title}]({url})")
    return "\n".join(lines)


def build_tool_spec_google() -> List[types.Tool]:
    tools = []
    
    # Add Grounding with Google Search
    if SETTINGS.enable_google_grounding:
        tools.append(types.Tool(google_search=types.GoogleSearch()))
    
    # Add Tavily search as fallback
    if SETTINGS.enable_web_search and TAVILY_API_KEY and not SETTINGS.enable_google_grounding:
        tavily_search_declaration = types.FunctionDeclaration(
            name="tavily_search",
            description="必要に応じてウェブ検索を行い、要約と参照URLを返します。",
            parameters={
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
        )
        tools.append(types.Tool(function_declarations=[tavily_search_declaration]))
    
    return tools


def build_tool_spec_openai() -> Optional[List[dict]]:
    if not (SETTINGS.enable_web_search and TAVILY_API_KEY):
        return None
    
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


async def build_messages(message: discord.Message, prompt: str) -> Union[List[types.Content], List[dict]]:
    if SETTINGS.platform == "google":
        return await build_messages_google(message, prompt)
    else:
        return await build_messages_openai(message, prompt)


async def build_messages_google(message: discord.Message, prompt: str) -> List[types.Content]:
    contents: List[types.Content] = []
    
    # System instruction part
    system_parts = [build_system_prompt()]
    
    # Reply chain
    reply_chain = await fetch_reply_chain(message, SETTINGS.reply_chain_limit)
    if reply_chain:
        lines = ["返信チェーン参照:"]
        for m in reply_chain:
            content = format_message_content(m)
            if content:
                lines.append(f"- {m.author.display_name}: {content}")
        system_parts.append("\n".join(lines))
    
    # Channel history
    history = [
        m
        async for m in message.channel.history(
            limit=SETTINGS.channel_history_limit, before=message
        )
    ]
    history.reverse()
    
    # Combine system prompt with context
    combined_system = "\n\n".join(system_parts)
    contents.append(types.Content(role="user", parts=[types.Part(text=combined_system)]))
    contents.append(types.Content(role="model", parts=[types.Part(text="了解しました。")]))
    
    # Add history
    for m in history:
        content_text = format_message_content(m)
        if content_text:
            role = "model" if m.author == client.user else "user"
            author_prefix = "" if m.author == client.user else f"{m.author.display_name}: "
            contents.append(types.Content(role=role, parts=[types.Part(text=f"{author_prefix}{content_text}")]))
    
    # Add current prompt
    contents.append(types.Content(role="user", parts=[types.Part(text=f"{message.author.display_name}: {prompt}")]))
    
    return contents


async def build_messages_openai(message: discord.Message, prompt: str) -> List[dict]:
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


async def run_llm(messages: Union[List[types.Content], List[dict]]) -> tuple[str, int, int]:
    if SETTINGS.platform == "google":
        return await run_llm_google(messages)
    else:
        return await run_llm_openai(messages)


async def run_llm_google(contents: List[types.Content]) -> tuple[str, int, int]:
    tools = build_tool_spec_google()
    
    response = genai_client.models.generate_content(
        model=SETTINGS.model,
        contents=contents,
        config=types.GenerateContentConfig(
            tools=tools if tools else None,
        ),
    )
    
    # Get usage data
    input_tokens = 0
    output_tokens = 0
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        input_tokens = response.usage_metadata.prompt_token_count or 0
        output_tokens = response.usage_metadata.candidates_token_count or 0
    
    # Extract grounding citations if available
    grounding_citations = ""
    if SETTINGS.enable_citation_links and response.candidates and response.candidates[0].grounding_metadata:
        metadata = response.candidates[0].grounding_metadata
        if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
            citations = []
            for chunk in metadata.grounding_chunks:
                if hasattr(chunk, "web") and chunk.web:
                    citations.append(f"[{chunk.web.title}]({chunk.web.uri})")
            if citations:
                grounding_citations = "\n\n参照: " + " | ".join(citations)
    
    # Check if there are function calls (for Tavily search fallback)
    if response.candidates and response.candidates[0].content.parts:
        first_part = response.candidates[0].content.parts[0]
        if hasattr(first_part, 'function_call') and first_part.function_call:
            # Process function calls
            contents.append(response.candidates[0].content)
            
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    if fc.name == "tavily_search":
                        query = fc.args.get("query", "")
                        max_results = int(fc.args.get("max_results", 5))
                        tool_result = await tavily_search(query, max_results=max_results)
                        
                        # Add function response
                        contents.append(
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part(
                                        function_response=types.FunctionResponse(
                                            name="tavily_search",
                                            response={"result": tool_result}
                                        )
                                    )
                                ]
                            )
                        )
            
            # Get follow-up response
            follow_up = genai_client.models.generate_content(
                model=SETTINGS.model,
                contents=contents,
            )
            # Update token counts for follow-up
            if hasattr(follow_up, 'usage_metadata') and follow_up.usage_metadata:
                input_tokens += follow_up.usage_metadata.prompt_token_count or 0
                output_tokens += follow_up.usage_metadata.candidates_token_count or 0
            return (follow_up.text or "") + grounding_citations, input_tokens, output_tokens
    
    return (response.text or "") + grounding_citations, input_tokens, output_tokens


async def run_llm_openai(messages: List[dict]) -> tuple[str, int, int]:
    tools = build_tool_spec_openai()
    
    response = openai_client.chat.completions.create(
        model=SETTINGS.model,
        messages=messages,
        tools=tools,
        tool_choice="auto" if tools else None,
    )
    
    # Get usage data
    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0
    
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
        # Accumulate token counts from follow-up
        if follow_up.usage:
            input_tokens += follow_up.usage.prompt_tokens
            output_tokens += follow_up.usage.completion_tokens
        return follow_up.choices[0].message.content or "", input_tokens, output_tokens

    return message.content or "", input_tokens, output_tokens


@client.event
async def on_ready() -> None:
    print(f"Logged in as {client.user}")
    # Set initial status
    tokens_data = load_monthly_tokens()
    await update_bot_status(tokens_data)
    
    # Sync slash commands
    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} command(s)")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")


@bot.tree.command(name="llmcord", description="LMcordの設定とヘルプを表示")
@app_commands.describe(action="実行するアクション")
@app_commands.choices(action=[
    app_commands.Choice(name="help", value="help")
])
async def llmcord_command(interaction: discord.Interaction, action: str):
    """LMcord slash command handler."""
    if action == "help":
        help_text = build_help_message()
        await interaction.response.send_message(help_text, ephemeral=True)


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

    # Log incoming message
    logger.info(f"📨 Message from {message.author.name} in #{message.channel.name}: {prompt[:100]}...")

    await message.channel.typing()
    try:
        msg_data = await build_messages(message, prompt)
        reply, input_tokens, output_tokens = await run_llm(msg_data)
        if not reply:
            reply = "申し訳ありません、応答の生成に失敗しました。"
        
        # Log response
        logger.info(f"✅ Responded to {message.author.name} ({input_tokens} in, {output_tokens} out)")
        
        chunks = split_discord_message(reply)
        for i, chunk in enumerate(chunks):
            if i == 0:
                await message.channel.send(chunk, reference=message)
            else:
                await message.channel.send(chunk)
        
        # Send token count as an embed
        embed = discord.Embed(
            color=discord.Color.greyple(),
            description=f"💎 Tokens: {input_tokens}/{output_tokens}\n{SETTINGS.embed_footer}"
        )
        await message.channel.send(embed=embed)
        
        # Update monthly token counts and bot status
        tokens_data = update_monthly_tokens(input_tokens, output_tokens)
        await update_bot_status(tokens_data)
    except Exception as exc:
        # Log error
        logger.error(f"❌ Error processing message from {message.author.name}: {exc}", exc_info=True)
        await message.channel.send(
            f"エラーが発生しました: {exc}", reference=message
        )


async def main() -> None:
    await bot.start(DISCORD_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
