import asyncio
import json
import logging
import math
import os
import re
import tempfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Union

import aiohttp
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
    channel_history_limit: int
    reply_chain_limit: int
    enable_web_search: bool
    search_provider: str
    enable_google_grounding: bool
    persona: Optional[str]
    enable_citation_links: bool
    embed_footer: str
    max_monthly_cost_usd: float
    max_requests_per_minute: int
    max_attachments_per_message: int
    max_attachment_size_mb: int
    usd_to_jpy_rate: float


def load_settings() -> Settings:
    channel_history_limit = int(os.getenv("CHANNEL_HISTORY_LIMIT", "15"))
    reply_chain_limit = int(os.getenv("REPLY_CHAIN_LIMIT", "15"))
    enable_web_search = env_bool("ENABLE_WEB_SEARCH", True)
    search_provider = os.getenv("SEARCH_PROVIDER", "tavily")
    enable_google_grounding = env_bool("ENABLE_GOOGLE_GROUNDING", True)
    persona = os.getenv("PERSONA")
    enable_citation_links = env_bool("ENABLE_CITATION_LINKS", False)
    embed_footer = os.getenv("EMBED_FOOTER", "[LMcord](https://github.com/lunae-f/LMcord), made with ❤️‍🔥 by Lunae. | MIT License")
    max_monthly_cost_usd = float(os.getenv("MAX_MONTHLY_COST_USD", "0"))
    max_requests_per_minute = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "10"))
    max_attachments_per_message = int(os.getenv("MAX_ATTACHMENTS_PER_MESSAGE", "10"))
    max_attachment_size_mb = int(os.getenv("MAX_ATTACHMENT_SIZE_MB", "10"))
    usd_to_jpy_rate = float(os.getenv("USD_TO_JPY_RATE", "150"))
    return Settings(
        channel_history_limit=channel_history_limit,
        reply_chain_limit=reply_chain_limit,
        enable_web_search=enable_web_search,
        search_provider=search_provider,
        enable_google_grounding=enable_google_grounding,
        persona=persona,
        enable_citation_links=enable_citation_links,
        embed_footer=embed_footer,
        max_monthly_cost_usd=max_monthly_cost_usd,
        max_requests_per_minute=max_requests_per_minute,
        max_attachments_per_message=max_attachments_per_message,
        max_attachment_size_mb=max_attachment_size_mb,
        usd_to_jpy_rate=usd_to_jpy_rate,
    )


SETTINGS = load_settings()

@dataclass
class Profile:
    name: str
    platform: str
    model: str
    api_key: str
    base_url: Optional[str]
    input_cost_per_1m: float
    output_cost_per_1m: float


PROFILES_FILE = "profiles.json"
PROFILES: dict[str, Profile] = {}
ACTIVE_PROFILE_NAME: Optional[str] = None
CHANNEL_PROFILE_OVERRIDES: dict[int, str] = {}


def resolve_env_value(value: Optional[str]) -> str:
    if not value:
        return ""
    match = re.fullmatch(r"\$\{([A-Z0-9_]+)\}", value)
    if match:
        return os.getenv(match.group(1), "")
    return value


def load_profiles() -> None:
    global PROFILES, ACTIVE_PROFILE_NAME
    if not os.path.exists(PROFILES_FILE):
        raise RuntimeError(f"{PROFILES_FILE} is required")
    with open(PROFILES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    profiles = data.get("profiles", [])
    active_name = data.get("active_profile")
    if not profiles:
        raise RuntimeError("profiles.json must contain at least one profile")
    loaded: dict[str, Profile] = {}
    for p in profiles:
        name = p.get("name")
        platform = p.get("platform")
        model = p.get("model")
        api_key = resolve_env_value(p.get("api_key"))
        base_url = resolve_env_value(p.get("base_url")) or None
        input_cost = float(p.get("input_cost_per_1m", 0))
        output_cost = float(p.get("output_cost_per_1m", 0))
        if not name or not platform or not model:
            raise RuntimeError("Each profile must include name, platform, model")
        if platform not in {"google", "openai"}:
            raise RuntimeError(f"Unsupported platform in profile '{name}': {platform}")
        if not api_key:
            logger.warning(f"⚠️ プロファイル '{name}' のAPIキーが設定されていません。このプロファイルは利用できません。")
        loaded[name] = Profile(
            name=name,
            platform=platform,
            model=model,
            api_key=api_key,
            base_url=base_url,
            input_cost_per_1m=input_cost,
            output_cost_per_1m=output_cost,
        )
    PROFILES = loaded
    ACTIVE_PROFILE_NAME = active_name or next(iter(loaded.keys()))
    if ACTIVE_PROFILE_NAME not in PROFILES:
        raise RuntimeError(f"active_profile '{ACTIVE_PROFILE_NAME}' not found in profiles.json")


def get_active_profile(channel_id: Optional[int] = None) -> Profile:
    if channel_id is not None:
        override = CHANNEL_PROFILE_OVERRIDES.get(channel_id)
        if override and override in PROFILES:
            return PROFILES[override]
    if not ACTIVE_PROFILE_NAME or ACTIVE_PROFILE_NAME not in PROFILES:
        raise RuntimeError("No active profile loaded")
    return PROFILES[ACTIVE_PROFILE_NAME]


def set_channel_profile(channel_id: int, profile_name: str) -> None:
    if profile_name not in PROFILES:
        raise RuntimeError(f"Unknown profile: {profile_name}")
    CHANNEL_PROFILE_OVERRIDES[channel_id] = profile_name


def list_profiles() -> List[str]:
    return list(PROFILES.keys())


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN is required")

load_profiles()

GOOGLE_CLIENTS: dict[str, genai.Client] = {}
OPENAI_CLIENTS: dict[tuple[str, str], OpenAI] = {}


def get_google_client(profile: Profile) -> genai.Client:
    if not GOOGLE_AVAILABLE:
        raise RuntimeError("google-genai package is not installed. Install with: pip install google-genai")
    if not profile.api_key:
        raise RuntimeError(f"API key missing for profile '{profile.name}'")
    if profile.api_key in GOOGLE_CLIENTS:
        return GOOGLE_CLIENTS[profile.api_key]
    client = genai.Client(api_key=profile.api_key)
    GOOGLE_CLIENTS[profile.api_key] = client
    return client


def get_openai_client(profile: Profile) -> OpenAI:
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package is not installed. Install with: pip install openai")
    if not profile.api_key:
        raise RuntimeError(f"API key missing for profile '{profile.name}'")
    base_url = profile.base_url or DEFAULT_BASE_URL
    cache_key = (profile.api_key, base_url)
    if cache_key in OPENAI_CLIENTS:
        return OPENAI_CLIENTS[cache_key]
    client = OpenAI(api_key=profile.api_key, base_url=base_url)
    OPENAI_CLIENTS[cache_key] = client
    return client

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(command_prefix="!", intents=intents)
client = bot  # For backward compatibility

TOKENS_FILE = "data/tokens_monthly.json"

# Rate limiting (per user)
REQUEST_TIMES: dict[int, deque[datetime]] = {}

# Token update lock
TOKENS_LOCK = asyncio.Lock()


def ensure_data_dir() -> None:
    """Ensure data directory exists."""
    os.makedirs("data", exist_ok=True)


def load_monthly_tokens() -> dict:
    """Load monthly token counts from file."""
    ensure_data_dir()
    if not os.path.exists(TOKENS_FILE):
        return {"month": "", "input": 0, "output": 0, "cost": 0.0, "requests": 0}
    try:
        with open(TOKENS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"month": "", "input": 0, "output": 0, "cost": 0.0, "requests": 0}


def save_monthly_tokens(data: dict) -> None:
    """Save monthly token counts to file."""
    ensure_data_dir()
    try:
        with open(TOKENS_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def calculate_cost(input_tokens: int, output_tokens: int, profile: Profile) -> float:
    """Calculate cost based on tokens and profile pricing."""
    input_cost = (input_tokens / 1_000_000) * profile.input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * profile.output_cost_per_1m
    return round(input_cost + output_cost, 6)


def check_rate_limit(user_id: int) -> Optional[str]:
    """Check if user has exceeded rate limit. Returns error message if exceeded, None otherwise."""
    now = datetime.now()
    cutoff = now.timestamp() - 60
    
    if user_id not in REQUEST_TIMES:
        REQUEST_TIMES[user_id] = deque()
    
    user_times = REQUEST_TIMES[user_id]
    while user_times and user_times[0].timestamp() < cutoff:
        user_times.popleft()
    
    if len(user_times) >= SETTINGS.max_requests_per_minute:
        return f"⚠️ レート制限: あなたは1分間に{SETTINGS.max_requests_per_minute}リクエストまでです。少し待ってから再試行してください。"
    
    user_times.append(now)
    return None


def get_cost_warning(tokens_data: dict) -> Optional[str]:
    """Check if cost warning threshold is reached. Returns warning message if applicable."""
    if SETTINGS.max_monthly_cost_usd <= 0:
        return None
    
    cost = tokens_data.get("cost", 0.0)
    cost_ratio = cost / SETTINGS.max_monthly_cost_usd
    
    if cost_ratio >= 0.95:
        return f"🚨 **警告**: 月間コスト上限の95%に達しました ({format_usd_cost(cost)} / {format_usd_cost(SETTINGS.max_monthly_cost_usd)})"
    elif cost_ratio >= 0.90:
        return f"⚠️ **警告**: 月間コスト上限の90%に達しました ({format_usd_cost(cost)} / {format_usd_cost(SETTINGS.max_monthly_cost_usd)})"
    elif cost_ratio >= 0.80:
        return f"⚠️ 注意: 月間コスト上限の80%に達しました ({format_usd_cost(cost)} / {format_usd_cost(SETTINGS.max_monthly_cost_usd)})"
    elif cost_ratio >= 0.50:
        return f"💡 月間コスト上限の50%に達しました ({format_usd_cost(cost)} / {format_usd_cost(SETTINGS.max_monthly_cost_usd)})"
    
    return None


def update_monthly_tokens(input_tokens: int, output_tokens: int, profile: Profile) -> dict:
    """Update monthly token counts and reset if month changed. Must be called within TOKENS_LOCK."""
    current_month = datetime.now().strftime("%Y-%m")
    tokens_data = load_monthly_tokens()
    
    # Reset if month changed
    if tokens_data["month"] != current_month:
        tokens_data = {"month": current_month, "input": 0, "output": 0, "cost": 0.0, "requests": 0}
    
    tokens_data["input"] += input_tokens
    tokens_data["output"] += output_tokens
    tokens_data["cost"] = round(tokens_data.get("cost", 0.0) + calculate_cost(input_tokens, output_tokens, profile), 6)
    tokens_data["requests"] = tokens_data.get("requests", 0) + 1
    save_monthly_tokens(tokens_data)
    return tokens_data


async def update_bot_status(tokens_data: dict) -> None:
    """Update bot status with monthly token counts and cost."""
    if not client.user:
        return
    cost = tokens_data.get("cost", 0.0)
    cost_str = format_usd_cost(cost)
    jpy_str = format_jpy_cost(cost, SETTINGS.usd_to_jpy_rate)
    input_token_str = format_token_count(tokens_data['input'])
    output_token_str = format_token_count(tokens_data['output'])
    activity = discord.Activity(
        type=discord.ActivityType.playing,
        name=f"💸 {cost_str} {jpy_str} | 📥 {input_token_str} • 📤 {output_token_str}"
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
        parts.extend(
            f"[添付] {a.filename} ({format_file_size(a.size)})" for a in message.attachments
        )
    content = "\n".join(p for p in parts if p)
    return content


def format_file_size(size_bytes: int) -> str:
    size_mb = size_bytes / (1024 * 1024)
    return f"{size_mb:.2f}MB"


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def estimate_image_tokens(width: Optional[int], height: Optional[int]) -> int:
    if not width or not height:
        return 258
    if width <= 384 and height <= 384:
        return 258
    tiles_x = math.ceil(width / 768)
    tiles_y = math.ceil(height / 768)
    return 258 * tiles_x * tiles_y


def estimate_attachment_tokens(attachment: discord.Attachment) -> int:
    if attachment.content_type and attachment.content_type.startswith("image/"):
        return estimate_image_tokens(attachment.width, attachment.height)
    if attachment.width and attachment.height:
        return estimate_image_tokens(attachment.width, attachment.height)
    return max(1, math.ceil(attachment.size / 4))


def format_attachment_list(attachments: List[discord.Attachment]) -> str:
    return ", ".join(
        f"{a.filename} ({format_file_size(a.size)})" for a in attachments
    )


def validate_attachments(attachments: List[discord.Attachment], max_count: int, max_size_bytes: int) -> List[str]:
    errors: List[str] = []
    if len(attachments) > max_count:
        errors.append(
            f"添付数が上限({max_count}件)を超えています: {format_attachment_list(attachments)}"
        )
    oversize = [a for a in attachments if a.size > max_size_bytes]
    if oversize:
        errors.append(
            f"サイズ上限({format_file_size(max_size_bytes)})を超えています: {format_attachment_list(oversize)}"
        )
    return errors


async def build_gemini_file_parts(profile: Profile, attachments: List[discord.Attachment]) -> tuple[List[types.Part], int]:
    client = get_google_client(profile)
    parts: List[types.Part] = []
    estimated_tokens = 0
    max_retries = 3
    
    for attachment in attachments:
        estimated_tokens += estimate_attachment_tokens(attachment)
        data = await attachment.read()
        suffix = os.path.splitext(attachment.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        
        uploaded = None
        last_error = None
        for attempt in range(max_retries):
            try:
                uploaded = await asyncio.to_thread(client.files.upload, file=tmp_path)
                break
            except Exception as e:
                last_error = e
                logger.warning(f"ファイルアップロード失敗 (試行 {attempt + 1}/{max_retries}): {attachment.filename} - {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        
        if not uploaded:
            raise RuntimeError(
                f"ファイルのアップロードに失敗しました（{max_retries}回試行）: {attachment.filename}\n"
                f"エラー: {last_error}"
            )
        
        file_uri = getattr(uploaded, "uri", None) or getattr(uploaded, "file_uri", None)
        mime_type = getattr(uploaded, "mime_type", None) or attachment.content_type or "application/octet-stream"
        if not file_uri:
            raise RuntimeError(f"ファイルのアップロードに失敗しました: {attachment.filename}")
        parts.append(types.Part(file_data=types.FileData(file_uri=file_uri, mime_type=mime_type)))
    
    return parts, estimated_tokens


def estimate_tokens_from_contents(contents: List[types.Content]) -> int:
    estimated = 0
    for content in contents:
        for part in content.parts:
            text = getattr(part, "text", None)
            if text:
                estimated += estimate_text_tokens(text)
    return estimated


def format_token_count(count: int) -> str:
    """Format token count with k suffix for values >= 1000."""
    if count >= 1000:
        return f"{count / 1000:.1f}k"
    return str(count)


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in human-readable format."""
    if seconds < 1:
        return f"{seconds:.2f}s"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = seconds / 60
        return f"{minutes:.1f}m"


def format_jpy_cost(usd_cost: float, rate: float) -> str:
    """Format cost in JPY with ≈ prefix."""
    jpy_cost = usd_cost * rate
    if jpy_cost < 1:
        return f"(≈ ¥{jpy_cost:.2f})"
    return f"(≈ ¥{jpy_cost:.0f})"


def format_usd_cost(usd_cost: float) -> str:
    """Format cost in USD or cents."""
    if usd_cost < 0.10:
        cents = usd_cost * 100
        return f"{cents:.2f}¢"
    return f"${usd_cost:.4f}"


@dataclass
class BuildResult:
    payload: Union[List[types.Content], List[dict]]
    estimated_input_tokens: Optional[int] = None


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


def build_help_message(channel_id: Optional[int] = None) -> str:
    profile = get_active_profile(channel_id)
    settings_lines = [
        "現在の設定:",
        f"- プロファイル: {profile.name}",
        f"- プラットフォーム: {profile.platform}",
        f"- モデル: {profile.model}",
    ]
    if profile.base_url:
        settings_lines.append(f"- エンドポイント: {profile.base_url}")
    
    # Google Grounding表示（google使用時のみ）
    if profile.platform == "google":
        grounding_status = "有効" if SETTINGS.enable_google_grounding else "無効"
        settings_lines.append(f"- Google Grounding with Google Search: {grounding_status}")
    
    settings_lines.extend([
        f"- チャンネル履歴参照: {SETTINGS.channel_history_limit}件",
        f"- 返信参照チェーン: {SETTINGS.reply_chain_limit}件",
        f"- Web検索: {'有効' if SETTINGS.enable_web_search and TAVILY_API_KEY else '無効'}",
        f"- 検索プロバイダ: {SETTINGS.search_provider}",
        f"- 出典リンク表示: {'有効' if SETTINGS.enable_citation_links else '無効'}",
        f"- 1分間の最大リクエスト数: {SETTINGS.max_requests_per_minute}",
        f"- 月間コスト上限(USD): {SETTINGS.max_monthly_cost_usd}",
    ])
    settings_lines.append(
        f"- 添付上限: {SETTINGS.max_attachments_per_message}件 / {SETTINGS.max_attachment_size_mb}MB/件"
    )
    if profile.platform == "google":
        settings_lines.append("- 添付ファイル入力: 有効（Gemini Files API）")
    else:
        settings_lines.append("- 添付ファイル入力: 無効（OpenAIでは添付不可）")
    
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
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": True,
                    "include_raw_content": False,
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                response.raise_for_status()
                data = await response.json()
    except Exception as e:
        logger.error(f"Tavily検索エラー: {e}")
        return f"検索に失敗しました: {e}"
    
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


async def build_messages(message: discord.Message, prompt: str, profile: Profile) -> BuildResult:
    if profile.platform == "google":
        contents, estimated_input_tokens = await build_messages_google(message, prompt, profile)
        return BuildResult(payload=contents, estimated_input_tokens=estimated_input_tokens)
    messages = await build_messages_openai(message, prompt, profile)
    return BuildResult(payload=messages)


async def build_messages_google(message: discord.Message, prompt: str, profile: Profile) -> tuple[List[types.Content], int]:
    contents: List[types.Content] = []
    estimated_input_tokens = 0
    
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
    estimated_input_tokens += estimate_text_tokens(combined_system)
    contents.append(types.Content(role="user", parts=[types.Part(text=combined_system)]))
    contents.append(types.Content(role="model", parts=[types.Part(text="了解しました。")]))
    
    # Add history
    for m in history:
        content_text = format_message_content(m)
        if content_text:
            role = "model" if m.author == client.user else "user"
            author_prefix = "" if m.author == client.user else f"{m.author.display_name}: "
            estimated_input_tokens += estimate_text_tokens(f"{author_prefix}{content_text}")
            contents.append(types.Content(role=role, parts=[types.Part(text=f"{author_prefix}{content_text}")]))
    
    # Add current prompt with attachments
    prompt_text = f"{message.author.display_name}: {prompt}"
    estimated_input_tokens += estimate_text_tokens(prompt_text)
    parts = [types.Part(text=prompt_text)]
    if message.attachments:
        file_parts, attachment_tokens = await build_gemini_file_parts(profile, message.attachments)
        parts.extend(file_parts)
        estimated_input_tokens += attachment_tokens
    contents.append(types.Content(role="user", parts=parts))
    
    return contents, estimated_input_tokens


async def build_messages_openai(message: discord.Message, prompt: str, profile: Profile) -> List[dict]:
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


async def run_llm(messages: BuildResult, profile: Profile) -> tuple[str, int, int, bool]:
    if profile.platform == "google":
        return await run_llm_google(messages.payload, profile, messages.estimated_input_tokens)
    return await run_llm_openai(messages.payload, profile)


async def run_llm_google(contents: List[types.Content], profile: Profile, estimated_input_tokens: Optional[int]) -> tuple[str, int, int, bool]:
    tools = build_tool_spec_google()
    client = get_google_client(profile)
    used_estimate = False
    
    response = await asyncio.to_thread(
        client.models.generate_content,
        model=profile.model,
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
    else:
        used_estimate = True
        estimated = estimated_input_tokens if estimated_input_tokens is not None else estimate_tokens_from_contents(contents)
        input_tokens = estimated
        output_tokens = estimate_text_tokens(response.text or "")
    
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
            
            estimated_follow_up_input = 0
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    if fc.name == "tavily_search":
                        query = fc.args.get("query", "")
                        max_results = int(fc.args.get("max_results", 5))
                        tool_result = await tavily_search(query, max_results=max_results)
                        estimated_follow_up_input += estimate_text_tokens(query)
                        estimated_follow_up_input += estimate_text_tokens(tool_result)
                        
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
            follow_up = await asyncio.to_thread(
                client.models.generate_content,
                model=profile.model,
                contents=contents,
            )
            # Update token counts for follow-up
            if hasattr(follow_up, 'usage_metadata') and follow_up.usage_metadata:
                input_tokens += follow_up.usage_metadata.prompt_token_count or 0
                output_tokens += follow_up.usage_metadata.candidates_token_count or 0
            else:
                used_estimate = True
                input_tokens += estimated_follow_up_input
                output_tokens += estimate_text_tokens(follow_up.text or "")
            return (follow_up.text or "") + grounding_citations, input_tokens, output_tokens, used_estimate
    
    return (response.text or "") + grounding_citations, input_tokens, output_tokens, used_estimate


async def run_llm_openai(messages: List[dict], profile: Profile) -> tuple[str, int, int, bool]:
    tools = build_tool_spec_openai()
    client = get_openai_client(profile)
    
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=profile.model,
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
        follow_up = await asyncio.to_thread(
            client.chat.completions.create,
            model=profile.model,
            messages=messages,
        )
        # Accumulate token counts from follow-up
        if follow_up.usage:
            input_tokens += follow_up.usage.prompt_tokens
            output_tokens += follow_up.usage.completion_tokens
        return follow_up.choices[0].message.content or "", input_tokens, output_tokens, False

    return message.content or "", input_tokens, output_tokens, False


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


lmcord_group = app_commands.Group(name="lmcord", description="LMcordの操作コマンド")


@lmcord_group.command(name="help", description="現在の設定とヘルプを表示")
async def llmcord_help(interaction: discord.Interaction):
    help_text = build_help_message(interaction.channel_id)
    await interaction.response.send_message(help_text, ephemeral=True)


@lmcord_group.command(name="profiles", description="利用可能なプロファイル一覧を表示")
async def llmcord_profiles(interaction: discord.Interaction):
    channel_id = interaction.channel_id
    active_default = ACTIVE_PROFILE_NAME or "(none)"
    active_channel = CHANNEL_PROFILE_OVERRIDES.get(channel_id) if channel_id else None
    lines = ["利用可能なプロファイル:"]
    for name in list_profiles():
        marks = []
        if name == active_default:
            marks.append("default")
        if active_channel and name == active_channel:
            marks.append("channel")
        suffix = f" ({', '.join(marks)})" if marks else ""
        lines.append(f"- {name}{suffix}")
    await interaction.response.send_message("\n".join(lines), ephemeral=True)


@lmcord_group.command(name="switch", description="このチャンネルのプロファイルを切り替え")
@app_commands.describe(profile="切り替え先プロファイル")
@app_commands.choices(profile=[
    app_commands.Choice(name=name, value=name) for name in list_profiles()
])
async def llmcord_switch(interaction: discord.Interaction, profile: str):
    if not interaction.channel_id:
        await interaction.response.send_message("チャンネル情報が取得できませんでした。", ephemeral=True)
        return
    try:
        set_channel_profile(interaction.channel_id, profile)
        await interaction.response.send_message(f"このチャンネルのプロファイルを '{profile}' に切り替えました。")
    except Exception as exc:
        await interaction.response.send_message(f"切り替えに失敗しました: {exc}", ephemeral=True)


@lmcord_group.command(name="stats", description="月次統計情報を表示")
async def llmcord_stats(interaction: discord.Interaction):
    tokens_data = load_monthly_tokens()
    month = tokens_data.get("month", "N/A")
    input_tokens = tokens_data.get("input", 0)
    output_tokens = tokens_data.get("output", 0)
    cost = tokens_data.get("cost", 0.0)
    requests = tokens_data.get("requests", 0)
    
    cost_str = format_usd_cost(cost)
    jpy_str = format_jpy_cost(cost, SETTINGS.usd_to_jpy_rate)
    input_token_str = format_token_count(input_tokens)
    output_token_str = format_token_count(output_tokens)
    
    avg_input = input_tokens // requests if requests > 0 else 0
    avg_output = output_tokens // requests if requests > 0 else 0
    avg_cost = cost / requests if requests > 0 else 0.0
    avg_input_str = format_token_count(avg_input)
    avg_output_str = format_token_count(avg_output)
    avg_cost_str = format_usd_cost(avg_cost)
    avg_jpy_str = format_jpy_cost(avg_cost, SETTINGS.usd_to_jpy_rate)
    
    stats_lines = [
        f"📊 月次統計情報 ({month})",
        "",
        f"💸 総コスト: {cost_str} {jpy_str}",
        f"📥 総入力トークン: {input_token_str}",
        f"📤 総出力トークン: {output_token_str}",
        f"📋 総リクエスト数: {requests}",
        "",
        f"💸 平均コスト: {avg_cost_str} {avg_jpy_str}",
        f"📥 平均入力トークン: {avg_input_str}",
        f"📤 平均出力トークン: {avg_output_str}",
    ]
    
    await interaction.response.send_message("\n".join(stats_lines), ephemeral=True)


bot.tree.add_command(lmcord_group)


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
        await message.channel.send(build_help_message(message.channel.id))
        return

    # Check rate limit (per user)
    rate_limit_error = check_rate_limit(message.author.id)
    if rate_limit_error:
        await message.channel.send(rate_limit_error, reference=message)
        return

    profile = get_active_profile(message.channel.id)

    attachments = list(message.attachments or [])
    if attachments:
        max_size_bytes = SETTINGS.max_attachment_size_mb * 1024 * 1024
        errors = validate_attachments(
            attachments,
            SETTINGS.max_attachments_per_message,
            max_size_bytes,
        )
        if errors:
            error_lines = ["添付ファイルの制限により処理できません。"]
            error_lines.extend(f"- {err}" for err in errors)
            await message.channel.send("\n".join(error_lines), reference=message)
            return
        if profile.platform == "openai":
            error_lines = [
                "OpenAIプロファイルでは添付ファイル入力に対応していません。",
                "添付を削除して再送してください。",
                f"添付: {format_attachment_list(attachments)}",
            ]
            await message.channel.send("\n".join(error_lines), reference=message)
            return

    # Log incoming message
    logger.info(f"📨 Message from {message.author.name} in #{message.channel.name} [{profile.name}]: {prompt[:100]}...")

    await message.channel.typing()
    start_time = datetime.now()
    try:
        msg_data = await build_messages(message, prompt, profile)
        reply, input_tokens, output_tokens, used_estimate = await run_llm(msg_data, profile)
        elapsed_time = (datetime.now() - start_time).total_seconds()
        if not reply:
            reply = "申し訳ありません、応答の生成に失敗しました。"
        
        # Log response
        estimate_flag = "推定" if used_estimate else "実測"
        logger.info(
            f"✅ Responded to {message.author.name} [{profile.name}] ({input_tokens} in, {output_tokens} out, {estimate_flag})"
        )
        
        chunks = split_discord_message(reply)
        for i, chunk in enumerate(chunks):
            if i == 0:
                await message.channel.send(chunk, reference=message)
            else:
                await message.channel.send(chunk)
        
        # Send token count as an embed
        cost = calculate_cost(input_tokens, output_tokens, profile)
        cost_str = format_usd_cost(cost)
        jpy_str = format_jpy_cost(cost, SETTINGS.usd_to_jpy_rate)
        input_token_str = format_token_count(input_tokens)
        output_token_str = format_token_count(output_tokens)
        elapsed_str = format_elapsed_time(elapsed_time)
        estimate_suffix = " (推定)" if used_estimate else ""
        embed = discord.Embed(
            color=discord.Color.greyple(),
            description=f"💸 {cost_str} {jpy_str} | 📥 {input_token_str} • 📤 {output_token_str} • ⏱️ {elapsed_str}{estimate_suffix}\n{SETTINGS.embed_footer}"
        )
        await message.channel.send(embed=embed)
        
        # Update monthly token counts and bot status (with lock)
        async with TOKENS_LOCK:
            tokens_data = update_monthly_tokens(input_tokens, output_tokens, profile)
        await update_bot_status(tokens_data)
        
        # Check and send cost warning
        cost_warning = get_cost_warning(tokens_data)
        if cost_warning:
            await message.channel.send(cost_warning)
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
