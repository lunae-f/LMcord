# llm on discord

Discord上で @メンションで呼び出せるLLMボットです。Google Gemini APIまたはOpenAI互換API（OpenAI、OpenRouter等）を使用し、必要に応じてTavilyでWeb検索し、参照URLを必ず提示します。

## 主な機能
- `@ボット名 質問` で起動
- チャンネル履歴50件参照
- 返信チェーン最大50件参照
- 必要時のみTavilyでWeb検索（参照URL必須）
- 日本語固定・丁寧・正確さ重視
- Google Gemini / OpenAI / OpenRouter など対応

## サポートするAPI
### Google Gemini API (PLATFORM=google)
- 既定のプラットフォーム
- `GOOGLE_API_KEY` が必要

### OpenAI互換API (PLATFORM=openai)
- OpenAI
- OpenRouter
- その他OpenAI互換のプロバイダ
- `OPENAI_API_KEY` と `OPENAI_BASE_URL` が必要

## セットアップ
1) 依存関係のインストール
```bash
pip install -r requirements.txt
```

2) .env作成
`.env.example` を `.env` にコピーして値を設定してください。

### Google Gemini APIを使う場合
```env
PLATFORM=google
GOOGLE_API_KEY=your_google_api_key
MODEL=gemini-2.0-flash-exp
```

### OpenAI APIを使う場合
```env
PLATFORM=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL=gpt-4o
```

### OpenRouterを使う場合
```env
PLATFORM=openai
OPENAI_API_KEY=your_openrouter_api_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
MODEL=anthropic/claude-3.5-sonnet
```

必須環境変数:
- `DISCORD_TOKEN`
- `GOOGLE_API_KEY` (Google使用時) または `OPENAI_API_KEY` (OpenAI互換使用時)
- `TAVILY_API_KEY`（Web検索を使う場合）

## Discordボットの作成とトークン取得
1) Discord Developer Portal にアクセスし、新しいアプリを作成します。
2) 左メニューの「Bot」から Bot を追加し、Token を発行します。
3) `DISCORD_TOKEN` に発行したトークンを設定します。

### 必要なIntents
以下を有効にしてください。
- MESSAGE CONTENT INTENT

### 必要な権限（推奨）
ボットをサーバーに招待する際、以下の権限が必要です。
- View Channels
- Read Message History
- Send Messages
- Embed Links（任意）

### 招待URLの作り方
1) Developer Portal の「OAuth2」→「URL Generator」へ移動
2) Scopes で「bot」を選択
3) Bot Permissions で上記権限を選択
4) 生成されたURLでサーバーに招待

## 起動

### ローカル環境
```bash
python bot.py
```

### Docker Compose（推奨）
```bash
docker-compose up -d
```

ログ確認:
```bash
docker-compose logs -f
```

停止:
```bash
docker-compose down
```

### Docker（手動ビルド）
```bash
docker build -t llm-discord-bot .
docker run -d --name llm-discord-bot --env-file .env llm-discord-bot
```

## 使い方
Discordで以下のように呼び出してください。
```
@ボット名 質問内容
```

空入力の場合は現在の設定値とヘルプを返します。

