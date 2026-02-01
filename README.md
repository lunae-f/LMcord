# llm on discord

Discord上で @メンションで呼び出せるLLMボットです。OpenAI互換APIを使用し、既定はGoogle GeminiのOpenAI互換エンドポイントを想定しています。必要に応じてTavilyでWeb検索し、参照URLを必ず提示します。

## 主な機能
- `@ボット名 質問` で起動
- チャンネル履歴50件参照
- 返信チェーン最大50件参照
- 必要時のみTavilyでWeb検索（参照URL必須）
- 日本語固定・丁寧・正確さ重視

## セットアップ
1) 依存関係のインストール

2) .env作成
`.env.example` を `.env` にコピーして値を設定してください。

必要な環境変数:
- DISCORD_TOKEN
- OPENAI_API_KEY
- TAVILY_API_KEY（Web検索を使う場合）

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

