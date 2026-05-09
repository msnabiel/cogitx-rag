"""Telegram bot integration for CogitX-RAG"""

import os
import tempfile
from typing import Any, Callable, Awaitable

from loguru import logger
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

from src.config.settings import settings
from src.core.models import Query


def _format_citations(citations):
    if not citations:
        return ""
    lines = ["", "Sources:"]
    for item in citations[:3]:
        label = item.get("label") or item.get("citation") or "[?]"
        content = item.get("content", "").strip().replace("\n", " ")
        snippet = content[:140].strip()
        lines.append(f"- {label}")
        if snippet:
            lines.append(f"  {snippet}{'...' if len(content) > 140 else ''}")
    return "\n".join(lines)


def create_telegram_app(
    query_rag: Callable[[Query], Awaitable[Any]],
    ingest_files_and_query: Callable[[list[str], str, str], Awaitable[Any]],
) -> Application:
    app = Application.builder().token(settings.telegram.telegram_bot_token).build()

    async def _answer(update: Update, context: ContextTypes.DEFAULT_TYPE, query_text: str, session_id: str):
        result = await query_rag(Query(text=query_text, session_id=session_id, top_k=3))
        citation_text = _format_citations(getattr(result, "citations", []))
        await update.message.reply_text(f"{result.answer}{citation_text}")

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Send a question or attach a document with your question.")

    async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text:
            return
        session_id = f"telegram_{update.effective_chat.id}"
        await _answer(update, context, update.message.text, session_id)

    async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.document:
            return
        document = update.message.document
        query_text = update.message.caption or update.message.text or "Summarize this document in one line."
        session_id = f"telegram_{update.effective_chat.id}_{update.message.message_id}"

        tg_file = await context.bot.get_file(document.file_id)
        suffix = os.path.splitext(document.file_name or "document")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            await tg_file.download_to_drive(custom_path=tmp.name)
            tmp_path = tmp.name

        try:
            response = await ingest_files_and_query([tmp_path], query_text, session_id)
            await update.message.reply_text(f"{response['answer']}{_format_citations(response.get('citations', []))}")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    return app


async def start_telegram_bot(
    query_rag: Callable[[Query], Awaitable[Any]],
    ingest_files_and_query: Callable[[list[str], str, str], Awaitable[Any]],
    enabled: bool = True,
):
    if not enabled:
        logger.warning("Telegram integration disabled in settings")
        return

    if not settings.telegram.telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN not configured")
        return

    logger.info("Starting Telegram bot...")
    app = create_telegram_app(query_rag, ingest_files_and_query)
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
