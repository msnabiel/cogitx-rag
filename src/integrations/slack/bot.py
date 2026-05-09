"""Slack bot integration for CogitX-RAG"""

import asyncio
from typing import Any, Callable, Awaitable
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from loguru import logger
from src.config.settings import settings
from src.core.models import Query


def create_slack_app(query_rag: Callable[[Query], Awaitable[Any]]):
    """Create a Slack app wired to the current RAG workflow."""
    app = AsyncApp(
        token=settings.slack.slack_bot_token,
        signing_secret=settings.slack.slack_signing_secret
    )

    @app.event("app_mention")
    async def handle_mention(event, say, client):
        """Handle @bot mentions."""
        try:
            user_id = event["user"]
            text = event["text"]
            channel = event["channel"]
            thread_ts = event.get("thread_ts", event["ts"])
            query_text = text.split(">", 1)[-1].strip() if ">" in text else text

            logger.info(f"Slack query from {user_id}: {query_text}")
            await say(text="🤔 Processing your question...", thread_ts=thread_ts)

            query = Query(
                text=query_text,
                user_id=user_id,
                session_id=f"slack_{channel}_{thread_ts}",
                top_k=3,
                include_graph=True,
                include_memory=True
            )
            response = await query_rag(query)

            blocks = [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Answer:*\n{response.answer}"}
                },
                {"type": "divider"},
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"⚡ Confidence: {response.confidence:.0%} | ⏱️ {response.processing_time_ms:.0f}ms | 📚 {len(response.sources)} sources"
                        }
                    ]
                }
            ]

            if response.sources:
                source_text = "\n".join([
                    f"• {i}. [{s.source.value}] {s.content[:100]}..."
                    for i, s in enumerate(response.sources[:3], 1)
                ])
                blocks.insert(-1, {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Sources:*\n{source_text}"}
                })

            await say(blocks=blocks, thread_ts=thread_ts)

        except Exception as e:
            logger.error(f"Slack query error: {e}", exc_info=True)
            await say(text=f"❌ Error: {str(e)}", thread_ts=thread_ts)

    @app.command("/cogitx")
    async def handle_command(ack, command, say):
        """Handle /cogitx slash command."""
        await ack()

        try:
            query_text = command["text"]
            user_id = command["user_id"]
            channel_id = command["channel_id"]

            if not query_text:
                await say("Usage: `/cogitx <your question>`")
                return

            logger.info(f"Slack command from {user_id}: {query_text}")

            query = Query(
                text=query_text,
                user_id=user_id,
                session_id=f"slack_{channel_id}",
                top_k=3,
                include_graph=True,
                include_memory=True
            )
            response = await query_rag(query)

            await say(
                text=f"*Question:* {query_text}\n\n*Answer:*\n{response.answer}\n\n_Confidence: {response.confidence:.0%} | {response.processing_time_ms:.0f}ms_",
                response_type="ephemeral"
            )

        except Exception as e:
            logger.error(f"Slack command error: {e}", exc_info=True)
            await say(text=f"❌ Error: {str(e)}", response_type="ephemeral")

    return app


async def start_slack_bot(query_rag: Callable[[Query], Awaitable[Any]]):
    """Start Slack bot in socket mode"""
    if not settings.slack.slack_enabled:
        logger.warning("Slack integration disabled in settings")
        return

    if not settings.slack.slack_app_token:
        logger.error("SLACK_APP_TOKEN not configured")
        return

    logger.info("Starting Slack bot...")

    app = create_slack_app(query_rag)
    handler = AsyncSocketModeHandler(app, settings.slack.slack_app_token)
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(start_slack_bot())
