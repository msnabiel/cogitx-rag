"""Slack bot integration for CogitX-RAG"""

import asyncio
import os
import tempfile
from typing import Any, Callable, Awaitable
import httpx
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from loguru import logger
from src.config.settings import settings
from src.core.models import Query


def create_slack_app(
    query_rag: Callable[[Query], Awaitable[Any]],
    ingest_and_query: Callable[[str, str, str], Awaitable[Any]],
    ingest_files_and_query: Callable[[list[str], str, str], Awaitable[Any]],
):
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

            files = event.get("files", [])
            if files:
                token = settings.slack.slack_bot_token
                tmp_paths = []
                try:
                    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as http_client:
                        for file_item in files:
                            file_id = file_item.get("id")
                            file_name = file_item.get("name", "uploaded_file")
                            if not file_id:
                                continue

                            file_info = await client.files_info(file=file_id)
                            file_obj = file_info.get("file", {})
                            download_url = file_obj.get("url_private_download") or file_obj.get("url_private")
                            if not download_url:
                                continue

                            response = await http_client.get(
                                download_url,
                                headers={"Authorization": f"Bearer {token}"},
                            )
                            response.raise_for_status()
                            suffix = os.path.splitext(file_name)[1]
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                tmp.write(response.content)
                                tmp_paths.append(tmp.name)

                    if tmp_paths:
                        response = await ingest_files_and_query(
                            tmp_paths,
                            query_text,
                            f"slack_{channel}_{thread_ts}",
                        )
                        citation_text = ""
                        citations = response.get("citations", [])
                        if citations:
                            citation_text = "\n\n*Sources:*\n" + "\n".join(
                                f"• {item['citation']} {item['content'][:120]}..."
                                for item in citations[:3]
                            )
                        await say(text=f"{response['answer']}{citation_text}", thread_ts=thread_ts)
                        return
                finally:
                    for tmp_path in tmp_paths:
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

            query = Query(
                text=query_text,
                user_id=user_id,
                session_id=f"slack_{channel}_{thread_ts}",
                top_k=3,
                include_graph=True,
                include_memory=True
            )
            response = await query_rag(query)

            citations = getattr(response, "citations", [])
            blocks = [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": response.answer}
                }
            ]

            if citations:
                source_text = "\n".join(
                    f"• {item['citation']} {item['content'][:100]}..."
                    for item in citations[:3]
                )
                blocks.append({"type": "divider"})
                blocks.append({
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Sources:*\n{source_text}"}
                })

            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Confidence: {getattr(response, 'confidence', 0.0):.0%}"
                    }
                ]
            })

            await say(blocks=blocks, thread_ts=thread_ts)

        except Exception as e:
            logger.error("Slack query error: {}", str(e))
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

            citations = getattr(response, "citations", [])
            citation_text = ""
            if citations:
                citation_text = "\n\nSources:\n" + "\n".join(
                    f"• {item['citation']} {item['content'][:120]}..."
                    for item in citations[:3]
                )
            await say(text=f"{response.answer}{citation_text}", response_type="ephemeral")

        except Exception as e:
            logger.error(f"Slack command error: {e}", exc_info=True)
            await say(text=f"❌ Error: {str(e)}", response_type="ephemeral")

    return app


async def start_slack_bot(
    query_rag: Callable[[Query], Awaitable[Any]],
    ingest_and_query: Callable[[str, str, str], Awaitable[Any]],
    ingest_files_and_query: Callable[[list[str], str, str], Awaitable[Any]],
    enabled: bool = True,
):
    """Start Slack bot in socket mode"""
    if not enabled:
        logger.warning("Slack integration disabled in settings")
        return

    if not settings.slack.slack_app_token:
        logger.error("SLACK_APP_TOKEN not configured")
        return

    logger.info("Starting Slack bot...")

    app = create_slack_app(query_rag, ingest_and_query, ingest_files_and_query)
    handler = AsyncSocketModeHandler(app, settings.slack.slack_app_token)
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(start_slack_bot())
