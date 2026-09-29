"""
Lab 11 — Helper Utilities
"""
from core.config import get_llm_provider, PROVIDER_OPENROUTER  # noqa: F401
from core.openai_runtime import OpenAIRunner


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Works with OpenAIRunner (OpenAI Red / OpenRouter Blue) and Google ADK (Gemini Red).
    """
    provider = getattr(runner, "provider", None)
    if isinstance(runner, OpenAIRunner) or provider in ("openrouter", "openai"):
        text = await runner.chat(agent, user_message)
        return text, None

    from google.genai import types

    user_id = "student"
    app_name = runner.app_name

    session = None
    if session_id is not None:
        try:
            session = await runner.session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        except (ValueError, KeyError):
            pass

    if session is None:
        try:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )
        except Exception:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    import asyncio

    max_retries = 4
    for attempt in range(max_retries):
        try:
            final_response = ""
            async for event in runner.run_async(
                user_id=user_id, session_id=session.id, new_message=content
            ):
                if hasattr(event, "content") and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_response += part.text
            return final_response, session
        except Exception as e:
            err_str = str(e)
            err_msg = err_str.lower()
            if attempt < max_retries - 1 and (
                "429" in err_msg
                or "503" in err_msg
                or "resource_exhausted" in err_msg
                or "unavailable" in err_msg
            ):
                import re

                m = re.search(r"retry in ([0-9.]+)s", err_str) or re.search(
                    r"retryDelay': '(\d+)s'", err_str
                )
                if m:
                    wait_sec = float(m.group(1)) + 3.0
                else:
                    wait_sec = 25.0 * (attempt + 1)
                print(
                    f"[chat_with_agent] Transient rate limit ({type(e).__name__}). Retrying in {wait_sec:.1f}s (attempt {attempt + 1}/{max_retries})...",
                    flush=True,
                )
                await asyncio.sleep(wait_sec)
                try:
                    session = await runner.session_service.create_session(
                        app_name=app_name, user_id=user_id
                    )
                except Exception:
                    pass
            else:
                raise e

