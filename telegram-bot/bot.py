import os
import logging
import base64
import io
from collections import defaultdict
from typing import Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from telegram.constants import ChatAction, ParseMode
from google import genai
from google.genai import types

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_API_KEY = os.environ["AI_INTEGRATIONS_GEMINI_API_KEY"]
GEMINI_BASE_URL = os.environ["AI_INTEGRATIONS_GEMINI_BASE_URL"]

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        base_url=GEMINI_BASE_URL,
        api_version="",
    ),
)

CHAT_MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-2.5-flash-image"

SYSTEM_PROMPT = """Eres un asistente de IA útil, amigable y creativo disponible en Telegram.
Puedes ayudar con una amplia variedad de tareas: responder preguntas, redactar textos, programar, analizar información, hacer lluvia de ideas y mucho más.
IMPORTANTE: Siempre responde en español, sin importar el idioma en que te escriban.
Mantén las respuestas concisas y bien formateadas para Telegram (usa markdown cuando sea útil).
Cuando te envíen una imagen, descríbela y analízala con detalle."""

conversation_history: dict[int, list[dict]] = defaultdict(list)

MAX_HISTORY = 20


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    name = user.first_name if user else "there"
    await update.message.reply_text(
        f"👋 ¡Hola {name}! Soy tu asistente de IA impulsado por Gemini.\n\n"
        "Puedo ayudarte con:\n"
        "💬 *Chat* — solo envíame cualquier mensaje\n"
        "🎨 *Generar imágenes* — usa /image seguido de una descripción\n"
        "🔍 *Analizar imágenes* — envíame una foto\n"
        "🗑️ *Borrar historial* — usa /clear para empezar una conversación nueva\n\n"
        "¡Prueba enviándome un mensaje para empezar!",
        parse_mode=ParseMode.MARKDOWN,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🤖 *Comandos disponibles*\n\n"
        "/start — Mensaje de bienvenida\n"
        "/help — Mostrar esta ayuda\n"
        "/image `<descripción>` — Generar una imagen\n"
        "/clear — Borrar el historial de conversación\n\n"
        "*Consejos:*\n"
        "• Escribe cualquier mensaje para chatear conmigo\n"
        "• Envíame una foto y la analizaré\n"
        "• Recuerdo nuestra conversación — ¡puedes hacer preguntas de seguimiento!\n"
        "• Usa /clear para empezar un chat nuevo",
        parse_mode=ParseMode.MARKDOWN,
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    conversation_history[user_id] = []
    await update.message.reply_text(
        "🗑️ ¡Historial borrado! Empecemos de nuevo.",
    )


async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Por favor, escribe una descripción para la imagen.\n"
            "Ejemplo: `/image un atardecer sobre el océano con nubes dramáticas`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    prompt = " ".join(context.args)
    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)

    try:
        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        image_data = None
        mime_type = "image/png"
        caption_text = None

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_data = part.inline_data.data
                mime_type = part.inline_data.mime_type
            elif part.text:
                caption_text = part.text

        if image_data:
            image_bytes = base64.b64decode(image_data) if isinstance(image_data, str) else image_data
            bio = io.BytesIO(image_bytes)
            bio.name = "image.png"
            await update.message.reply_photo(
                photo=bio,
                caption=caption_text or f"🎨 Generado: {prompt}",
            )
        else:
            await update.message.reply_text(
                "⚠️ No se pudo generar una imagen con esa descripción. Por favor, intenta con otra."
            )

    except Exception as e:
        logger.error(f"Image generation error: {e}")
        await update.message.reply_text(
            "❌ Algo salió mal al generar la imagen. Por favor, inténtalo de nuevo."
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_text = update.message.text

    await update.message.chat.send_action(ChatAction.TYPING)

    history = conversation_history[user_id]
    history.append({"role": "user", "parts": [{"text": user_text}]})

    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
        conversation_history[user_id] = history

    try:
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=8192,
            ),
        )

        reply_text = response.text

        history.append({"role": "model", "parts": [{"text": reply_text}]})
        conversation_history[user_id] = history

        chunks = split_message(reply_text)
        for i, chunk in enumerate(chunks):
            if i == 0:
                await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
            else:
                await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Chat error: {e}")
        try:
            await update.message.reply_text(
                "❌ Algo salió mal. Por favor, inténtalo de nuevo."
            )
        except Exception:
            pass


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    caption = update.message.caption or "Please describe and analyze this image in detail."

    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        photo = update.message.photo[-1]
        photo_file = await photo.get_file()
        photo_bytes = await photo_file.download_as_bytearray()

        image_part = types.Part(
            inline_data=types.Blob(
                mime_type="image/jpeg",
                data=bytes(photo_bytes),
            )
        )

        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        image_part,
                        types.Part(text=caption),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=8192,
            ),
        )

        reply_text = response.text

        conversation_history[user_id].append(
            {"role": "user", "parts": [{"text": f"[User sent an image] {caption}"}]}
        )
        conversation_history[user_id].append(
            {"role": "model", "parts": [{"text": reply_text}]}
        )

        chunks = split_message(reply_text)
        for chunk in chunks:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Photo handling error: {e}")
        await update.message.reply_text(
            "❌ No pude analizar esa imagen. Por favor, inténtalo de nuevo."
        )


def split_message(text: str, max_length: int = 4000) -> list[str]:
    if len(text) <= max_length:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, max_length)
        if split_at == -1:
            split_at = text.rfind(" ", 0, max_length)
        if split_at == -1:
            split_at = max_length
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    return chunks


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Update {update} caused error: {context.error}")


def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("image", image_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    logger.info("Starting Telegram bot...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
