#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memofey v8.0: мемы, AI-подписи (Groq), генерация картинок (Pollinations), HTTP keep-alive для Render."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
import re
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from functools import lru_cache
from io import BytesIO
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.error import BadRequest, Forbidden, RetryAfter, TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.disable(logging.CRITICAL)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMO_TOKEN = os.getenv("BOT_TOKEN", "").strip()
ADMIN_ID = int(os.getenv("ADMIN_ID", "5877375975"))
DATA_DIR = os.getenv("MEMOFEY_DIR", BASE_DIR).strip() or BASE_DIR
USER_TEMPLATES_DIR = os.path.join(DATA_DIR, "user_templates")
PREVIEW_DIR = os.path.join(USER_TEMPLATES_DIR, ".previews")
PHRASES_FILE = os.path.join(DATA_DIR, "custom_phrases.json")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
AI_MEMORY_FILE = os.path.join(DATA_DIR, "ai_memory.json")

GROQ_API_KEY = ""
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b").strip()
CONNECT_TIMEOUT = 15
READ_TIMEOUT = int(os.getenv("AI_REQUEST_TIMEOUT", "120"))
AI_MAX_TOKENS = int(os.getenv("AI_MAX_TOKENS", "400"))

POLLINATIONS_IMAGE_URL = os.getenv("POLLINATIONS_IMAGE_URL", "https://image.pollinations.ai/prompt/").rstrip("/") + "/"
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY", "").strip()
GEN_WIDTH = int(os.getenv("GEN_WIDTH", "1280"))
GEN_HEIGHT = int(os.getenv("GEN_HEIGHT", "720"))
GEN_READ_TIMEOUT = int(os.getenv("GEN_READ_TIMEOUT", "180"))
GEN_MIN_INTERVAL = float(os.getenv("GEN_MIN_INTERVAL", "46"))
GEN_PROMPT_MAX_LEN = 600

JPEG_QUALITY = 88
PREVIEW_QUALITY = 72
PREVIEW_SIZE = 512
MAX_MEME_SIZE = 1600
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
BROADCAST_CONCURRENCY = 8
BROADCAST_CHUNK_SIZE = 30
BROADCAST_MAX_LEN = 3500
PAGE_SIZE = 10
AI_MEMORIES_CAP = 500
TOKEN_PATTERN = re.compile(r"^\d{6,12}:[A-Za-z0-9_-]{30,64}$")
RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS", 1)
HTTP_HEADERS = {"User-Agent": "MemofeyBot/8.0"}


def _retry_after_seconds(resp, default: float) -> float:
    try:
        return min(120.0, max(1.0, float(resp.headers.get("Retry-After", ""))))
    except (TypeError, ValueError):
        pass
    try:
        m = re.search(r"[Rr]etry after (\d+)", resp.text or "")
        if m:
            return min(120.0, max(1.0, float(m.group(1))))
    except Exception:
        pass
    return default


def _groq_vision_request(
    system_prompt: str,
    user_prompt: str,
    image_base64: str,
    temperature: float = 0.7,
) -> dict:
    if not GROQ_API_KEY:
        return {"content": "", "error": "Не задан GROQ_API_KEY."}
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
        **HTTP_HEADERS,
    }
    payload = {
        "model": GROQ_VISION_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ],
            },
        ],
        "temperature": temperature,
        "max_completion_tokens": AI_MAX_TOKENS,
        "reasoning_effort": "none",
        "reasoning_format": "hidden",
        "stream": False,
    }
    last_error = "Неизвестная ошибка Groq."
    for attempt in range(3):
        try:
            resp = requests.post(
                GROQ_API_URL, headers=headers, json=payload,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )
        except requests.exceptions.RequestException as exc:
            last_error = f"Сеть: {str(exc)[:150]}"
            if attempt < 2:
                time.sleep(2.0)
                continue
            return {"content": "", "error": last_error}

        if resp.status_code == 200:
            try:
                data = resp.json()
                content = (data["choices"][0]["message"].get("content") or "").strip()
                if content:
                    return {"content": content, "error": None}
                last_error = "Groq вернул пустой ответ."
            except (KeyError, IndexError, ValueError, TypeError):
                last_error = "Groq вернул некорректный ответ."
            if attempt < 2:
                time.sleep(1.5)
                continue
            return {"content": "", "error": last_error}

        try:
            err = resp.json().get("error", {})
            msg = err.get("message") if isinstance(err, dict) else str(err)
        except ValueError:
            msg = resp.text[:150]
        last_error = f"HTTP {resp.status_code}: {str(msg)[:180]}"

        if resp.status_code == 429:
            if attempt < 2:
                time.sleep(_retry_after_seconds(resp, 3 + attempt * 3))
                continue
        elif resp.status_code in (401, 403):
            return {"content": "", "error": last_error + " — проверь ключ GROQ_API_KEY (доступ отклонён)."}
        elif resp.status_code == 400:
            low = str(msg).lower()
            if attempt == 0 and ("reasoning" in low or "max_completion_tokens" in low or "unsupported" in low or "unknown" in low):
                payload.pop("reasoning_effort", None)
                payload.pop("reasoning_format", None)
                payload["max_tokens"] = payload.pop("max_completion_tokens", AI_MAX_TOKENS)
                continue
            if attempt < 2 and ("model" in low and ("decommission" in low or "not" in low or "exist" in low)):
                payload["model"] = "meta-llama/llama-4-scout-17b-16e-instruct"
                payload.pop("reasoning_effort", None)
                payload.pop("reasoning_format", None)
                continue
            return {"content": "", "error": last_error}
        elif resp.status_code >= 500:
            if attempt < 2:
                time.sleep(2.0 + attempt * 2)
                continue
        else:
            return {"content": "", "error": last_error}
    return {"content": "", "error": last_error}


async def groq_vision_chat(system_prompt: str, user_prompt: str, image_base64: str, temperature: float) -> dict:
    return await asyncio.to_thread(_groq_vision_request, system_prompt, user_prompt, image_base64, temperature)


_GEN_LOCK = threading.Lock()
_GEN_LAST_TS = 0.0


def _pollinations_image_request(prompt: str) -> dict:
    global _GEN_LAST_TS
    prompt = " ".join(prompt.split())[:GEN_PROMPT_MAX_LEN]
    if not prompt:
        return {"data": None, "error": "Пустое описание."}
    seed = random.randint(1, 2_000_000_000)
    params = (
        f"?width={GEN_WIDTH}&height={GEN_HEIGHT}&seed={seed}"
        f"&nologo=true&private=true&safe=false&enhance=true"
    )
    url = POLLINATIONS_IMAGE_URL + quote(prompt, safe="") + params
    headers = dict(HTTP_HEADERS)
    if POLLINATIONS_API_KEY:
        headers["Authorization"] = f"Bearer {POLLINATIONS_API_KEY}"
    last_error = "Неизвестная ошибка Pollinations."
    with _GEN_LOCK:
        for attempt in range(4):
            wait = GEN_MIN_INTERVAL - (time.monotonic() - _GEN_LAST_TS)
            if wait > 0:
                time.sleep(wait)
            _GEN_LAST_TS = time.monotonic()
            try:
                resp = requests.get(url, headers=headers, timeout=(CONNECT_TIMEOUT, GEN_READ_TIMEOUT))
            except requests.exceptions.RequestException as exc:
                last_error = f"Сеть: {str(exc)[:150]}"
                if attempt < 3:
                    continue
                return {"data": None, "error": last_error}
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if resp.status_code == 200 and ctype.startswith("image") and resp.content:
                return {"data": resp.content, "error": None}
            if resp.status_code in (402, 429, 503):
                delay = _retry_after_seconds(resp, 15 + attempt * 10)
                last_error = f"Лимит бесплатных запросов (HTTP {resp.status_code})."
                if attempt < 3:
                    time.sleep(min(delay, 90))
                    continue
                return {"data": None, "error": last_error + " Попробуй ещё раз через минуту."}
            try:
                err = resp.json().get("error", {})
                msg = err.get("message") if isinstance(err, dict) else str(err)
            except ValueError:
                msg = resp.text[:150]
            last_error = f"HTTP {resp.status_code}: {str(msg)[:180]}"
            if resp.status_code in (400, 422):
                return {"data": None, "error": "Запрос отклонён фильтром сервиса. Измени описание."}
            if attempt < 3 and resp.status_code >= 500:
                time.sleep(3)
                continue
            return {"data": None, "error": last_error}
    return {"data": None, "error": last_error}


async def pollinations_generate(prompt: str) -> dict:
    return await asyncio.to_thread(_pollinations_image_request, prompt)


IO_LOCK = threading.RLock()
PHRASE_LOCK = threading.RLock()
USER_LOCK = threading.RLock()
TEMPLATE_LOCK = threading.RLock()


def load_json(path: str, default):
    with IO_LOCK:
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, type(default)) else default
        except Exception:
            return default


def save_json(path: str, data) -> bool:
    with IO_LOCK:
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            for attempt in range(3):
                try:
                    os.replace(tmp, path)
                    return True
                except OSError:
                    if attempt < 2:
                        time.sleep(0.5)
                    else:
                        shutil.move(tmp, path)
                        return True
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            return False
    return False


def load_ai_memory():
    return load_json(AI_MEMORY_FILE, {"likes": [], "dislikes": []})


def save_ai_memory(memory):
    memory["likes"] = memory.get("likes", [])[-50:]
    memory["dislikes"] = memory.get("dislikes", [])[-50:]
    save_json(AI_MEMORY_FILE, memory)


BASE_TOP = [
    "Я: КОД РАБОТАЕТ", "ТЫ ПОСЛЕ ДЕБАГА", "СЕНЬОР СМОТРИТ ТВОЙ КОД",
    "Я: БЫСТРО ПОФИКШУ", "ДЕДЛАЙН ЧЕРЕЗ ЧАС", "ЗАКАЗЧИК МЕНЯЕТ ТЗ",
    "КОГДА НЕТ ИНТЕРНЕТА", "AI ЗАМЕНИТ НАС", "ЗАРПЛАТА ПРИШЛА",
    "КОГДА ВСПОМНИЛ ПРО СТАНДАП", "ДЖУН СПРАШИВАЕТ ПОЧЕМУ РАБОТАЕТ",
    "Я НА КОД-РЕВЬЮ", "КОГДА НАШЁЛ ЧУЖОЙ КОММЕНТ", "ПЯТНИЦА, ДЕПЛОЙ В 17:59",
    "КОГДА СЕРВЕР ЛЁГ", "Я: ВСЕГО ОДИН МАЛЕНЬКИЙ КОММИТ",
    "КОГДА ЗАБЫЛ ТОЧКУ С ЗАПЯТОЙ", "ТЕСТИРОВЩИК ОТКРЫВАЕТ ЗАДАЧУ",
    "КОГДА БАЗА ДАННЫХ ВЫРОСЛА", "Я: ПЕРЕПИШУ ЗА ВЫХОДНЫЕ",
]
BASE_BOTTOM = [
    "А ОНО НЕ РАБОТАЕТ", "НА МОЁМ КОМПЕ РАБОТАЛО", "ЗАВТРА ПОЧИНИМ",
    "ВСЁ С НУЛЯ", "ГДЕ ДЕНЬГИ ЛЕБОВСКИ", "РАБОТАЕМ ОФЛАЙН",
    "ПИВА НЕ КУПИЛ", "И ТУТ НАЛОГИ", "И ЭТО ФИЧА, А НЕ БАГ",
    "НО НИКТО НЕ ЗАМЕТИЛ", "А ЛОГИ ПУСТЫЕ", "НО ДЕДЛАЙН ПЕРЕНЕСЛИ",
    "И STACK OVERFLOW УПАЛ", "НО GIT ГОВОРИТ КОНФЛИКТ", "А ТЕСТЫ КРАСНЫЕ",
    "НО ДОКУ НИКТО НЕ ЧИТАЕТ", "И ПРОД ЛЁГ", "НО КЛИЕНТ ХОЧЕТ ВЧЕРА",
    "А КОФЕМАШИНА СЛОМАЛАСЬ", "НО ВСТРЕЧА МОГЛА БЫТЬ ПИСЬМОМ",
]

_PHRASE_KEYS = ("top", "bottom", "top_removed", "bottom_removed")
_CUSTOM = load_json(PHRASES_FILE, {k: [] for k in _PHRASE_KEYS})
if not isinstance(_CUSTOM, dict):
    _CUSTOM = {}
for _key in _PHRASE_KEYS:
    _value = _CUSTOM.get(_key, [])
    _CUSTOM[_key] = [p for p in _value if isinstance(p, str)] if isinstance(_value, list) else []


def normalize_phrase(text: str) -> str:
    return " ".join(text.upper().split())


def _active_phrases(kind: str) -> list[str]:
    base = BASE_TOP if kind == "top" else BASE_BOTTOM
    removed = set(_CUSTOM.get(f"{kind}_removed", []))
    return [p for p in base if p not in removed] + list(_CUSTOM.get(kind, []))


def get_phrases(kind: str) -> list[str]:
    with PHRASE_LOCK:
        return list(_active_phrases(kind))


def add_phrase_sync(kind: str, raw: str) -> tuple[bool, str]:
    phrase = normalize_phrase(raw)[:200]
    if not phrase:
        return False, ""
    with PHRASE_LOCK:
        if phrase in _active_phrases(kind):
            return False, phrase
        base = BASE_TOP if kind == "top" else BASE_BOTTOM
        if phrase in base:
            removed = _CUSTOM.setdefault(f"{kind}_removed", [])
            if phrase in removed:
                removed.remove(phrase)
        else:
            _CUSTOM.setdefault(kind, []).append(phrase)
        save_json(PHRASES_FILE, _CUSTOM)
    return True, phrase


def delete_phrase_sync(kind: str, phrase: str) -> bool:
    with PHRASE_LOCK:
        if phrase not in _active_phrases(kind):
            return False
        custom = _CUSTOM.setdefault(kind, [])
        if phrase in custom:
            custom.remove(phrase)
        else:
            removed = _CUSTOM.setdefault(f"{kind}_removed", [])
            if phrase not in removed:
                removed.append(phrase)
        save_json(PHRASES_FILE, _CUSTOM)
    return True


_loaded_users = load_json(USERS_FILE, [])
USERS: set[int] = (
    {int(x) for x in _loaded_users if isinstance(x, (int, str)) and str(x).lstrip("-").isdigit()}
    if isinstance(_loaded_users, list) else set()
)


def users_snapshot() -> list[int]:
    with USER_LOCK:
        return sorted(USERS)


def add_user_sync(chat_id: int) -> bool:
    with USER_LOCK:
        if chat_id in USERS:
            return False
        USERS.add(chat_id)
        save_json(USERS_FILE, sorted(USERS))
    return True


def remove_users_sync(ids: set[int]) -> None:
    if not ids:
        return
    with USER_LOCK:
        USERS.difference_update(ids)
        save_json(USERS_FILE, sorted(USERS))


async def track_user(user_id: int) -> None:
    with USER_LOCK:
        exists = user_id in USERS
    if not exists:
        await asyncio.to_thread(add_user_sync, user_id)


EXIT_BUTTONS = {"◀️ Назад", "◀️ Выйти из поддержки", "◀️ Выйти из админки"}
GROUP_ALLOWED_TEXTS = {
    "🎭 Шаблоны", "🖼 Выбрать шаблон", "📸 Загрузить", "✍️ Добавить фразу",
    "🆘 Техподдержка", "🎲 Случайная надпись", "✍️ Своя надпись", "🧠 Нейросеть",
    "🎨 Сгенерировать картинку",
    "⬆️ Добавить верхнюю", "⬇️ Добавить нижнюю",
    "🛠 Админка", "🗑 Удалить шаблон", "🗑 Очистить всё", "🗑 Удалить фразу",
    "📊 Статистика", "📢 Рассылка",
} | EXIT_BUTTONS

CAPTION_HELP = (
    "✍️ Своя надпись — пришли текст.\n"
    "Формат: ВЕРХ | НИЗ\n"
    "Без «|» — текст будет снизу.\n"
    "🎲 Случайная надпись — возьму из банка фраз.\n"
    "🧠 Нейросеть — ИИ посмотрит картинку и придумает сам."
)


def main_kb(user_id: int) -> ReplyKeyboardMarkup:
    rows = [
        ["🎭 Шаблоны", "🖼 Выбрать шаблон"],
        ["📸 Загрузить", "🎨 Сгенерировать картинку"],
        ["✍️ Добавить фразу", "🆘 Техподдержка"],
    ]
    if user_id == ADMIN_ID:
        rows.append(["🛠 Админка"])
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


CAPTION_KB = ReplyKeyboardMarkup(
    [["🎲 Случайная надпись", "✍️ Своя надпись"], ["🧠 Нейросеть"], ["◀️ Назад"]],
    resize_keyboard=True,
)
PHRASE_KB = ReplyKeyboardMarkup(
    [["⬆️ Добавить верхнюю", "⬇️ Добавить нижнюю"], ["◀️ Назад"]], resize_keyboard=True
)
ADMIN_KB = ReplyKeyboardMarkup(
    [
        ["🗑 Удалить шаблон", "🗑 Очистить всё"],
        ["🗑 Удалить фразу"],
        ["📊 Статистика", "📢 Рассылка"],
        ["◀️ Выйти из админки"],
    ],
    resize_keyboard=True,
)
SUPPORT_KB = ReplyKeyboardMarkup([["◀️ Выйти из поддержки"]], resize_keyboard=True)
UPLOAD_KB = ReplyKeyboardMarkup([["◀️ Назад"]], resize_keyboard=True)
GEN_KB = ReplyKeyboardMarkup([["◀️ Назад"]], resize_keyboard=True)
CONFIRM_KB = InlineKeyboardMarkup(
    [[
        InlineKeyboardButton("✅ Подтвердить", callback_data="confirm"),
        InlineKeyboardButton("❌ Отмена", callback_data="cancelconfirm"),
    ]]
)


def setup_dirs() -> None:
    os.makedirs(USER_TEMPLATES_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)


def list_templates() -> list[str]:
    with TEMPLATE_LOCK:
        try:
            with os.scandir(USER_TEMPLATES_DIR) as entries:
                return sorted(
                    e.path for e in entries
                    if e.is_file() and e.name.lower().endswith((".jpg", ".jpeg", ".png"))
                )
        except OSError:
            return []


def preview_path(path: str) -> str:
    name = os.path.basename(path)
    return os.path.join(PREVIEW_DIR, f"{name}.preview.jpg")


def normalize_image_mode(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB") if img.mode != "RGB" else img


def ensure_preview(path: str) -> str:
    preview = preview_path(path)
    try:
        if os.path.exists(preview) and os.path.getmtime(preview) >= os.path.getmtime(path):
            return preview
        with Image.open(path) as src:
            img = ImageOps.exif_transpose(src)
            img.load()
            img = normalize_image_mode(img)
            img.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), RESAMPLE)
            img.save(preview, "JPEG", quality=PREVIEW_QUALITY, optimize=True, progressive=True)
        return preview
    except Exception:
        return path


def read_bytes(path: str) -> bytes | None:
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


def delete_template_file(path: str) -> None:
    for target in (path, preview_path(path)):
        try:
            os.remove(target)
        except OSError:
            pass


def clear_all_templates() -> int:
    files = list_templates()
    for path in files:
        delete_template_file(path)
    try:
        with os.scandir(PREVIEW_DIR) as entries:
            for entry in entries:
                if entry.is_file():
                    try:
                        os.remove(entry.path)
                    except OSError:
                        pass
    except OSError:
        pass
    return len(files)


def save_upload_sync(payload: bytes, owner_id: int) -> str:
    name = f"user_{owner_id}_{uuid.uuid4().hex[:10]}.jpg"
    final = os.path.join(USER_TEMPLATES_DIR, name)
    tmp = final + ".tmp"
    with Image.open(BytesIO(payload)) as src:
        img = ImageOps.exif_transpose(src)
        img.load()
        img = normalize_image_mode(img)
        img.thumbnail((MAX_MEME_SIZE, MAX_MEME_SIZE), RESAMPLE)
        img.save(tmp, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
    os.replace(tmp, final)
    return final


_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/impact.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _find_font() -> str | None:
    for path in _FONT_CANDIDATES:
        try:
            ImageFont.truetype(path, 12)
            return path
        except Exception:
            continue
    return None


_FONT_PATH = _find_font()


@lru_cache(maxsize=64)
def get_font(size: int):
    if _FONT_PATH:
        try:
            return ImageFont.truetype(_FONT_PATH, size)
        except Exception:
            pass
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def text_width(draw, text: str, font) -> float:
    try:
        return float(draw.textlength(text, font=font))
    except Exception:
        box = draw.textbbox((0, 0), text, font=font)
        return float(box[2] - box[0])


def _split_long_word(draw, word: str, font, max_width: float) -> list[str]:
    parts, current = [], ""
    for char in word:
        if current and text_width(draw, current + char, font) > max_width:
            parts.append(current)
            current = char
        else:
            current += char
    if current:
        parts.append(current)
    return parts


def wrap_text(draw, text: str, font, max_width: float) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        if text_width(draw, word, font) > max_width:
            if current:
                lines.append(current)
                current = ""
            parts = _split_long_word(draw, word, font, max_width)
            lines.extend(parts[:-1])
            current = parts[-1] if parts else ""
            continue
        candidate = f"{current} {word}".strip()
        if text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_text_safe(draw, xy, text, font, stroke_width: int) -> None:
    try:
        draw.text(xy, text, font=font, fill="white", stroke_width=stroke_width, stroke_fill="black")
    except (TypeError, AttributeError):
        draw.text(xy, text, font=font, fill="white")


def create_meme_image(path: str, top: str, bottom: str) -> BytesIO | None:
    try:
        with Image.open(path) as src:
            img = ImageOps.exif_transpose(src)
            img.load()
            img = normalize_image_mode(img)
            img = img.copy()
        if max(img.size) > MAX_MEME_SIZE:
            img.thumbnail((MAX_MEME_SIZE, MAX_MEME_SIZE), RESAMPLE)
        width, height = img.size
        draw = ImageDraw.Draw(img)
        margin = max(16, width // 20)
        max_text_width = max(32, width - 2 * margin)
        pad = max(10, height // 40)
        font_size = max(18, min(88, width // 16))
        top_lines: list[str] = []
        bottom_lines: list[str] = []
        while font_size >= 12:
            font = get_font(font_size)
            line_h = int(font_size * 1.22)
            top_lines = wrap_text(draw, top, font, max_text_width) if top else []
            bottom_lines = wrap_text(draw, bottom, font, max_text_width) if bottom else []
            used = (len(top_lines) + len(bottom_lines)) * line_h + 2 * pad
            if used <= height * 0.70 or font_size == 12:
                break
            font_size -= 2
        font = get_font(font_size)
        line_h = int(font_size * 1.22)
        stroke = max(2, font_size // 12)

        def paint(lines: list[str], start_y: int) -> None:
            y = start_y
            for line in lines:
                w = text_width(draw, line, font)
                x = max(0, int((width - w) // 2))
                draw_text_safe(draw, (x, y), line, font, stroke)
                y += line_h

        if top_lines:
            paint(top_lines, pad)
        if bottom_lines:
            paint(bottom_lines, max(pad, height - pad - len(bottom_lines) * line_h))
        buf = BytesIO()
        img.save(buf, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
        buf.seek(0)
        buf.name = "meme.jpg"
        return buf
    except Exception:
        return None


def parse_caption(text: str) -> tuple[str, str]:
    if "|" in text:
        first, second = text.split("|", 1)
        return normalize_phrase(first)[:200], normalize_phrase(second)[:200]
    return "", normalize_phrase(text)[:400]


async def send_meme(
    context,
    chat_id: int,
    top: str,
    bottom: str,
    forced_path: str | None = None,
    reply_to: int | None = None,
    ai_generated: bool = False,
    meme_id: str | None = None,
) -> None:
    path = forced_path if forced_path and os.path.exists(forced_path) else None
    if not path:
        files = await asyncio.to_thread(list_templates)
        if not files:
            await context.bot.send_message(
                chat_id, "❌ Нет шаблонов. Нажми «📸 Загрузить» или «🎨 Сгенерировать картинку».",
                reply_markup=CAPTION_KB,
            )
            return
        path = random.choice(files)
    buf = await asyncio.to_thread(create_meme_image, path, top, bottom)
    if buf is None:
        await context.bot.send_message(chat_id, "❌ Не удалось создать мем.")
        return
    caption = "\n".join(p for p in (top, bottom) if p)[:1024] or None
    kb = CAPTION_KB
    if ai_generated and meme_id:
        mem = context.bot_data.setdefault("ai_memories", {})
        mem[meme_id] = {"top": top, "bottom": bottom, "image_path": path}
        while len(mem) > AI_MEMORIES_CAP:
            mem.pop(next(iter(mem)))
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("👍 Нормально", callback_data=f"ai_rate:like:{meme_id}"),
            InlineKeyboardButton("👎 Дизлайк", callback_data=f"ai_rate:dislike:{meme_id}"),
        ]])
    try:
        await context.bot.send_photo(
            chat_id, photo=buf, caption=caption, reply_markup=kb, reply_to_message_id=reply_to,
        )
    except RetryAfter as exc:
        await asyncio.sleep(exc.retry_after)
        try:
            await context.bot.send_photo(
                chat_id, photo=buf, caption=caption, reply_markup=kb, reply_to_message_id=reply_to,
            )
        except TelegramError:
            pass
    except BadRequest:
        await context.bot.send_message(chat_id, "❌ Telegram отклонил изображение.")
    except TelegramError:
        pass
    finally:
        buf.close()


async def random_meme(context, chat_id: int, forced_path: str | None) -> None:
    top = random.choice(get_phrases("top") or ["МЕМОФЕЙ"])
    bottom = random.choice(get_phrases("bottom") or ["А НА МОЁМ РАБОТАЛО"])
    await send_meme(context, chat_id, top, bottom, forced_path)


ADMIN_REPLY_MAP: OrderedDict[int, int] = OrderedDict()


def map_reply(message_id: int, chat_id: int) -> None:
    ADMIN_REPLY_MAP[message_id] = chat_id
    while len(ADMIN_REPLY_MAP) > 5000:
        ADMIN_REPLY_MAP.popitem(last=False)


async def forward_to_admin(context, message, user) -> None:
    try:
        sent = await message.copy(chat_id=ADMIN_ID)
        map_reply(sent.message_id, message.chat.id)
        note = await context.bot.send_message(
            ADMIN_ID,
            f"⬆️ От: {user.full_name} (id: {user.id})\n💬 Ответь РЕПЛАЕМ на пересланное сообщение.",
        )
        map_reply(note.message_id, message.chat.id)
        await message.reply_text("✅ Отправлено. Жди ответ здесь.", reply_markup=SUPPORT_KB)
    except Exception:
        await message.reply_text("❌ Не удалось связаться с админом.", reply_markup=SUPPORT_KB)


async def answer_support(context, message, chat_id: int) -> bool:
    try:
        await context.bot.send_message(chat_id, "🛠 Ответ техподдержки:")
        await message.copy(chat_id=chat_id)
        return True
    except Exception:
        return False


async def show_template_page(context, chat_id: int, mode: str, page: int, old_message_id: int | None = None) -> None:
    files = await asyncio.to_thread(list_templates)
    if not files:
        await context.bot.send_message(
            chat_id, "❌ Шаблонов нет. Загрузи картинку через «📸 Загрузить».",
            reply_markup=main_kb(chat_id),
        )
        return
    pages = max(1, (len(files) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(0, min(page, pages - 1))
    start = page * PAGE_SIZE
    batch = files[start:start + PAGE_SIZE]
    if old_message_id:
        try:
            await context.bot.delete_message(chat_id, old_message_id)
        except TelegramError:
            pass
    media, buffers = [], []
    for offset, path in enumerate(batch):
        preview = await asyncio.to_thread(ensure_preview, path)
        data = await asyncio.to_thread(read_bytes, preview)
        if not data:
            continue
        bio = BytesIO(data)
        bio.name = f"template_{start + offset + 1}.jpg"
        buffers.append(bio)
        media.append(InputMediaPhoto(media=bio, caption=f"№{start + offset + 1}"))
    if media:
        try:
            await context.bot.send_media_group(chat_id, media)
        except TelegramError:
            pass
        finally:
            for bio in buffers:
                try:
                    bio.close()
                except Exception:
                    pass
    prefix = "pick" if mode == "pick" else "del"
    rows, row = [], []
    for offset in range(len(batch)):
        row.append(InlineKeyboardButton(f"№{start + offset + 1}", callback_data=f"{prefix}:{page}:{offset}"))
        if len(row) == 5:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅️", callback_data=f"{prefix}nav:{page - 1}"))
    nav.append(InlineKeyboardButton(f"{page + 1}/{pages}", callback_data="noop"))
    if page + 1 < pages:
        nav.append(InlineKeyboardButton("➡️", callback_data=f"{prefix}nav:{page + 1}"))
    rows.append(nav)
    rows.append([InlineKeyboardButton("❌ Отмена", callback_data=f"{prefix}cancel")])
    text = (
        f"🖼 Шаблоны {start + 1}–{start + len(batch)} из {len(files)}."
        + ("\nНажми номер, чтобы выбрать." if mode == "pick" else "\nНажми номер, чтобы удалить.")
    )
    await context.bot.send_message(chat_id, text, reply_markup=InlineKeyboardMarkup(rows))


def phrase_page_view(kind: str, page: int) -> tuple[str, InlineKeyboardMarkup]:
    phrases = get_phrases(kind)
    pages = max(1, (len(phrases) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(0, min(page, pages - 1))
    start = page * PAGE_SIZE
    batch = phrases[start:start + PAGE_SIZE]
    title = "⬆️ Верхние фразы" if kind == "top" else "⬇️ Нижние фразы"
    lines = [f"{title} — всего {len(phrases)}", f"Страница {page + 1}/{pages}", ""]
    for offset, phrase in enumerate(batch):
        lines.append(f"№{start + offset + 1}. {phrase}")
    if not batch:
        lines.append("Список пуст.")
    rows = []
    for offset, phrase in enumerate(batch):
        label = f"🗑 №{start + offset + 1} · {phrase[:28]}"
        rows.append([InlineKeyboardButton(label, callback_data=f"phdel:{kind}:{start + offset}")])
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅️", callback_data=f"phnav:{kind}:{page - 1}"))
    nav.append(InlineKeyboardButton("📋 Список фраз", callback_data="phmenu"))
    if page + 1 < pages:
        nav.append(InlineKeyboardButton("➡️", callback_data=f"phnav:{kind}:{page + 1}"))
    rows.append(nav)
    rows.append([InlineKeyboardButton("🏠 Главное меню", callback_data="home")])
    return "\n".join(lines)[:3800], InlineKeyboardMarkup(rows)


async def send_phrase_menu(context, chat_id: int) -> None:
    kb = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("⬆️ Верхние", callback_data="ph:top:0"),
            InlineKeyboardButton("⬇️ Нижние", callback_data="ph:bottom:0"),
        ],
        [InlineKeyboardButton("🏠 Главное меню", callback_data="home")],
    ])
    await context.bot.send_message(
        chat_id, "🗑 Удаление фраз.\nВыбери список. Удаление — с подтверждением.", reply_markup=kb,
    )


async def broadcast_worker(context, admin_chat: int, text: str) -> None:
    users = await asyncio.to_thread(users_snapshot)
    status = await context.bot.send_message(admin_chat, f"📢 Рассылка запущена. Получателей: {len(users)}")
    if not users:
        await status.edit_text("📢 Получателей нет.")
        return
    sem = asyncio.Semaphore(BROADCAST_CONCURRENCY)
    blocked: set[int] = set()
    sent = failed = 0
    payload = f"📢 Рассылка от админа:\n{text}"

    async def send_one(uid: int) -> str:
        async with sem:
            for attempt in range(3):
                try:
                    await context.bot.send_message(uid, payload)
                    return "ok"
                except RetryAfter as exc:
                    await asyncio.sleep(exc.retry_after + 0.5)
                except Forbidden:
                    return "blocked"
                except TelegramError:
                    if attempt == 2:
                        return "fail"
                    await asyncio.sleep(1 + attempt)
            return "fail"

    for pos in range(0, len(users), BROADCAST_CHUNK_SIZE):
        chunk = users[pos:pos + BROADCAST_CHUNK_SIZE]
        results = await asyncio.gather(*(send_one(uid) for uid in chunk))
        for uid, result in zip(chunk, results):
            if result == "ok":
                sent += 1
            else:
                failed += 1
                if result == "blocked":
                    blocked.add(uid)
        try:
            await status.edit_text(
                f"📢 Рассылка: {min(pos + BROADCAST_CHUNK_SIZE, len(users))}/{len(users)}\n✅ {sent} · ❌ {failed}"
            )
        except TelegramError:
            pass
        await asyncio.sleep(0.5)
    if blocked:
        await asyncio.to_thread(remove_users_sync, blocked)
    try:
        await status.edit_text(
            f"✅ Рассылка завершена.\nДоставлено: {sent}\nНе доставлено: {failed}\nЗаблокировали бота: {len(blocked)}"
        )
    except TelegramError:
        pass


def spawn_background(context, coro) -> None:
    tasks = context.bot_data.setdefault("bg_tasks", set())
    task = asyncio.create_task(coro)
    tasks.add(task)
    task.add_done_callback(tasks.discard)


class LRUDict(OrderedDict):
    def __init__(self, maxsize=10000):
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            self.popitem(last=False)


def chat_lock(context, chat_id: int) -> asyncio.Lock:
    locks = context.bot_data.setdefault("chat_locks", LRUDict())
    if chat_id not in locks:
        locks[chat_id] = asyncio.Lock()
    return locks[chat_id]


def is_admin(user_id: int) -> bool:
    return user_id == ADMIN_ID


async def safe_delete(context, chat_id: int, message_id: int | None) -> None:
    if not message_id:
        return
    try:
        await context.bot.delete_message(chat_id, message_id)
    except TelegramError:
        pass


async def finish_callback(q, context, text: str, kb=None) -> None:
    try:
        await q.edit_message_text(text, reply_markup=kb)
    except BadRequest as exc:
        if "not modified" in str(exc).lower():
            return
        if q.message:
            try:
                await q.message.reply_text(text, reply_markup=kb)
            except TelegramError:
                pass
    except TelegramError:
        if q.message:
            try:
                await q.message.reply_text(text, reply_markup=kb)
            except TelegramError:
                pass


async def go_home(context, chat_id: int, user_id: int) -> None:
    await context.bot.send_message(chat_id, "🏠 Главное меню:", reply_markup=main_kb(user_id))


def _clean_ai_content(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = re.sub(r"(?is)<think>.*", "", text)
    text = text.replace("```", " ")
    return text.strip()


def _extract_meme_lines(content: str) -> tuple[str, str]:
    content = _clean_ai_content(content)
    top = ""
    bottom = ""
    for raw_line in content.splitlines():
        line = raw_line.strip().strip('"«»“”').strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith(("ВЕРХ:", "ВЕРХ :", "TOP:", "TOP :")):
            top = line.split(":", 1)[1].strip()
        elif upper.startswith(("НИЗ:", "НИЗ :", "BOTTOM:", "BOTTOM :")):
            bottom = line.split(":", 1)[1].strip()
        if top and bottom:
            break
    if not top or not bottom:
        lines = []
        for raw_line in content.splitlines():
            line = raw_line.strip().strip('"«»“”').strip()
            if line:
                lines.append(line)
        if len(lines) >= 2:
            top, bottom = lines[0], lines[1]
        elif len(lines) == 1:
            bottom = lines[0]
    return top.strip(), bottom.strip()


def _image_base64(path: str) -> str | None:
    data = read_bytes(path)
    if not data:
        return None
    return base64.b64encode(data).decode("utf-8")


async def generate_ai_caption(image_path: str) -> dict | None:
    try:
        image_path = await asyncio.to_thread(ensure_preview, image_path)
    except Exception:
        pass
    memory = await asyncio.to_thread(load_ai_memory)
    likes = [x for x in memory.get("likes", []) if isinstance(x, dict)][-3:]
    dislikes = [x for x in memory.get("dislikes", []) if isinstance(x, dict)][-3:]
    system_prompt = (
        "Ты — автор коротких подписей для мемов. "
        "Отвечай только на русском языке. "
        "Нужно ровно две короткие строки без пояснений, без кавычек, без списков и без английского. "
        "Формат ответа строго:\n"
        "ВЕРХ: короткая ситуация\n"
        "НИЗ: короткий панчлайн"
    )
    user_prompt = (
        "Придумай надпись для мема по этой картинке. "
        "Верни только две строки в формате ВЕРХ и НИЗ."
    )
    if likes:
        user_prompt += "\nХорошие примеры:\n" + "\n".join(
            f"ВЕРХ: {x.get('top', '')} / НИЗ: {x.get('bottom', '')}"
            for x in likes if x.get("top") or x.get("bottom")
        )
    if dislikes:
        user_prompt += "\nТак делать не надо:\n" + "\n".join(
            f"ВЕРХ: {x.get('top', '')} / НИЗ: {x.get('bottom', '')}"
            for x in dislikes if x.get("top") or x.get("bottom")
        )
    img_base64 = await asyncio.to_thread(_image_base64, image_path)
    if not img_base64:
        return None
    bad_markers = (
        "ТЕКСТ В КАРТИНКЕ", "ПО ЭТОЙ КАРТИНКЕ", "ПРИДУМАЙ", "ОТВЕТЬ",
        "ВЕРХ: ...", "НИЗ: ...", "ПРИМЕРЫ", "ТАК ДЕЛАТЬ",
    )

    def has_cyrillic(s: str) -> bool:
        return any("\u0400" <= ch <= "\u04FF" for ch in s)

    last_error = ""
    for attempt in range(2):
        temperature = 0.7 if attempt == 0 else 0.25
        final_user_prompt = user_prompt
        if attempt == 1:
            final_user_prompt += (
                "\nНапоминание: нужен только русский язык. "
                "Ответ строго из двух строк: ВЕРХ и НИЗ. Без пояснений."
            )
        result = await groq_vision_chat(system_prompt, final_user_prompt, img_base64, temperature)
        content = result.get("content") or ""
        if not content:
            last_error = result.get("error") or "пустой ответ"
            continue
        top, bottom = _extract_meme_lines(content)
        top = normalize_phrase(top)[:200]
        bottom = normalize_phrase(bottom)[:200]
        if any(marker in f"{top} {bottom}" for marker in bad_markers):
            continue
        if not has_cyrillic(top + bottom):
            continue
        if not top or not bottom:
            continue
        return {"top": top, "bottom": bottom}
    if last_error:
        return {"error": last_error}
    return None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message or not update.effective_chat:
        return
    chat = update.effective_chat
    is_private = chat.type == "private"
    async with chat_lock(context, chat.id):
        context.user_data.clear()
        await track_user(update.effective_user.id)
        await update.effective_message.reply_text(
            "👋 Привет! Я Memofey v8.0\n"
            "🎭 Шаблоны — случайная картинка из твоей коллекции\n"
            "🖼 Выбрать шаблон — выбор вручную\n"
            "📸 Загрузить — сделать из фото шаблон\n"
            "🎨 Сгенерировать картинку — нарисую по описанию, можно сохранить в шаблоны\n"
            "✍️ Добавить фразу — пополнить банк надписей\n"
            "🧠 Нейросеть — ИИ посмотрит картинку и придумает надпись\n"
            "🆘 Техподдержка — связь с админом\n"
            "Своя надпись: ВЕРХ | НИЗ. Без «|» текст пойдёт вниз.\n"
            "В группах работают команды /meme и /ai.",
            reply_markup=main_kb(update.effective_user.id) if is_private else None,
        )


async def cmd_meme(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    if not message or not chat or not user:
        return
    await track_user(user.id)
    text = " ".join(context.args or []).strip()
    if text:
        top, bottom = parse_caption(text)
    else:
        top = random.choice(get_phrases("top") or ["МЕМОФЕЙ"])
        bottom = random.choice(get_phrases("bottom") or ["А НА МОЁМ РАБОТАЛО"])
    await send_meme(context, chat.id, top, bottom, reply_to=message.message_id)


async def cmd_ai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    if not message or not chat or not user:
        return
    await track_user(user.id)
    ud = context.user_data if chat.type == "private" else {}
    await ai_generate(context, chat.id, ud, message)


async def handle_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    ud = context.user_data
    if ud.get("upload_busy"):
        return
    ud["upload_busy"] = True
    status = await message.reply_text("⏳ Проверяю и сохраняю шаблон…")
    try:
        if not message.photo:
            raise ValueError("Нужно фото, а не другой тип медиа.")
        file = await context.bot.get_file(message.photo[-1].file_id)
        raw = bytes(await file.download_as_bytearray())
        if len(raw) > MAX_UPLOAD_BYTES:
            raise ValueError("Файл слишком большой (макс. 15 МБ).")
        path = await asyncio.to_thread(save_upload_sync, raw, update.effective_user.id)
        await asyncio.to_thread(ensure_preview, path)
        await status.edit_text(
            "✅ Сохранено! Шаблон доступен в «🎭 Шаблоны» и «🖼 Выбрать шаблон».",
            reply_markup=main_kb(update.effective_user.id),
        )
    except Exception as exc:
        try:
            await status.edit_text(f"❌ Не удалось сохранить: {exc}", reply_markup=main_kb(update.effective_user.id))
        except TelegramError:
            pass
    finally:
        ud["flow"] = None
        ud.pop("upload_busy", None)


async def ai_generate(context, chat_id: int, ud, message) -> None:
    path = ud.get("chosen_template")
    if not path or not os.path.exists(path):
        files = await asyncio.to_thread(list_templates)
        if not files:
            await message.reply_text("❌ Нет шаблонов. Нажми «📸 Загрузить».", reply_markup=CAPTION_KB)
            return
        path = random.choice(files)
    status = await message.reply_text("🤖 Нейросеть смотрит на картинку и придумывает надпись…")
    result = await generate_ai_caption(path)
    if not result or result.get("error"):
        reason = (result or {}).get("error") or "модель не вернула подходящий ответ"
        try:
            await status.edit_text(
                "❌ Нейросеть не смогла придумать надпись.\n"
                f"Причина: {reason}\n"
                f"Модель: {GROQ_VISION_MODEL}"
            )
        except TelegramError:
            pass
        return
    try:
        await status.delete()
    except TelegramError:
        pass
    meme_id = uuid.uuid4().hex
    await send_meme(
        context, chat_id, result["top"], result["bottom"], path,
        message.message_id, ai_generated=True, meme_id=meme_id,
    )


async def generate_picture(context, chat_id: int, ud, message, prompt: str) -> None:
    if ud.get("gen_busy"):
        await message.reply_text("⏳ Предыдущая картинка ещё рисуется, подожди.")
        return
    ud["gen_busy"] = True
    status = await message.reply_text(
        "🎨 Рисую картинку…\n"
        "Бесплатный сервис рисует по одной картинке примерно раз в минуту — может занять до пары минут."
    )
    try:
        result = await pollinations_generate(prompt)
        if not result.get("data"):
            try:
                await status.edit_text(
                    f"❌ Не удалось сгенерировать картинку.\nПричина: {result.get('error')}",
                    reply_markup=main_kb(message.from_user.id if message.from_user else chat_id),
                )
            except TelegramError:
                pass
            return
        try:
            path = await asyncio.to_thread(save_upload_sync, result["data"], chat_id)
        except Exception:
            try:
                await status.edit_text("❌ Сервис вернул повреждённое изображение. Попробуй ещё раз.")
            except TelegramError:
                pass
            return
        try:
            await status.delete()
        except TelegramError:
            pass
        gen_id = uuid.uuid4().hex[:16]
        gens = context.bot_data.setdefault("generated_images", LRUDict(500))
        gens[gen_id] = path
        buf = BytesIO(result["data"])
        buf.name = "generated.jpg"
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("➕ Добавить в шаблоны", callback_data=f"genadd:{gen_id}")],
            [
                InlineKeyboardButton("🔁 Ещё вариант", callback_data=f"genagain:{gen_id}"),
                InlineKeyboardButton("❌ Удалить", callback_data=f"gendel:{gen_id}"),
            ],
        ])
        try:
            await context.bot.send_photo(
                chat_id, photo=buf,
                caption=f"🎨 {prompt[:900]}\nДобавить эту картинку в шаблоны?",
                reply_markup=kb, reply_to_message_id=message.message_id,
            )
        except TelegramError:
            await message.reply_text("❌ Telegram отклонил изображение.")
        finally:
            buf.close()
    finally:
        ud.pop("gen_busy", None)
        if ud.get("flow") == "wait_gen_prompt":
            ud["flow"] = None


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    chat = update.effective_chat
    if not message or not user or not chat:
        return
    if chat.type not in ("private", "group", "supergroup"):
        return
    is_private = chat.type == "private"
    async with chat_lock(context, chat.id):
        ud = context.user_data
        flow = ud.get("flow")
        if is_admin(user.id) and message.reply_to_message:
            target = ADMIN_REPLY_MAP.get(message.reply_to_message.message_id)
            if target:
                await answer_support(context, message, target)
                return
        if message.photo:
            if flow == "wait_upload":
                await track_user(user.id)
                await handle_upload(update, context)
            return
        text = (message.text or "").strip()
        if not text:
            if flow == "wait_upload":
                await message.reply_text("📸 Пришли именно фото.", reply_markup=UPLOAD_KB)
            return
        if not is_private and flow is None and text not in GROUP_ALLOWED_TEXTS:
            return
        await track_user(user.id)
        if text in EXIT_BUTTONS:
            ud.clear()
            await message.reply_text("🏠 Главное меню:", reply_markup=main_kb(user.id))
            return
        if flow == "wait_upload":
            await message.reply_text("📸 Пришли именно фото.", reply_markup=UPLOAD_KB)
            return
        if flow == "support":
            await forward_to_admin(context, message, user)
            return
        if flow == "wait_gen_prompt":
            if len(text) > GEN_PROMPT_MAX_LEN:
                await message.reply_text(f"⚠️ Слишком длинное описание. Максимум {GEN_PROMPT_MAX_LEN} символов.", reply_markup=GEN_KB)
                return
            ud["last_gen_prompt"] = text
            await generate_picture(context, chat.id, ud, message, text)
            return
        if flow in {"wait_top", "wait_bottom"}:
            if text in {"⬆️ Добавить верхнюю", "⬇️ Добавить нижнюю"}:
                await message.reply_text("Пришли текст фразы.", reply_markup=PHRASE_KB)
                return
            kind = "top" if flow == "wait_top" else "bottom"
            ok, phrase = await asyncio.to_thread(add_phrase_sync, kind, text)
            await message.reply_text(
                f"✅ Добавлено: «{phrase}»" if ok else "⚠️ Такая фраза уже есть или пуста.",
                reply_markup=PHRASE_KB,
            )
            ud["flow"] = "phrase_menu"
            return
        if flow == "wait_broadcast":
            if not is_admin(user.id):
                ud.clear()
                return
            if len(text) > BROADCAST_MAX_LEN:
                await message.reply_text(f"⚠️ Слишком длинный текст. Максимум {BROADCAST_MAX_LEN} символов.")
                return
            ud["pending"] = {"type": "broadcast", "text": text}
            ud["flow"] = "admin"
            await message.reply_text("📢 Подтверди рассылку:\n" + text[:1500], reply_markup=CONFIRM_KB)
            return
        if text == "🎲 Случайная надпись":
            await random_meme(context, chat.id, ud.get("chosen_template"))
            return
        if text == "✍️ Своя надпись":
            ud["flow"] = "wait_text"
            await message.reply_text(
                "✍️ Пришли надпись.\nФормат: ВЕРХ | НИЗ\nБез «|» — текст будет снизу.\nОтмена: «◀️ Назад».",
                reply_markup=CAPTION_KB,
            )
            return
        if text == "🧠 Нейросеть":
            await ai_generate(context, chat.id, ud, message)
            return
        if flow == "wait_text":
            top, bottom = parse_caption(text)
            await send_meme(context, chat.id, top, bottom, ud.get("chosen_template"), message.message_id)
            ud["flow"] = "caption"
            return
        if text == "🎭 Шаблоны":
            ud["flow"] = "caption"
            ud.pop("chosen_template", None)
            await message.reply_text("🎲 Какую надпись поставить?\n" + CAPTION_HELP, reply_markup=CAPTION_KB)
            return
        if text == "🖼 Выбрать шаблон":
            ud["flow"] = "pick"
            await show_template_page(context, chat.id, "pick", 0)
            return
        if text == "📸 Загрузить":
            ud["flow"] = "wait_upload"
            await message.reply_text(
                "📸 Пришли фото — оно станет шаблоном.\nОтмена: «◀️ Назад».", reply_markup=UPLOAD_KB,
            )
            return
        if text == "🎨 Сгенерировать картинку":
            ud["flow"] = "wait_gen_prompt"
            await message.reply_text(
                "🎨 Опиши, что нарисовать.\n"
                "Например: «рыжий кот за офисным столом с ноутбуком, смешное фото».\n"
                "Можно писать по-русски или по-английски.\n"
                "Отмена: «◀️ Назад».",
                reply_markup=GEN_KB,
            )
            return
        if text == "🆘 Техподдержка":
            ud["flow"] = "support"
            try:
                await context.bot.send_message(ADMIN_ID, f"🆘 {user.full_name} (id: {user.id}) открыл обращение.")
            except TelegramError:
                pass
            await message.reply_text("🛠 Режим поддержки. Пиши вопрос — он уйдёт админу.", reply_markup=SUPPORT_KB)
            return
        if text == "✍️ Добавить фразу":
            ud["flow"] = "phrase_menu"
            await message.reply_text("✍️ Что добавляем?", reply_markup=PHRASE_KB)
            return
        if text == "⬆️ Добавить верхнюю":
            ud["flow"] = "wait_top"
            await message.reply_text("⬆️ Пришли ВЕРХНЮЮ фразу:", reply_markup=PHRASE_KB)
            return
        if text == "⬇️ Добавить нижнюю":
            ud["flow"] = "wait_bottom"
            await message.reply_text("⬇️ Пришли НИЖНЮЮ фразу:", reply_markup=PHRASE_KB)
            return
        if not is_admin(user.id):
            if is_private:
                await message.reply_text("ℹ️ Используй кнопки меню или /start.", reply_markup=main_kb(user.id))
            return
        if text == "🛠 Админка":
            ud["flow"] = "admin"
            await message.reply_text("🛠 Админ-панель:", reply_markup=ADMIN_KB)
            return
        if text == "🗑 Удалить шаблон":
            ud["flow"] = "admin_delete"
            await show_template_page(context, chat.id, "del", 0)
            return
        if text == "🗑 Очистить всё":
            ud["pending"] = {"type": "clear"}
            await message.reply_text("⚠️ Удалить ВСЕ шаблоны и превью? Это необратимо.", reply_markup=CONFIRM_KB)
            return
        if text == "🗑 Удалить фразу":
            ud["flow"] = "phrase_admin"
            await send_phrase_menu(context, chat.id)
            return
        if text == "📊 Статистика":
            await message.reply_text(
                "📊 Статистика\n"
                f"🖼 Шаблонов: {len(await asyncio.to_thread(list_templates))}\n"
                f"⬆️ Верхних фраз: {len(get_phrases('top'))}\n"
                f"⬇️ Нижних фраз: {len(get_phrases('bottom'))}\n"
                f"👥 Пользователей: {len(await asyncio.to_thread(users_snapshot))}",
                reply_markup=ADMIN_KB,
            )
            return
        if text == "📢 Рассылка":
            ud["flow"] = "wait_broadcast"
            await message.reply_text(f"📢 Пришли текст рассылки (до {BROADCAST_MAX_LEN} символов).", reply_markup=ADMIN_KB)
            return
        if is_private:
            await message.reply_text("ℹ️ Используй кнопки меню или /start.", reply_markup=main_kb(user.id))


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q or not q.data or not q.message:
        return
    chat_id = q.message.chat.id
    user_id = q.from_user.id if q.from_user else 0
    if q.message.chat.type not in ("private", "group", "supergroup"):
        await q.answer()
        return
    data = q.data
    parts = data.split(":")
    cmd = parts[0]
    ud = context.user_data
    if cmd == "ai_rate" and len(parts) == 3:
        rating, meme_id = parts[1], parts[2]
        ai_data = context.bot_data.get("ai_memories", {}).pop(meme_id, None)
        if not ai_data:
            await q.answer("Этот мем уже оценён.")
            return
        memory = load_ai_memory()
        key = "likes" if rating == "like" else "dislikes"
        memory.setdefault(key, []).append({"top": ai_data["top"], "bottom": ai_data["bottom"]})
        save_ai_memory(memory)
        try:
            await q.edit_message_reply_markup(reply_markup=None)
        except TelegramError:
            pass
        await q.answer("🧠 Нейросеть запомнила твою оценку!")
        return
    if cmd in {"genadd", "gendel", "genagain"} and len(parts) == 2:
        gen_id = parts[1]
        gens = context.bot_data.setdefault("generated_images", LRUDict(500))
        path = gens.get(gen_id)
        if not path or not os.path.exists(path):
            await q.answer("Картинка устарела, сгенерируй заново.", show_alert=True)
            return
        if cmd == "genadd":
            gens.pop(gen_id, None)
            try:
                await q.edit_message_reply_markup(reply_markup=None)
            except TelegramError:
                pass
            await q.answer("✅ Добавлено в шаблоны!")
            await context.bot.send_message(
                chat_id,
                "✅ Картинка добавлена в шаблоны. Доступна в «🎭 Шаблоны» и «🖼 Выбрать шаблон».",
                reply_markup=main_kb(user_id),
            )
            return
        if cmd == "gendel":
            gens.pop(gen_id, None)
            await asyncio.to_thread(delete_template_file, path)
            try:
                await q.edit_message_reply_markup(reply_markup=None)
            except TelegramError:
                pass
            await q.answer("🗑 Картинка удалена.")
            return
        if cmd == "genagain":
            prompt = ud.get("last_gen_prompt") or (q.message.caption or "").lstrip("🎨 ").split("\n")[0]
            if not prompt:
                await q.answer("Описание не найдено, отправь его заново.", show_alert=True)
                return
            await q.answer("🔁 Рисую новый вариант…")
            ud["last_gen_prompt"] = prompt
            await generate_picture(context, chat_id, ud, q.message, prompt)
            return
    if cmd == "noop":
        await q.answer()
        return
    admin_cmds = {"del", "delnav", "delcancel", "ph", "phnav", "phdel", "phmenu", "clearall"}
    if cmd in admin_cmds and not is_admin(user_id):
        await q.answer("Нет доступа.", show_alert=True)
        return
    await q.answer()
    async with chat_lock(context, chat_id):
        try:
            if cmd in {"pick", "del"} and len(parts) == 3:
                mode = "pick" if cmd == "pick" else "del"
                page, offset = int(parts[1]), int(parts[2])
                files = await asyncio.to_thread(list_templates)
                index = page * PAGE_SIZE + offset
                if index >= len(files) or not os.path.exists(files[index]):
                    await finish_callback(q, context, "⚠️ Шаблон уже удалён. Открой список заново.")
                    return
                path = files[index]
                if mode == "pick":
                    ud["chosen_template"] = path
                    ud["flow"] = "caption"
                    await safe_delete(context, chat_id, q.message.message_id)
                    await context.bot.send_message(
                        chat_id,
                        f"✅ Выбран шаблон №{index + 1}. Какую надпись поставить?\n" + CAPTION_HELP,
                        reply_markup=CAPTION_KB,
                    )
                else:
                    ud["pending"] = {"type": "template", "path": path, "number": index + 1}
                    await finish_callback(q, context, f"🗑 Удалить шаблон №{index + 1}?", CONFIRM_KB)
                return
            if cmd in {"picknav", "delnav"} and len(parts) == 2:
                mode = "pick" if cmd == "picknav" else "del"
                await show_template_page(context, chat_id, mode, int(parts[1]), q.message.message_id)
                return
            if cmd in {"pickcancel", "delcancel"}:
                ud.pop("chosen_template", None)
                ud["flow"] = None
                await safe_delete(context, chat_id, q.message.message_id)
                await go_home(context, chat_id, user_id)
                return
            if cmd == "phmenu":
                await safe_delete(context, chat_id, q.message.message_id)
                await send_phrase_menu(context, chat_id)
                return
            if cmd == "ph" and len(parts) == 3:
                kind = parts[1] if parts[1] in {"top", "bottom"} else "top"
                text, kb = phrase_page_view(kind, int(parts[2]))
                await finish_callback(q, context, text, kb)
                return
            if cmd == "phnav" and len(parts) == 3:
                text, kb = phrase_page_view(parts[1], int(parts[2]))
                await finish_callback(q, context, text, kb)
                return
            if cmd == "phdel" and len(parts) == 3:
                kind = parts[1]
                index = int(parts[2])
                phrases = get_phrases(kind)
                if index < 0 or index >= len(phrases):
                    await finish_callback(q, context, "⚠️ Фраза уже изменена.")
                    return
                phrase = phrases[index]
                ud["pending"] = {"type": "phrase", "kind": kind, "phrase": phrase}
                await finish_callback(q, context, f"🗑 Удалить фразу?\n«{phrase}»", CONFIRM_KB)
                return
            if cmd == "clearall":
                ud["pending"] = {"type": "clear"}
                await finish_callback(q, context, "⚠️ Удалить все шаблоны?", CONFIRM_KB)
                return
            if cmd == "cancelconfirm":
                ud.pop("pending", None)
                await finish_callback(q, context, "❌ Отменено.")
                return
            if cmd == "confirm":
                pending = ud.pop("pending", None)
                if not pending:
                    await finish_callback(q, context, "⚠️ Нет действия для подтверждения.")
                    return
                kind_action = pending.get("type")
                if kind_action == "phrase":
                    ok = await asyncio.to_thread(delete_phrase_sync, pending["kind"], pending["phrase"])
                    text, kb = phrase_page_view(pending["kind"], 0)
                    await finish_callback(
                        q, context,
                        ("✅ Удалено.\n" if ok else "⚠️ Фраза уже удалена.\n") + text, kb,
                    )
                elif kind_action == "template":
                    await asyncio.to_thread(delete_template_file, pending["path"])
                    await finish_callback(
                        q, context,
                        f"🗑 Шаблон №{pending['number']} удалён.\nОсталось: {len(await asyncio.to_thread(list_templates))}",
                    )
                elif kind_action == "clear":
                    count = await asyncio.to_thread(clear_all_templates)
                    await finish_callback(q, context, f"🗑 Удалено шаблонов: {count}.")
                elif kind_action == "broadcast":
                    spawn_background(context, broadcast_worker(context, chat_id, pending["text"]))
                    await finish_callback(q, context, "📢 Рассылка запущена в фоне.")
                else:
                    await finish_callback(q, context, "⚠️ Неизвестное действие.")
                return
            if cmd == "home":
                ud.clear()
                await safe_delete(context, chat_id, q.message.message_id)
                await go_home(context, chat_id, user_id)
                return
        except (ValueError, IndexError, KeyError):
            await finish_callback(q, context, "⚠️ Некорректная кнопка. Открой меню заново.")
        except Exception:
            await finish_callback(q, context, "❌ Внутренняя ошибка. Попробуй /start.")


async def memo_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text("❌ Произошла внутренняя ошибка. Попробуй /start.")
        except TelegramError:
            pass


def remove_old_logs() -> None:
    for name in ("bot.log", "bot.log.1", "bot.log.2", "bot.log.3", "bot.log.4", "bot.log.5"):
        try:
            path = os.path.join(DATA_DIR, name)
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def prepare_environment() -> None:
    setup_dirs()
    remove_old_logs()


async def health_handler(request):
    from aiohttp import web
    return web.Response(text="OK", content_type="text/plain")


async def start_health_server() -> None:
    port_env = os.getenv("PORT", "").strip()
    if not port_env:
        return
    try:
        port = int(port_env)
    except ValueError:
        return
    from aiohttp import web
    app = web.Application()
    app.router.add_get("/", health_handler)
    app.router.add_get("/health", health_handler)
    app.router.add_get("/ping", health_handler)
    app.router.add_head("/", health_handler)
    app.router.add_head("/health", health_handler)
    app.router.add_route("*", "/{tail:.*}", health_handler)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=port)
    await site.start()
    print(f"✅ HTTP-сервер для UptimeRobot слушает порт {port} (пути /, /health, /ping).")


def build_memo_app() -> Application:
    app = (
        Application.builder()
        .token(MEMO_TOKEN)
        .concurrent_updates(8)
        .read_timeout(30)
        .write_timeout(30)
        .connect_timeout(15)
        .pool_timeout(30)
        .build()
    )
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("meme", cmd_meme))
    app.add_handler(CommandHandler("ai", cmd_ai))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(~filters.COMMAND, on_message))
    app.add_error_handler(memo_error_handler)
    return app


async def run_all() -> None:
    await asyncio.to_thread(prepare_environment)
    await start_health_server()
    apps = [build_memo_app()]
    initialized: list[Application] = []
    try:
        for app in apps:
            await app.initialize()
            initialized.append(app)
        await apps[0].bot.set_my_commands([
            BotCommand("start", "Запустить бота"),
            BotCommand("help", "Справка по боту"),
            BotCommand("meme", "Мем: /meme или /meme ВЕРХ | НИЗ"),
            BotCommand("ai", "Нейро-мем по случайному шаблону"),
        ])
        for app in apps:
            await app.start()
            await app.updater.start_polling(
                drop_pending_updates=True,
                allowed_updates=Update.ALL_TYPES,
            )
        print("✅ Мемофей запущен. Остановка — Ctrl+C.")
        await asyncio.Event().wait()
    finally:
        for app in reversed(initialized):
            try:
                if app.updater and app.updater.running:
                    await app.updater.stop()
            except Exception:
                pass
            try:
                if app.running:
                    await app.stop()
            except Exception:
                pass
            try:
                await app.shutdown()
            except Exception:
                pass


def main() -> None:
    if not MEMO_TOKEN or not TOKEN_PATTERN.match(MEMO_TOKEN):
        print("❌ Проверьте токен бота в начале файла.")
        return
    try:
        asyncio.run(run_all())
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()