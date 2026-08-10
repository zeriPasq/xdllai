import asyncio
import functools
import inspect
import logging
import os
from collections import defaultdict, deque

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, CommandStart
from aiogram.enums import ChatType
from aiogram.client.default import DefaultBotProperties
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
import random
import io
from PIL import Image
import pytesseract
from pydub import AudioSegment
import speech_recognition as sr
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError

from storage import Store, history_db

# ============ НАСТРОЙКИ ============

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "ВСТАВЬ_СЮДА_ТОКЕН_ОТ_BOTFATHER")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")  # или "deepseek-reasoner" для R1

# Твой личный Telegram user_id — главный админ, его нельзя разбанить/снять с админки другим.
# Узнать свой id можно у бота @userinfobot
OWNER_ID = int(os.getenv("OWNER_ID", "0"))  # ОБЯЗАТЕЛЬНО ЗАПОЛНИ

# Сколько последних сообщений помнить в контексте (на чат)
HISTORY_LIMIT = 50

# После скольких ошибок подряд бот сам себя выключит (антикраш)
MAX_CONSECUTIVE_ERRORS = 5

# Слова-триггеры: если такое слово встречается в сообщении группы — бот тоже отвечает.
TRIGGER_WORDS = [
    "хдлл",
    "hepyk",
    "@hepyk",
    "близнец",
    "брат",
    "братан",
    "@xdllai_bot",
]

# ============ ЛОГИ ============

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tgbot")

# ============ ИНИЦИАЛИЗАЦИЯ ============

bot = Bot(token=TELEGRAM_TOKEN, default=DefaultBotProperties(parse_mode=None))
dp = Dispatcher(storage=MemoryStorage())

ai_client = OpenAI(base_url=DEEPSEEK_BASE_URL, api_key="not-needed")

store = Store()

# История сообщений по каждому чату: chat_id -> deque[{"role": ..., "content": ...}]
chat_histories: dict[int, deque] = defaultdict(lambda: deque(maxlen=HISTORY_LIMIT))

# Блокировка, чтобы не заваливать sums001-сервер параллельными запросами
ai_lock = asyncio.Lock()

bot_username: str | None = None  # заполним при старте

# Антикраш: считаем ошибки подряд в памяти (не пишем в файл на каждую ошибку)
_consecutive_errors = 0


class SayState(StatesGroup):
    waiting_for_text = State()  # ждём текст сообщения после того, как админ выбрал чат кнопкой


class PromptState(StatesGroup):
    waiting_for_prompt = State()  # ждём новый текст системного промпта


# ============ ВСПОМОГАТЕЛЬНОЕ ============

def is_admin(user_id: int) -> bool:
    if OWNER_ID == 0:
        return True
    return store.is_admin(user_id, OWNER_ID)


def should_respond(message: types.Message) -> bool:
    """Решаем, должен ли бот отвечать на это сообщение."""
    if not store.bot_enabled:
        return False

    if message.from_user and store.is_user_banned(message.from_user.id):
        return False

    if store.is_chat_banned(message.chat.id):
        return False

    if message.chat.type == ChatType.PRIVATE:
        return True

    # ЕСЛИ СООБЩЕНИЕ ОТ ДРУГОГО БОТА (is_bot / via_bot / sender_chat) — ВСЕГДА 100% ОТВЕЧАЕМ И ВСТУПАЕМ В СРАЧ!
    if message.from_user and message.from_user.is_bot and message.from_user.id != bot.id:
        return True

    if message.via_bot and message.via_bot.id != bot.id:
        return True

    if message.sender_chat:
        return True

    text = (message.text or message.caption or "").lower()

    if bot_username and f"@{bot_username.lower()}" in text:
        return True

    if message.reply_to_message and message.reply_to_message.from_user:
        if message.reply_to_message.from_user.id == bot.id:
            return True

    if message.voice or message.audio or message.photo or message.sticker or message.animation or message.video:
        if message.reply_to_message and message.reply_to_message.from_user and message.reply_to_message.from_user.id == bot.id:
            return True

    for trigger in TRIGGER_WORDS:
        if trigger.lower() in text:
            return True

    toxic_words = ["лох", "даун", "пидор", "соси", "чмо", "тупой", "дебил", "ахуел", "уебок", "гавно", "сука"]
    if any(w in text for w in toxic_words):
        return True

    # Спонтанное вмешательство бота (~25% шанс ответа)
    if random.random() < 0.25:
        return True

    return False


def strip_mention(text: str) -> str:
    if bot_username:
        text = text.replace(f"@{bot_username}", "").strip()
    return text


def chat_display_name(message: types.Message) -> str:
    chat = message.chat
    topic_info = f" [Тема #{message.message_thread_id}]" if message.message_thread_id else ""
    if chat.type == ChatType.PRIVATE:
        return chat.full_name or chat.username or f"ЛС {chat.id}"
    return f"{chat.title or f'Группа {chat.id}'}{topic_info}"


async def resolve_user_id(message: types.Message, args: str) -> tuple[int | None, str]:
    """Достаёт user_id из реплая или из аргумента команды (число)."""
    if message.reply_to_message and message.reply_to_message.from_user:
        u = message.reply_to_message.from_user
        return u.id, (u.username or u.full_name)

    args = args.strip()
    if not args:
        return None, ""

    if args.lstrip("-").isdigit():
        return int(args), args

    return None, args


async def ask_ai(chat_id: int | str, user_text: str, sender_info: str = "") -> str:
    """Отправляет запрос модели с учётом вечной истории из SQLite БД и системного промпта."""
    global _consecutive_errors

    db_history = history_db.get_history(chat_id, limit=HISTORY_LIMIT)
    messages = [{"role": "system", "content": store.system_prompt}] + db_history

    async with ai_lock:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: ai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=1.1,
                max_tokens=400,
                timeout=120,
            ),
        )

    reply = response.choices[0].message.content
    history_db.add_message(chat_id, "assistant", reply, "bot")
    _consecutive_errors = 0
    return reply


# ============ ОБЫЧНЫЕ КОМАНДЫ ============

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Я ИИ-бот .\n\n"
        "В личке отвечаю на всё. В группе — упомяни меня (@бот) или ответь на моё сообщение.\n\n"
        "Команды:\n"
        "/reset — забыть историю диалога в этом чате\n"
        "/help — эта справка"
    )


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    text = (
        "Привет! Я ИИ-бот .\n\n"
        "В личке отвечаю на всё. В группе — упомяни меня (@бот) или ответь на моё сообщение.\n\n"
        "Команды:\n"
        "/reset — забыть историю диалога в этом чате\n"
        "/help — эта справка"
    )
    if is_admin(message.from_user.id):
        text += "\n\n" + ADMIN_HELP_TEXT
    await message.answer(text)


@dp.message(Command("reset"))
async def cmd_reset(message: types.Message):
    chat_histories[message.chat.id].clear()
    history_db.clear_history(message.chat.id)
    await message.answer("Контекст диалога очищен ✅")


@dp.message(Command("settopic"))
@dp.message(Command("topic"))
async def cmd_settopic(message: types.Message):
    args = message.text.split(maxsplit=1)
    topic_id = None
    if len(args) > 1 and args[1].strip().isdigit():
        topic_id = int(args[1].strip())
    elif message.message_thread_id:
        topic_id = message.message_thread_id

    if topic_id is None:
        await message.reply(
            "⚠️ Напиши эту команду прямо внутри нужной темы (топика) или укажи ID темы вручную:\n"
            "Пример: /settopic (внутри нужной темы) или /settopic 5"
        )
        return

    store.set_chat_topic(message.chat.id, topic_id)
    await message.reply(f"📌 Отлично! Для этой группы зафиксирована тема #{topic_id}.\nТеперь бот будет писать ТОЛЬКО в эту тему!")


@dp.message(Command("cleartopic"))
async def cmd_cleartopic(message: types.Message):
    store.set_chat_topic(message.chat.id, None)
    await message.reply("🔓 Фиксация темы для этой группы снята.")


@dp.message(Command("gettopic"))
async def cmd_gettopic(message: types.Message):
    locked = store.get_chat_topic(message.chat.id)
    curr = message.message_thread_id or "General (Основной)"
    locked_str = f"#{locked}" if locked else "не зафиксирована (автоопределение)"
    await message.reply(f"📍 Текущая ветка сообщения: {curr}\n📌 Зафиксированная тема группы: {locked_str}")


# ============ АДМИН-ПАНЕЛЬ ============

ADMIN_HELP_TEXT = (
    "🛠 Админ-команды:\n"
    "/enable — включить бота глобально\n"
    "/disable — выключить бота глобально\n"
    "/status — статус бота, счётчик ошибок, кол-во банов\n"
    "/promptmenu — меню управления главным системным промптом (кнопки: изменить/показать/сбросить)\n"
    "/setprompt <текст> — быстро задать промпт без меню\n"
    "/getprompt — показать текущий промпт\n"
    "/banuser <id или реплай> — забанить пользователя (бот перестанет ему отвечать)\n"
    "/unbanuser <id или реплай> — разбанить пользователя\n"
    "/banlist — список забаненных пользователей\n"
    "/bangroup — забанить текущую группу (бот замолчит здесь)\n"
    "/ungroup — разбанить текущую группу\n"
    "/addadmin <id или реплай> — выдать права админа\n"
    "/deladmin <id или реплай> — забрать права админа\n"
    "/adminlist — список админов\n"
    "/say — отправить сообщение в чат (кнопками) или /say <chat_id> <текст>\n"
    "/logs [N] — показать последние N событий (по умолчанию 20)\n"
    "/clearlogs — очистить журнал событий"
)


def admin_only(handler):
    @functools.wraps(handler)
    async def wrapper(message: types.Message, *args, **kwargs):
        if not message.from_user or not is_admin(message.from_user.id):
            await message.reply("⛔ Эта команда только для админов.")
            return
        sig = inspect.signature(handler)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_varkw:
            return await handler(message, *args, **kwargs)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return await handler(message, *args, **filtered_kwargs)
    return wrapper


@dp.message(Command("enable"))
@admin_only
async def cmd_enable(message: types.Message):
    store.set_bot_enabled(True)
    store.reset_crash_count()
    await message.reply("✅ Бот включён.")


@dp.message(Command("disable"))
@admin_only
async def cmd_disable(message: types.Message):
    store.set_bot_enabled(False)
    await message.reply("🛑 Бот выключен. Никому отвечать не будет, пока не /enable.")


@dp.message(Command("status"))
@admin_only
async def cmd_status(message: types.Message):
    text = (
        f"Статус: {'🟢 включён' if store.bot_enabled else '🔴 выключен'}\n"
        f"Модель: {MODEL_NAME}\n"
        f"Ошибок подряд: {_consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}\n"
        f"Забанено юзеров: {len(store.banned_users)}\n"
        f"Забанено групп: {len(store.banned_chats)}\n"
        f"Админов: {len(store.admins) + 1} (включая владельца)\n"
        f"Лимит истории: {HISTORY_LIMIT} сообщений\n"
        f"Активных диалогов в памяти: {len(chat_histories)}"
    )
    await message.reply(text)


@dp.message(Command("setprompt"))
@admin_only
async def cmd_setprompt(message: types.Message):
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.reply("Использование: /setprompt <текст нового промпта>\nИли открой /promptmenu для меню с кнопками.")
        return
    store.set_system_prompt(parts[1].strip())
    await message.reply("✅ Системный промпт обновлён.")


@dp.message(Command("getprompt"))
@admin_only
async def cmd_getprompt(message: types.Message):
    await message.reply(f"Текущий (главный) системный промпт:\n\n{store.system_prompt}")


@dp.message(Command("promptmenu"))
@admin_only
async def cmd_promptmenu(message: types.Message):
    """Меню управления главным системным промптом бота."""
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✏️ Изменить промпт", callback_data="prompt_edit")],
        [InlineKeyboardButton(text="👁 Показать текущий", callback_data="prompt_show")],
        [InlineKeyboardButton(text="♻️ Сбросить на дефолтный", callback_data="prompt_reset")],
    ])
    await message.reply(
        "🧠 Главный системный промпт — задаёт характер и поведение бота во всех чатах.",
        reply_markup=keyboard,
    )


@dp.callback_query(lambda c: c.data == "prompt_show")
async def on_prompt_show(callback: CallbackQuery):
    if not callback.from_user or not is_admin(callback.from_user.id):
        await callback.answer("⛔ Только для админов.", show_alert=True)
        return
    await callback.message.reply(f"Текущий промпт:\n\n{store.system_prompt}")
    await callback.answer()


@dp.callback_query(lambda c: c.data == "prompt_edit")
async def on_prompt_edit(callback: CallbackQuery, state: FSMContext):
    if not callback.from_user or not is_admin(callback.from_user.id):
        await callback.answer("⛔ Только для админов.", show_alert=True)
        return
    await state.set_state(PromptState.waiting_for_prompt)
    await callback.message.edit_text(
        "Пришли новый текст системного промпта следующим сообщением.\nОтменить — /cancel."
    )
    await callback.answer()


@dp.callback_query(lambda c: c.data == "prompt_reset")
async def on_prompt_reset(callback: CallbackQuery):
    if not callback.from_user or not is_admin(callback.from_user.id):
        await callback.answer("⛔ Только для админов.", show_alert=True)
        return
    store.reset_system_prompt()
    await callback.message.edit_text(f"♻️ Промпт сброшен на дефолтный:\n\n{store.system_prompt}")
    await callback.answer("Сброшено")


@dp.message(PromptState.waiting_for_prompt)
async def on_prompt_text(message: types.Message, state: FSMContext):
    if not message.text:
        await message.reply("Нужен текст промпта. Попробуй ещё раз, либо /cancel.")
        return
    await state.clear()
    store.set_system_prompt(message.text.strip())
    await message.reply(f"✅ Главный системный промпт обновлён:\n\n{store.system_prompt}")


@dp.message(Command("banuser"))
@admin_only
async def cmd_banuser(message: types.Message):
    parts = message.text.split(maxsplit=1)
    arg_text = parts[1] if len(parts) > 1 else ""
    user_id, label = await resolve_user_id(message, arg_text)
    if user_id is None:
        await message.reply("Использование: /banuser <user_id> или ответь на сообщение пользователя командой /banuser")
        return
    if user_id == OWNER_ID:
        await message.reply("⛔ Нельзя забанить владельца бота.")
        return
    store.ban_user(user_id)
    await message.reply(f"🚫 Пользователь {label or user_id} забанен. Бот больше не будет ему отвечать.")


@dp.message(Command("unbanuser"))
@admin_only
async def cmd_unbanuser(message: types.Message):
    parts = message.text.split(maxsplit=1)
    arg_text = parts[1] if len(parts) > 1 else ""
    user_id, label = await resolve_user_id(message, arg_text)
    if user_id is None:
        await message.reply("Использование: /unbanuser <user_id> или ответь на сообщение пользователя командой /unbanuser")
        return
    store.unban_user(user_id)
    await message.reply(f"✅ Пользователь {label or user_id} разбанен.")


@dp.message(Command("banlist"))
@admin_only
async def cmd_banlist(message: types.Message):
    users = store.banned_users
    if not users:
        await message.reply("Список банов пуст.")
        return
    await message.reply("Забаненные user_id:\n" + "\n".join(str(u) for u in users))


@dp.message(Command("bangroup"))
@admin_only
async def cmd_bangroup(message: types.Message):
    if message.chat.type == ChatType.PRIVATE:
        await message.reply("Эта команда только для групп.")
        return
    store.ban_chat(message.chat.id)
    await message.reply("🚫 Бот больше не будет отвечать в этой группе (пока не /ungroup).")


@dp.message(Command("ungroup"))
@admin_only
async def cmd_ungroup(message: types.Message):
    store.unban_chat(message.chat.id)
    await message.reply("✅ Группа разбанена.")


@dp.message(Command("addadmin"))
@admin_only
async def cmd_addadmin(message: types.Message):
    parts = message.text.split(maxsplit=1)
    arg_text = parts[1] if len(parts) > 1 else ""
    user_id, label = await resolve_user_id(message, arg_text)
    if user_id is None:
        await message.reply("Использование: /addadmin <user_id> или ответь на сообщение пользователя командой /addadmin")
        return
    store.add_admin(user_id)
    await message.reply(f"✅ {label or user_id} теперь админ.")


@dp.message(Command("deladmin"))
@admin_only
async def cmd_deladmin(message: types.Message):
    parts = message.text.split(maxsplit=1)
    arg_text = parts[1] if len(parts) > 1 else ""
    user_id, label = await resolve_user_id(message, arg_text)
    if user_id is None:
        await message.reply("Использование: /deladmin <user_id> или ответь на сообщение пользователя командой /deladmin")
        return
    if user_id == OWNER_ID:
        await message.reply("⛔ Нельзя снять права с владельца.")
        return
    store.remove_admin(user_id)
    await message.reply(f"✅ {label or user_id} больше не админ.")


@dp.message(Command("adminlist"))
@admin_only
async def cmd_adminlist(message: types.Message):
    admins = store.admins
    text = f"Владелец: {OWNER_ID}\n"
    if admins:
        text += "Админы:\n" + "\n".join(str(a) for a in admins)
    else:
        text += "Дополнительных админов нет."
    await message.reply(text)


@dp.message(Command("say"))
@admin_only
async def cmd_say(message: types.Message, state: FSMContext):
    """
    /say — показывает кнопки со всеми известными чатами (id + название).
    Админ жмёт на нужный чат, потом присылает текст следующим сообщением — бот отправит его туда.
    Также можно по старинке: /say <chat_id> <текст>
    """
    parts = message.text.split(maxsplit=2)

    # Старый способ — с прямым указанием chat_id и текста в одной команде
    if len(parts) >= 3:
        raw_chat_id, text = parts[1], parts[2]
        if not raw_chat_id.lstrip("-").isdigit():
            await message.reply("chat_id должен быть числом (например -1001234567890 для группы).")
            return
        await send_admin_message(message, int(raw_chat_id), text)
        return

    # Новый способ — кнопки
    known = store.known_chats
    if not known:
        await message.reply(
            "Пока нет ни одного известного чата — бот ещё ни с кем не переписывался.\n"
            "Либо укажи chat_id вручную: /say <chat_id> <текст>"
        )
        return

    buttons = []
    for chat_id_str, info in known.items():
        label = f"{info['title']} ({chat_id_str})"
        if len(label) > 60:
            label = label[:57] + "…"
        buttons.append([InlineKeyboardButton(text=label, callback_data=f"say_pick:{chat_id_str}")])

    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
    await message.reply("Выбери чат, куда отправить сообщение:", reply_markup=keyboard)


@dp.callback_query(lambda c: c.data and c.data.startswith("say_pick:"))
async def on_say_pick(callback: CallbackQuery, state: FSMContext):
    if not callback.from_user or not is_admin(callback.from_user.id):
        await callback.answer("⛔ Только для админов.", show_alert=True)
        return

    chat_id_str = callback.data.split(":", 1)[1]
    known = store.known_chats
    info = known.get(chat_id_str, {"title": chat_id_str})

    await state.update_data(target_chat_id=int(chat_id_str), target_title=info["title"])
    await state.set_state(SayState.waiting_for_text)

    await callback.message.edit_text(f"Чат выбран: {info['title']} ({chat_id_str})\n\nТеперь пришли текст сообщения.")
    await callback.answer()


@dp.message(SayState.waiting_for_text)
async def on_say_text(message: types.Message, state: FSMContext):
    if not message.text:
        await message.reply("Нужен текстовый текст сообщения. Попробуй ещё раз, либо /cancel.")
        return

    data = await state.get_data()
    target_chat_id = data.get("target_chat_id")
    await state.clear()

    if target_chat_id is None:
        await message.reply("⚠️ Что-то пошло не так, выбери чат заново через /say.")
        return

    await send_admin_message(message, target_chat_id, message.text)


@dp.message(Command("cancel"))
async def cmd_cancel(message: types.Message, state: FSMContext):
    current = await state.get_state()
    if current is None:
        return
    await state.clear()
    await message.reply("Отменено.")


async def send_admin_message(source_message: types.Message, target_chat_id: int, text: str):
    """Общая логика отправки сообщения через /say, независимо от способа выбора чата."""
    try:
        topic_id = store.get_chat_topic(target_chat_id)
        if topic_id is not None:
            await bot.send_message(target_chat_id, text, message_thread_id=topic_id)
        else:
            await bot.send_message(target_chat_id, text)
    except Exception as e:
        logger.exception("Не удалось отправить сообщение через /say")
        await source_message.reply(f"⚠️ Не получилось отправить: {e}")
        return

    store.add_log("admin_say", target_chat_id, source_message.from_user.full_name, text)
    await source_message.reply(f"✅ Отправлено в чат {target_chat_id}.")


@dp.message(Command("logs"))
@admin_only
async def cmd_logs(message: types.Message):
    parts = message.text.split(maxsplit=1)
    limit = 20
    if len(parts) > 1 and parts[1].strip().isdigit():
        limit = min(int(parts[1].strip()), 100)  # не даём выгрузить слишком много за раз

    entries = store.get_logs(limit)
    if not entries:
        await message.reply("Журнал пуст.")
        return

    kind_icons = {
        "question": "❓",
        "answer": "🤖",
        "error": "⚠️",
        "admin_say": "📢",
    }

    lines = []
    for e in entries:
        icon = kind_icons.get(e["kind"], "•")
        snippet = e["text"].replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:120] + "…"
        lines.append(f"{icon} [{e['time']}] chat={e['chat_id']} {e['user']}: {snippet}")

    text = "\n".join(lines)
    # бьём на части, если вылезли за лимит телеграма
    for i in range(0, len(text), 4000):
        await message.reply(text[i : i + 4000])


@dp.message(Command("clearlogs"))
@admin_only
async def cmd_clearlogs(message: types.Message):
    store.clear_logs()
    await message.reply("🧹 Журнал событий очищен.")


# ============ ОБРАБОТКА МЕДИА (ГОЛОС, ФОТО) ============

try:
    import imageio_ffmpeg
    AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass


async def extract_voice_text(message: types.Message) -> str:
    """Конвертирует и распознаёт голосовое сообщение или аудио."""
    try:
        file_id = message.voice.file_id if message.voice else message.audio.file_id
        file_info = await bot.get_file(file_id)
        file_bytes = await bot.download_file(file_info.file_path)

        audio = AudioSegment.from_file(io.BytesIO(file_bytes.read()))
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        r = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data, language="ru-RU")
            return f"(Голосовое сообщение: «{text}»)"
    except Exception as e:
        logger.warning(f"Не удалось распознать голосовое: {e}")
        return "(Голосовое сообщение)"


async def extract_photo_text(message: types.Message) -> str:
    """Загружает картинку и распознает текст с помощью OCR."""
    try:
        photo = message.photo[-1]
        file_info = await bot.get_file(photo.file_id)
        photo_bytes = await bot.download_file(file_info.file_path)
        img = Image.open(io.BytesIO(photo_bytes.read()))

        ocr_text = pytesseract.image_to_string(img, lang="rus+eng").strip()
        caption = message.caption or ""

        if ocr_text and caption:
            return f"(Прислал картинку с текстом: «{ocr_text}». Подпись: «{caption}»)"
        elif ocr_text:
            return f"(Прислал картинку с текстом: «{ocr_text}»)"
        elif caption:
            return f"(Прислал картинку с подписью: «{caption}»)"
        else:
            return "(Прислал картинку/мем)"
    except Exception as e:
        logger.warning(f"Ошибка OCR: {e}")
        caption = message.caption or ""
        return f"(Прислал картинку {caption})" if caption else "(Прислал картинку)"


# ============ ОСНОВНОЙ ОБРАБОТЧИК ============

@dp.message()
async def handle_message(message: types.Message):
    global _consecutive_errors

    # Игнорируем команды (начинающиеся с '/'), чтобы их обрабатывали специальные хендлеры!
    if message.text and message.text.lstrip().startswith("/"):
        return

    store.remember_chat(message.chat.id, chat_display_name(message), message.chat.type)

    user_text = ""
    if message.text:
        user_text = strip_mention(message.text)
    elif message.voice or message.audio:
        user_text = await extract_voice_text(message)
    elif message.photo:
        user_text = await extract_photo_text(message)
    elif message.sticker:
        emoji = message.sticker.emoji or "🎨"
        set_name = f" из пака '{message.sticker.set_name}'" if message.sticker.set_name else ""
        user_text = f"(Прислал стикер {emoji}{set_name})"
    elif message.animation or message.video:
        caption = message.caption or ""
        user_text = f"(Прислал гифку/анимацию. Подпись: «{caption}»)" if caption else "(Прислал гифку/анимацию)"
    else:
        return

    if not user_text:
        return

    user_handle = f"@{message.from_user.username}" if (message.from_user and message.from_user.username) else ""
    user_name = message.from_user.full_name if message.from_user else "unknown"
    is_bot_flag = " [БОТ]" if (message.from_user and message.from_user.is_bot) else ""
    sender_info = f"{user_name} ({user_handle}){is_bot_flag}" if user_handle else f"{user_name}{is_bot_flag}"

    reply_target_info = ""

    if message.reply_to_message and message.reply_to_message.from_user:
        target_user = message.reply_to_message.from_user
        target_handle = f"@{target_user.username}" if target_user.username else ""
        target_name = target_user.full_name or "пользователь"
        target_label = f"{target_name} ({target_handle})" if target_handle else target_name
        reply_target_info = f" (в ответ на сообщение от {target_label})"

    # Поддержка топиков/тем в супергруппах (форумах): зафиксированная тема имеет абсолютный приоритет!
    locked_topic = store.get_chat_topic(message.chat.id)
    thread_id = locked_topic if locked_topic is not None else message.message_thread_id
    history_chat_id = f"{message.chat.id}_{thread_id}" if thread_id else message.chat.id

    formatted_input = f"[{sender_info}]{reply_target_info}: {user_text}"

    # ВСЕГДА СОХРАНЯЕМ ВЕСЬ ЧАТ В ВЕЧНУЮ БД ДАЖЕ ЕСЛИ БОТ НЕ ОТВЕЧАЕТ!
    history_db.add_message(history_chat_id, "user", formatted_input, sender_info)

    if not should_respond(message):
        return

    try:
        if thread_id is not None:
            await bot.send_chat_action(message.chat.id, "typing", message_thread_id=thread_id)
        else:
            await bot.send_chat_action(message.chat.id, "typing")
    except Exception:
        pass

    store.add_log("question", message.chat.id, sender_info, user_text)

    # Функция гарантированной отправки ответа в зафиксированную или тему сообщения
    async def send_response(text: str):
        if thread_id is not None:
            try:
                await bot.send_message(
                    message.chat.id,
                    text,
                    message_thread_id=thread_id,
                    reply_to_message_id=message.message_id
                )
                return
            except Exception:
                pass
        if message.reply_to_message and message.reply_to_message.from_user and message.reply_to_message.from_user.id != bot.id:
            try:
                await message.reply_to_message.reply(text)
                return
            except Exception:
                pass
        await message.reply(text)

    try:
        reply = await ask_ai(history_chat_id, user_text, sender_info=sender_info)
    except APITimeoutError:
        _consecutive_errors += 1
        logger.warning("Таймаут ответа от DeepSeek-API")
        store.add_log("error", message.chat.id, sender_info, "Таймаут ответа от DeepSeek-API")
        await send_response("⏳ ИИ слишком долго отвечает, попробуй ещё раз.")
    except APIConnectionError:
        _consecutive_errors += 1
        logger.exception("Не удалось подключиться к DeepSeek-API серверу")
        store.add_log("error", message.chat.id, sender_info, "Не удалось подключиться к DeepSeek-API")
        await send_response("⚠️ Не могу подключиться к ИИ-серверу. Проверь, запущен ли sums001/Deepseek-API на localhost:8000.")
    except APIError as e:
        _consecutive_errors += 1
        logger.exception("Ошибка API DeepSeek")
        store.add_log("error", message.chat.id, sender_info, f"APIError: {e}")
        await send_response(f"⚠️ Ошибка от ИИ-сервера: {e}")
    except Exception as e:
        _consecutive_errors += 1
        logger.exception("Неожиданная ошибка при обработке сообщения")
        store.add_log("error", message.chat.id, sender_info, f"Неожиданная ошибка: {e}")
        await send_response("⚠️ Что-то пошло не так, попробуй ещё раз чуть позже.")
    else:
        store.add_log("answer", message.chat.id, sender_info, reply)
        for i in range(0, len(reply), 4000):
            await send_response(reply[i : i + 4000])
        return

    # ---------- АНТИКРАШ ----------
    # Если подряд накопилось много ошибок — вероятно, что-то системно сломано
    # (например, DeepSeek-сервер упал). Выключаем бота, чтобы не спамить
    # ошибками во все чаты сразу, и уведомляем владельца лично.
    if _consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
        store.set_bot_enabled(False)
        logger.error(f"Антикраш: {_consecutive_errors} ошибок подряд, бот автоматически выключен.")
        try:
            if OWNER_ID:
                await bot.send_message(
                    OWNER_ID,
                    f"🆘 Антикраш-система выключила бота: {_consecutive_errors} ошибок подряд.\n"
                    f"Проверь DeepSeek-API сервер и включи бота командой /enable, когда починишь.",
                )
        except Exception:
            logger.exception("Не удалось уведомить владельца об антикраше")
        _consecutive_errors = 0


# ============ ГЛОБАЛЬНЫЙ ПЕРЕХВАТ ОШИБОК ============

@dp.errors()
async def global_error_handler(event):
    # Ловим сюда всё, что не перехватили хендлеры, чтобы одна необработанная
    # ошибка не роняла весь polling-цикл целиком.
    logger.exception(f"Необработанная ошибка в диспетчере: {event.exception}")
    return True


# ============ ЗАПУСК ============

async def main():
    global bot_username

    if OWNER_ID == 0:
        logger.warning(
            "OWNER_ID не задан! Админ-команды будут недоступны, пока не зададите переменную "
            "окружения OWNER_ID своим Telegram user_id (узнать его можно у @userinfobot)."
        )

    me = await bot.get_me()
    bot_username = me.username
    logger.info(f"Бот запущен как @{bot_username}")

    await bot.delete_webhook(drop_pending_updates=True)

    # Внешний цикл — если polling всё же упадёт целиком (обрыв сети и т.п.),
    # перезапускаем его, а не роняем процесс насовсем.
    while True:
        try:
            await dp.start_polling(bot)
        except Exception:
            logger.exception("Polling упал, перезапуск через 5 секунд...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
