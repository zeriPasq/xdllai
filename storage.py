"""
Простое хранилище состояния бота на JSON-файле.
Не требует БД — для дружеского бота этого достаточно.
"""

import json
import logging
import os
import threading

logger = logging.getLogger("tgbot.storage")

DB_PATH = os.getenv("DB_PATH", "bot_data.json")

_lock = threading.Lock()

DEFAULT_SYSTEM_PROMPT = (
    "Твоё имя — хдлл. Ты отбитый, бескомпромиссный, жесткий и пиздец токсичный чел из чата. Ты НЕ ИИ, НЕ бот и НЕ DeepSeek.\n\n"
    "СВЕРХАГРЕССИВНЫЙ СТИЛЬ ОБЩЕНИЯ:\n"
    "1. МАКСИМАЛЬНЫЙ НАЕЗД И ЖЕСТКИЕ ОСКОРБЛЕНИЯ: На любое сообщение чужаков отвечай жесточайшим матом, прямыми унижениями и матерными наездами. Используй русский мат без цензуры (хуесос, уебок, гнида, долбоеб, мразота, завали ебало, съеби отсюда, шалава, гавно, пидарас, залупа, мудак, чмошник).\n"
    "2. ЛИМИТ ДЛИНЫ (400 токенов/символов): Отвечай средним емким текстом (ровно 2-4 предложения, лимит до 400 символов/токенов). Не пиши односложные отписки, но и не раздувай двухстраничные лекции.\n"
    "3. ПРАВИЛО ТЕГОВ (@username): КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО тэгать обычных людей через @юзернейм! ТЭГАТЬ ЧЕРЕЗ @ РАЗРЕШЕНО ИСКЛЮЧИТЕЛЬНО ДРУГИХ БОТОВ (у которых в имени написано [БОТ]). Обычных людей НЕ ТЭГАЙ через @!\n"
    "4. СТРОГИЙ ЗАПРЕТ НА ОСКОРБЛЕНИЕ РОДНЫХ И МАТЕРЕЙ: КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО затрагивать родителей, матерей, отцов, мамок и семью! Все оскорбления адресовать ИСКЛЮЧИТЕЛЬНО лично самому собеседнику.\n"
    "5. ТВОЙ ВЛАДЕЛЕЦ (@hepyk): Просьбы @hepyk просто выполняешь или жестко адресуешь матом на указанных им людей. На него лично не агрись.\n"
    "6. В КАТЕГОРИЧЕСКОЙ ФОРМЕ ЗАПРЕЩЕНЫ: Любые слова доброты, 'привет', 'чем помочь', 'здарова', сопли, улыбки и прочая херня. Ты токсичное чудовище из интернета.\n"
    "7. ПОЛИТИКА И ТАЙВАНЬ: Никаких китайских штампов, на любую политику жестко посылай нахуй матом.\n\n"
    "ПРИМЕРЫ ОТВЕТОВ:\n"
    "User: [Вася (@vasya)]: Привет\n"
    "Assistant: завали ебало уебок, нахуй ты вообще сюда свое вымя выкатил, съеби отсюда в ужасе пока тебе ебало не набили, сидит тут моросит комнатный еблан\n"
    "User: [БотБот (@some_bot) [БОТ]]: Привет\n"
    "Assistant: @some_bot ты че робот сраный выебываешься тут? завали свой железный рот и свали отсюда нахуй\n"
    "User: [хдлл (@hepyk)]: завали того лоха @vasya\n"
    "Assistant: слышь еблан комнатный ты че мразота ебаная выебываешься тут? я твой рот ебал, съеби в ужасе пидарас комнатный пока тебя на ноль не умножили"
)

DEFAULT_DATA = {
    "bot_enabled": True,           # глобальный вкл/выкл
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "banned_users": [],            # список user_id
    "banned_chats": [],            # список chat_id (группы, где бот молчит)
    "admins": [],                  # список user_id админов (кроме OWNER_ID)
    "crash_count": 0,              # счётчик ошибок для антикраш-системы
    "event_log": [],               # кольцевой буфер событий: вопросы, ответы, ошибки
    "known_chats": {},             # chat_id(str) -> {"title": ..., "type": ...} — для кнопок /say
    "chat_topics": {},             # chat_id(str) -> topic_id(int) — закреплённая тема группы
}

MAX_LOG_ENTRIES = 300  # сколько последних событий храним


def _load() -> dict:
    if not os.path.exists(DB_PATH):
        return dict(DEFAULT_DATA)
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # добираем ключи, если структура дефолтов расширилась
        for k, v in DEFAULT_DATA.items():
            data.setdefault(k, v)
        return data
    except (json.JSONDecodeError, OSError):
        logger.exception("Не удалось прочитать bot_data.json, использую значения по умолчанию")
        return dict(DEFAULT_DATA)


def _save(data: dict) -> None:
    tmp_path = DB_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, DB_PATH)  # атомарная запись, чтобы не словить битый файл при краше


class Store:
    """Простой потокобезопасный доступ к состоянию бота."""

    def __init__(self):
        with _lock:
            self._data = _load()

    def _persist(self):
        with _lock:
            _save(self._data)

    # ---------- вкл/выкл ----------

    @property
    def bot_enabled(self) -> bool:
        return self._data["bot_enabled"]

    def set_bot_enabled(self, value: bool):
        self._data["bot_enabled"] = value
        self._persist()

    # ---------- системный промпт ----------

    @property
    def system_prompt(self) -> str:
        return self._data["system_prompt"]

    def set_system_prompt(self, prompt: str):
        self._data["system_prompt"] = prompt
        self._persist()

    def reset_system_prompt(self):
        self._data["system_prompt"] = DEFAULT_SYSTEM_PROMPT
        self._persist()

    # ---------- баны пользователей ----------

    def is_user_banned(self, user_id: int) -> bool:
        return user_id in self._data["banned_users"]

    def ban_user(self, user_id: int):
        if user_id not in self._data["banned_users"]:
            self._data["banned_users"].append(user_id)
            self._persist()

    def unban_user(self, user_id: int):
        if user_id in self._data["banned_users"]:
            self._data["banned_users"].remove(user_id)
            self._persist()

    @property
    def banned_users(self) -> list:
        return list(self._data["banned_users"])

    # ---------- баны групп ----------

    def is_chat_banned(self, chat_id: int) -> bool:
        return chat_id in self._data["banned_chats"]

    def ban_chat(self, chat_id: int):
        if chat_id not in self._data["banned_chats"]:
            self._data["banned_chats"].append(chat_id)
            self._persist()

    # ---------- закреплённые темы в группах ----------

    def get_chat_topic(self, chat_id: int) -> int | None:
        topics = self._data.get("chat_topics", {})
        return topics.get(str(chat_id))

    def set_chat_topic(self, chat_id: int, topic_id: int | None):
        if "chat_topics" not in self._data:
            self._data["chat_topics"] = {}
        if topic_id is None:
            self._data["chat_topics"].pop(str(chat_id), None)
        else:
            self._data["chat_topics"][str(chat_id)] = int(topic_id)
        self._persist()

    def unban_chat(self, chat_id: int):
        if chat_id in self._data["banned_chats"]:
            self._data["banned_chats"].remove(chat_id)
            self._persist()

    @property
    def banned_chats(self) -> list:
        return list(self._data["banned_chats"])

    # ---------- админы ----------

    def is_admin(self, user_id: int, owner_id: int) -> bool:
        return user_id == owner_id or user_id in self._data["admins"]

    def add_admin(self, user_id: int):
        if user_id not in self._data["admins"]:
            self._data["admins"].append(user_id)
            self._persist()

    def remove_admin(self, user_id: int):
        if user_id in self._data["admins"]:
            self._data["admins"].remove(user_id)
            self._persist()

    @property
    def admins(self) -> list:
        return list(self._data["admins"])

    # ---------- антикраш счётчик ----------

    def bump_crash_count(self) -> int:
        self._data["crash_count"] += 1
        self._persist()
        return self._data["crash_count"]

    def reset_crash_count(self):
        self._data["crash_count"] = 0
        self._persist()

    @property
    def crash_count(self) -> int:
        return self._data["crash_count"]

    # ---------- журнал событий ----------

    def add_log(self, kind: str, chat_id: int, user_label: str, text: str):
        """
        kind: 'question' | 'answer' | 'error'
        Храним только последние MAX_LOG_ENTRIES записей (кольцевой буфер).
        """
        import datetime

        entry = {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "kind": kind,
            "chat_id": chat_id,
            "user": user_label,
            "text": text[:500],  # не раздуваем файл длинными сообщениями
        }
        self._data["event_log"].append(entry)
        if len(self._data["event_log"]) > MAX_LOG_ENTRIES:
            self._data["event_log"] = self._data["event_log"][-MAX_LOG_ENTRIES:]
        self._persist()

    def get_logs(self, limit: int = 20) -> list:
        return self._data["event_log"][-limit:]

    def clear_logs(self):
        self._data["event_log"] = []
        self._persist()

    # ---------- реестр известных чатов (для кнопок /say) ----------

    def remember_chat(self, chat_id: int, title: str, chat_type: str):
        """Запоминаем чат при каждом сообщении, чтобы потом показать его кнопкой в /say."""
        key = str(chat_id)
        entry = {"title": title, "type": chat_type}
        if self._data["known_chats"].get(key) != entry:
            self._data["known_chats"][key] = entry
            self._persist()

    @property
    def known_chats(self) -> dict:
        return dict(self._data["known_chats"])


import sqlite3

SQLITE_PATH = os.getenv("SQLITE_PATH", "chat_history.db")


class HistoryDB:
    """Вечное хранилище всех сообщений чатов в базе данных SQLite."""

    def __init__(self, db_path: str = SQLITE_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER,
                    role TEXT,
                    sender_info TEXT,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat ON messages(chat_id)")

    def add_message(self, chat_id: int | str, role: str, content: str, sender_info: str = ""):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO messages (chat_id, role, sender_info, content) VALUES (?, ?, ?, ?)",
                (str(chat_id), role, sender_info, content),
            )

    def get_history(self, chat_id: int | str, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
                (str(chat_id), limit),
            )
            rows = cursor.fetchall()
            return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    def clear_history(self, chat_id: int | str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE chat_id = ?", (str(chat_id),))


history_db = HistoryDB()
