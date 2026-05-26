from __future__ import annotations

import os
import random
import sqlite3
from contextlib import asynccontextmanager
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv, find_dotenv

from vkbottle import API
from vkbottle.bot import Bot, Message
from vkbottle.http import AiohttpClient

from safety_analyzer import SafetyAnalyzer
from sentiment_analyzer import SentimentAnalyzer


# ------------------------------------------------------------
# UNSAFE SSL FIX (Без сертификата безопасности, я не смог завести)
# ------------------------------------------------------------


class UnsafeAiohttpClient(AiohttpClient):
    @asynccontextmanager
    async def request(self, url: str, method: str = "POST", data=None, **kwargs):
        kwargs["ssl"] = False

        async with super().request(
            url=url,
            method=method,
            data=data,
            **kwargs,
        ) as response:
            yield response


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

env_path = find_dotenv()
print(f"Loaded .env from: {env_path}")

load_dotenv(env_path, override=True)

VK_GROUP_TOKEN = os.getenv("VK_GROUP_TOKEN")
WORK_CHAT_PEER_ID = int(os.getenv("WORK_CHAT_PEER_ID", "0"))

REPORT_INTERVAL_SECONDS = int(os.getenv("REPORT_INTERVAL_SECONDS", "1800"))
REPORT_TZ = ZoneInfo(os.getenv("REPORT_TZ", "Europe/Moscow"))

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/scratch-transformer-sentiment"))
SAFETY_MODEL_DIR = Path(os.getenv("SAFETY_MODEL_DIR", "models/scratch-transformer-safety"))
if not SAFETY_MODEL_DIR.exists():
    parent_safety_model_dir = Path("..") / SAFETY_MODEL_DIR
    if parent_safety_model_dir.exists():
        SAFETY_MODEL_DIR = parent_safety_model_dir
SAFETY_DANGEROUS_THRESHOLD = float(os.getenv("SAFETY_DANGEROUS_THRESHOLD", "0.5"))
DB_PATH = Path(os.getenv("DB_PATH", "sentiment_stats.sqlite3"))

TOP_MESSAGES_PER_USER = int(os.getenv("TOP_MESSAGES_PER_USER", "3"))
MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", "3"))

ADMIN_IDS_RAW = os.getenv("ADMIN_IDS", "")
ADMIN_IDS = {
    int(item.strip())
    for item in ADMIN_IDS_RAW.split(",")
    if item.strip().isdigit()
}

print(f"ENV WORK_CHAT_PEER_ID raw = {os.getenv('WORK_CHAT_PEER_ID')}")

if not VK_GROUP_TOKEN:
    raise RuntimeError("Не задан VK_GROUP_TOKEN в .env")

if not WORK_CHAT_PEER_ID:
    raise RuntimeError("Не задан WORK_CHAT_PEER_ID в .env")


# ------------------------------------------------------------
# INIT
# ------------------------------------------------------------

api = API(
    token=VK_GROUP_TOKEN,
    http_client=UnsafeAiohttpClient(),
)

bot = Bot(api=api)
analyzer = SentimentAnalyzer(MODEL_DIR)
safety_analyzer = SafetyAnalyzer(SAFETY_MODEL_DIR, SAFETY_DANGEROUS_THRESHOLD)


# ------------------------------------------------------------
# DATA STRUCTURES
# ------------------------------------------------------------

@dataclass
class ScoredMessage:
    text: str
    label: str
    confidence: float
    safety_label: str | None = None
    safety_confidence: float | None = None


@dataclass
class SafetyMessage:
    user_id: int
    text: str
    safety_label: str
    safety_confidence: float


user_name_cache: dict[int, str] = {}


# ------------------------------------------------------------
# SQLITE STORAGE
# ------------------------------------------------------------

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                peer_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                vk_message_id INTEGER,
                conversation_message_id INTEGER,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                label TEXT,
                confidence REAL,
                safety_label TEXT,
                safety_confidence REAL
            )
            """
        )

        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        if "safety_label" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN safety_label TEXT")
        if "safety_confidence" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN safety_confidence REAL")
        if "vk_message_id" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN vk_message_id INTEGER")
        if "conversation_message_id" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN conversation_message_id INTEGER")

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_peer_created
            ON messages(peer_id, created_at)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_user_created
            ON messages(user_id, created_at)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_peer_user_created
            ON messages(peer_id, user_id, created_at)
            """
        )


def save_message_to_db(
    peer_id: int,
    user_id: int,
    text: str,
    created_at: datetime,
    vk_message_id: int | None = None,
    conversation_message_id: int | None = None,
) -> None:
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO messages (
                peer_id,
                user_id,
                vk_message_id,
                conversation_message_id,
                text,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                peer_id,
                user_id,
                vk_message_id,
                conversation_message_id,
                text,
                created_at.isoformat(timespec="seconds"),
            ),
        )


def load_messages_from_db(
    peer_id: int,
    start_dt: datetime,
    end_dt: datetime,
) -> list[sqlite3.Row]:
    with db_connect() as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                peer_id,
                user_id,
                vk_message_id,
                conversation_message_id,
                text,
                created_at,
                label,
                confidence,
                safety_label,
                safety_confidence
            FROM messages
            WHERE peer_id = ?
              AND created_at >= ?
              AND created_at < ?
            ORDER BY created_at ASC
            """,
            (
                peer_id,
                start_dt.isoformat(timespec="seconds"),
                end_dt.isoformat(timespec="seconds"),
            ),
        ).fetchall()

    return list(rows)


def update_message_prediction(message_id: int, label: str, confidence: float) -> None:
    with db_connect() as conn:
        conn.execute(
            """
            UPDATE messages
            SET label = ?, confidence = ?
            WHERE id = ?
            """,
            (label, confidence, message_id),
        )


def update_message_safety_prediction(message_id: int, safety_label: str, safety_confidence: float) -> None:
    with db_connect() as conn:
        conn.execute(
            """
            UPDATE messages
            SET safety_label = ?, safety_confidence = ?
            WHERE id = ?
            """,
            (safety_label, safety_confidence, message_id),
        )


def delete_all_stats() -> int:
    with db_connect() as conn:
        cursor = conn.execute("DELETE FROM messages")
        return cursor.rowcount


# ------------------------------------------------------------
# UTILS
# ------------------------------------------------------------

def label_ru(label: str) -> str:
    return {
        "positive": "позитивно",
        "neutral": "нейтрально",
        "negative": "негативно",
    }.get(label, label)


def emoji_for_label(label: str) -> str:
    return {
        "positive": "🟢",
        "neutral": "⚪",
        "negative": "🔴",
    }.get(label, "⚪")


def safety_label_ru(label: str) -> str:
    return {
        "safe": "безопасно",
        "dangerous": "требует проверки",
    }.get(label, label)


def emoji_for_safety(label: str) -> str:
    return {
        "safe": "✅",
        "dangerous": "🚨",
    }.get(label, "⚪")


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip()


def shorten_text(text: str, limit: int = 160) -> str:
    text = normalize_text(text)

    if len(text) <= limit:
        return text

    return text[: limit - 3] + "..."


def is_admin(user_id: int) -> bool:
    return bool(ADMIN_IDS) and user_id in ADMIN_IDS


def get_period_range(period: str) -> tuple[datetime, datetime, str]:
    now = datetime.now(REPORT_TZ)
    today_start = datetime.combine(now.date(), time.min, tzinfo=REPORT_TZ)

    if period == "day":
        start = today_start
        end = start + timedelta(days=1)
        title = "за сегодня"

    elif period == "week":
        start = today_start - timedelta(days=6)
        end = today_start + timedelta(days=1)
        title = "за последние 7 дней"

    elif period == "month":
        start = today_start - timedelta(days=29)
        end = today_start + timedelta(days=1)
        title = "за последние 30 дней"

    else:
        raise ValueError(f"Unknown period: {period}")

    return start, end, title


async def get_user_name(user_id: int) -> str:
    """
    Возвращает VK-упоминание вида:
    @id248151842(Имя Ф.)
    """

    if user_id in user_name_cache:
        return user_name_cache[user_id]

    try:
        users = await bot.api.users.get(user_ids=[user_id])

        if users:
            user = users[0]

            first_name = getattr(user, "first_name", "") or ""
            last_name = getattr(user, "last_name", "") or ""

            if first_name and last_name:
                display_name = f"{first_name} {last_name[0]}."
            elif first_name:
                display_name = first_name
            else:
                display_name = f"id{user_id}"

            name = f"@id{user_id}({display_name})"
        else:
            name = f"@id{user_id}(id{user_id})"

    except Exception as error:
        print(f"[GET USER NAME ERROR] user_id={user_id}, error={error}")
        name = f"@id{user_id}(id{user_id})"

    user_name_cache[user_id] = name
    return name


def aggregate_user(scored_messages: list[ScoredMessage]) -> tuple[str, float, Counter]:
    """
    Возвращает:
    - итоговую тональность пользователя;
    - среднюю уверенность по итоговой тональности;
    - счётчик всех тональностей.
    """

    counts = Counter(item.label for item in scored_messages)

    if not counts:
        return "neutral", 0.0, counts

    most_common = counts.most_common()
    max_count = most_common[0][1]

    tied_labels = [
        label
        for label, count in most_common
        if count == max_count
    ]

    if len(tied_labels) == 1:
        final_label = tied_labels[0]
    else:
        final_label = max(
            tied_labels,
            key=lambda label: (
                sum(item.confidence for item in scored_messages if item.label == label)
                / max(1, counts[label])
            ),
        )

    avg_confidence = (
        sum(item.confidence for item in scored_messages if item.label == final_label)
        / max(1, counts[final_label])
    )

    return final_label, avg_confidence, counts


def is_command(text: str) -> bool:
    text = text.lower().strip()

    command_prefixes = (
        "/sentiment",
        "!sentiment",
        "/отчет",
        "!отчет",
        "/отчёт",
        "!отчёт",
        "/report",
        "!report",
        "/day",
        "!day",
        "/today",
        "!today",
        "/день",
        "!день",
        "/week",
        "!week",
        "/неделя",
        "!неделя",
        "/month",
        "!month",
        "/месяц",
        "!месяц",
        "/clearstats",
        "!clearstats",
        "/clear_stats",
        "!clear_stats",
        "/очистить",
        "!очистить",
        "/ping",
        "!ping",
        "/chatid",
        "!chatid",
        "/id",
        "!id",
        "/safety",
        "!safety",
        "/danger",
        "!danger",
        "/риски",
        "!риски",
        "/help",
        "!help",
        "/помощь",
        "!помощь",
        "/info",
        "!info",
    )

    exact_commands = (
        "отчет",
        "отчёт",
        "report",
        "help",
        "помощь",
        "info",
        "ping",
        "safety",
        "риски",
    )

    return text.startswith(command_prefixes) or text in exact_commands


# ------------------------------------------------------------
# REPORT LOGIC
# ------------------------------------------------------------

async def ensure_sentiment_predictions_for_rows(rows: list[sqlite3.Row]) -> None:
    rows_without_prediction = [
        row for row in rows
        if row["label"] is None or row["confidence"] is None
    ]

    if rows_without_prediction:
        texts_to_predict = [row["text"] for row in rows_without_prediction]
        predictions = await analyzer.predict(texts_to_predict)

        for row, (label, confidence) in zip(rows_without_prediction, predictions, strict=True):
            update_message_prediction(
                message_id=int(row["id"]),
                label=label,
                confidence=confidence,
            )


async def ensure_safety_predictions_for_rows(rows: list[sqlite3.Row]) -> None:
    rows_without_safety_prediction = [
        row for row in rows
        if row["safety_label"] is None or row["safety_confidence"] is None
    ]

    if not rows_without_safety_prediction:
        return

    texts_to_predict = [row["text"] for row in rows_without_safety_prediction]
    safety_predictions = await safety_analyzer.predict(texts_to_predict)

    for row, (safety_label, safety_confidence) in zip(
        rows_without_safety_prediction,
        safety_predictions,
        strict=True,
    ):
        update_message_safety_prediction(
            message_id=int(row["id"]),
            safety_label=safety_label,
            safety_confidence=safety_confidence,
        )


async def ensure_predictions_for_rows(rows: list[sqlite3.Row]) -> None:
    await ensure_sentiment_predictions_for_rows(rows)
    await ensure_safety_predictions_for_rows(rows)


async def build_report(period: str = "day", force: bool = False) -> str | None:
    start_dt, end_dt, period_title = get_period_range(period)

    rows = load_messages_from_db(
        peer_id=WORK_CHAT_PEER_ID,
        start_dt=start_dt,
        end_dt=end_dt,
    )

    if not rows:
        if force:
            return (
                f"За период {period_title} пока нет сообщений для анализа.\n\n"
                "Важно: я анализирую только обычные сообщения. "
                "Команды вроде /help, /ping, /sentiment в анализ не попадают."
            )
        return None

    await ensure_predictions_for_rows(rows)

    # Перечитываем после обновления label/confidence
    rows = load_messages_from_db(
        peer_id=WORK_CHAT_PEER_ID,
        start_dt=start_dt,
        end_dt=end_dt,
    )

    messages_by_user: dict[int, list[sqlite3.Row]] = defaultdict(list)

    for row in rows:
        messages_by_user[int(row["user_id"])].append(row)

    lines: list[str] = [
        f"📊 Отчёт по настроению {period_title}",
        f"Период: {start_dt.strftime('%d.%m.%Y')} — {(end_dt - timedelta(seconds=1)).strftime('%d.%m.%Y')}",
        "",
    ]

    sorted_users = sorted(
        messages_by_user.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    total_messages = 0
    total_counts = Counter()
    dangerous_rows: list[sqlite3.Row] = []

    for user_id, user_rows in sorted_users:
        scored_messages: list[ScoredMessage] = []

        for row in user_rows:
            if row["label"] is None or row["confidence"] is None:
                continue

            scored_messages.append(
                ScoredMessage(
                    text=row["text"],
                    label=row["label"],
                    confidence=float(row["confidence"]),
                    safety_label=row["safety_label"],
                    safety_confidence=(
                        float(row["safety_confidence"])
                        if row["safety_confidence"] is not None
                        else None
                    ),
                )
            )

        if not scored_messages:
            continue

        final_label, avg_confidence, counts = aggregate_user(scored_messages)
        name = await get_user_name(user_id)

        total_messages += len(scored_messages)
        total_counts.update(counts)
        user_dangerous_count = sum(
            1
            for item in scored_messages
            if item.safety_label == "dangerous"
        )
        dangerous_rows.extend(
            row for row in user_rows
            if row["safety_label"] == "dangerous"
            and row["safety_confidence"] is not None
        )

        top_messages = sorted(
            scored_messages,
            key=lambda item: item.confidence,
            reverse=True,
        )[:TOP_MESSAGES_PER_USER]

        lines.append(
            f"{emoji_for_label(final_label)} {name}: "
            f"{label_ru(final_label)} "
            f"({avg_confidence:.0%})"
        )

        lines.append(
            f"   Сообщений: {len(scored_messages)} | "
            f"🟢 {counts.get('positive', 0)} / "
            f"⚪ {counts.get('neutral', 0)} / "
            f"🔴 {counts.get('negative', 0)} | "
            f"🚨 dangerous: {user_dangerous_count}"
        )

        lines.append("   Самые вероятные сообщения:")

        for item in top_messages:
            safety_prefix = ""
            if item.safety_label == "dangerous":
                safety_score = item.safety_confidence or 0.0
                safety_prefix = f"🚨 dangerous {safety_score:.0%} | "

            lines.append(
                f"   — {emoji_for_label(item.label)} "
                f"{label_ru(item.label)} {item.confidence:.0%}: "
                f"{safety_prefix}«{shorten_text(item.text)}»"
            )

        lines.append("")

    lines.append("Итого по чату:")
    lines.append(
        f"Сообщений: {total_messages} | "
        f"🟢 {total_counts.get('positive', 0)} / "
        f"⚪ {total_counts.get('neutral', 0)} / "
        f"🔴 {total_counts.get('negative', 0)} | "
        f"🚨 dangerous: {len(dangerous_rows)}"
    )

    if dangerous_rows:
        lines.append("")
        lines.append("🚨 Сообщения, требующие ручной проверки:")
        for row in sorted(dangerous_rows, key=lambda item: float(item["safety_confidence"]), reverse=True)[:10]:
            name = await get_user_name(int(row["user_id"]))
            lines.append(
                f"— {name} | {float(row['safety_confidence']):.0%}: "
                f"«{shorten_text(row['text'], 220)}»"
            )
    else:
        lines.append("")
        lines.append("✅ Safety-модуль: сообщений, требующих ручной проверки, не найдено.")

    report = "\n".join(lines).strip()

    if len(report) > 4000:
        report = report[:3900]
        report += "\n\n...отчёт был обрезан, потому что он слишком длинный."

    return report


async def build_safety_report(period: str = "day", force: bool = False) -> str | None:
    start_dt, end_dt, period_title = get_period_range(period)
    rows = load_messages_from_db(
        peer_id=WORK_CHAT_PEER_ID,
        start_dt=start_dt,
        end_dt=end_dt,
    )

    if not rows:
        if force:
            return f"За период {period_title} пока нет сообщений для safety-анализа."
        return None

    await ensure_safety_predictions_for_rows(rows)
    rows = load_messages_from_db(
        peer_id=WORK_CHAT_PEER_ID,
        start_dt=start_dt,
        end_dt=end_dt,
    )

    dangerous_rows = [
        row for row in rows
        if row["safety_label"] == "dangerous"
        and row["safety_confidence"] is not None
    ]
    total_safety_counts = Counter(
        row["safety_label"]
        for row in rows
        if row["safety_label"] is not None
    )

    lines = [
        f"🛡 Safety-отчёт {period_title}",
        f"Период: {start_dt.strftime('%d.%m.%Y')} — {(end_dt - timedelta(seconds=1)).strftime('%d.%m.%Y')}",
        "",
        f"Всего проанализировано: {sum(total_safety_counts.values())}",
        f"✅ safe: {total_safety_counts.get('safe', 0)}",
        f"🚨 dangerous: {total_safety_counts.get('dangerous', 0)}",
    ]

    if dangerous_rows:
        lines.append("")
        lines.append("Сообщения, требующие ручной проверки:")
        for row in sorted(dangerous_rows, key=lambda item: float(item["safety_confidence"]), reverse=True)[:20]:
            name = await get_user_name(int(row["user_id"]))
            lines.append(
                f"— {name} | {float(row['safety_confidence']):.0%}: "
                f"«{shorten_text(row['text'], 260)}»"
            )
    else:
        lines.append("")
        lines.append("Подозрительных сообщений за период не найдено.")

    report = "\n".join(lines).strip()
    if len(report) > 4000:
        report = report[:3900]
        report += "\n\n...отчёт был обрезан, потому что он слишком длинный."
    return report


async def send_report(period: str = "day", force: bool = False) -> None:
    report = await build_report(period=period, force=force)

    if not report:
        return

    await bot.api.messages.send(
        peer_id=WORK_CHAT_PEER_ID,
        message=report,
        random_id=random.randint(1, 2_147_483_647),
    )


# ------------------------------------------------------------
# COMMAND HANDLERS
# ------------------------------------------------------------


@bot.on.chat_message(
    text=[
        "/help",
        "!help",
        "help",
        "/помощь",
        "!помощь",
        "помощь",
        "/info",
        "!info",
        "info",
    ]
)
async def bot_help(message: Message):
    if message.peer_id != WORK_CHAT_PEER_ID:
        return

    help_text = (
        "🤖 Информация о работе бота\n\n"
        "Я анализирую настроение сообщений и отмечаю сообщения, "
        "которые safety-модель считает потенциально опасными.\n\n"
        "Как это работает:\n"
        "1. Я читаю новые текстовые сообщения в этой беседе.\n"
        "2. Сохраняю их в SQLite, поэтому статистика не пропадает после перезапуска.\n"
        "3. При построении отчёта анализирую сообщения моделью тональности и safety-моделью.\n"
        "4. Для каждого участника определяю общий тон общения:\n"
        "   🟢 позитивный\n"
        "   ⚪ нейтральный\n"
        "   🔴 негативный\n"
        "5. В отчёте показываю самые уверенные сообщения, "
        "по которым модель сделала вывод.\n\n"
        "6. Для ИБ-задачи отдельно выделяю сообщения dangerous: "
        "их нужно проверять вручную, а не автоматически наказывать пользователя.\n\n"
        "Команды:\n"
        "/report или /sentiment или /отчет — отчёт за сегодня\n"
        "/safety или /риски — ИБ-отчёт по потенциально опасным сообщениям\n"
        "/day — отчёт за сегодня\n"
        "/week — отчёт за последние 7 дней\n"
        "/month — отчёт за последние 30 дней\n"
        "/clearstats — удалить всю статистику, только для админа\n"
        "/chatid или /id — узнать ID текущей беседы\n"
        "/ping — проверить, что бот работает\n"
        "/help — показать это сообщение\n\n"
        "Текущие настройки:\n"
        f"Рабочий чат: {WORK_CHAT_PEER_ID}\n"
        f"Интервал автоотчёта: {REPORT_INTERVAL_SECONDS} сек.\n"
        f"Часовой пояс: {REPORT_TZ}\n"
        f"SQLite база: {DB_PATH}\n"
        f"Safety-модель: {SAFETY_MODEL_DIR}\n"
        f"Порог dangerous: {SAFETY_DANGEROUS_THRESHOLD:.2f}\n"
        f"Минимальная длина сообщения: {MIN_TEXT_LENGTH} символов\n"
        f"Сообщений-примеров на участника: {TOP_MESSAGES_PER_USER}\n\n"
        "Важно:\n"
        "Я анализирую только сообщения, которые были написаны после подключения бота. "
        "Старую историю беседы я не подгружаю. "
        "Команды не попадают в анализ."
    )

    await message.answer(help_text)


@bot.on.chat_message(text=["/chatid", "!chatid", "/id", "!id"])
async def show_chat_id(message: Message):

    await message.answer(
        "🆔 Информация о чате\n\n"
        f"peer_id этого чата: {message.peer_id}\n"
        f"from_id отправителя: {message.from_id}\n\n"
        "Чтобы бот анализировал именно этот чат, вставь peer_id в .env:\n"
        f"WORK_CHAT_PEER_ID={message.peer_id}"
    )


@bot.on.chat_message(text=["/ping", "!ping", "ping"])
async def ping(message: Message):
    if message.peer_id != WORK_CHAT_PEER_ID:
        return

    await message.answer(
        "pong\n\n"
        f"peer_id={message.peer_id}\n"
        f"from_id={message.from_id}"
    )


@bot.on.chat_message(
    text=[
        "/sentiment",
        "!sentiment",
        "/отчет",
        "!отчет",
        "/отчёт",
        "!отчёт",
        "/report",
        "!report",
        "отчет",
        "отчёт",
        "report",
    ]
)
async def manual_report(message: Message):
    if message.peer_id != WORK_CHAT_PEER_ID:
        return

    report = await build_report(period="day", force=True)
    await message.answer(report)


@bot.on.chat_message(
    text=[
        "/safety",
        "!safety",
        "safety",
        "/danger",
        "!danger",
        "/риски",
        "!риски",
        "риски",
    ]
)
async def manual_safety_report(message: Message):
    if message.peer_id != WORK_CHAT_PEER_ID:
        return

    report = await build_safety_report(period="day", force=True)
    await message.answer(report)


@bot.on.chat_message(text=["/day", "!day", "/today", "!today", "/день", "!день"])
async def day_report(message: Message):
    if message.peer_id != WORK_CHAT_PEER_ID:
        return

    report = await build_report(period="day", force=True)
    await message.answer(report)


@bot.on.chat_message(text=["/week", "!week", "/неделя", "!неделя"])
async def week_report(message: Message):
    if message.peer_id != WORK_CHAT_PEER_ID:
        return

    report = await build_report(period="week", force=True)
    await message.answer(report)


@bot.on.chat_message(text=["/month", "!month", "/месяц", "!месяц"])
async def month_report(message: Message):
    if message.peer_id != WORK_CHAT_PEER_ID:
        return

    report = await build_report(period="month", force=True)
    await message.answer(report)


@bot.on.chat_message(
    text=[
        "/clearstats confirm",
        "!clearstats confirm",
        "/clear_stats confirm",
        "!clear_stats confirm",
        "/очистить confirm",
        "!очистить confirm",
    ]
)
async def clear_stats_confirm(message: Message):
    if message.peer_id != WORK_CHAT_PEER_ID:
        return

    if not is_admin(message.from_id):
        await message.answer(
            "⛔ У тебя нет прав на удаление статистики.\n\n"
            "Добавь свой VK ID в .env:\n"
            f"ADMIN_IDS={message.from_id}"
        )
        return

    deleted_count = delete_all_stats()

    await message.answer(
        "🗑 Статистика полностью удалена.\n\n"
        f"Удалено сообщений из базы: {deleted_count}"
    )


@bot.on.chat_message(
    text=[
        "/clearstats",
        "!clearstats",
        "/clear_stats",
        "!clear_stats",
        "/очистить",
        "!очистить",
    ]
)
async def clear_stats_help(message: Message):
    if message.peer_id != WORK_CHAT_PEER_ID:
        return

    if not is_admin(message.from_id):
        await message.answer(
            "⛔ У тебя нет прав на удаление статистики.\n\n"
            "Если ты владелец бота, добавь свой VK ID в .env:\n"
            f"ADMIN_IDS={message.from_id}"
        )
        return

    await message.answer(
        "⚠️ Эта команда удалит ВСЮ накопленную статистику из SQLite.\n\n"
        "Для подтверждения напиши:\n"
        "/clearstats confirm"
    )


# ------------------------------------------------------------
# MESSAGE COLLECTOR
# ------------------------------------------------------------


@bot.on.chat_message()
async def collect_chat_message(message: Message):
    """
    Собирает обычные сообщения только из нужного рабочего чата.
    Команды не анализируются.
    """

    if message.peer_id != WORK_CHAT_PEER_ID:
        return

    text = normalize_text(message.text or "")

    if not text:
        return

    if is_command(text):
        return

    if len(text) < MIN_TEXT_LENGTH:
        return

    save_message_to_db(
        peer_id=message.peer_id,
        user_id=message.from_id,
        text=text,
        created_at=datetime.now(REPORT_TZ),
        vk_message_id=getattr(message, "id", None),
        conversation_message_id=getattr(message, "conversation_message_id", None),
    )

    print(
        "[COLLECTOR] saved to sqlite:",
        "peer_id=", message.peer_id,
        "user_id=", message.from_id,
        "vk_message_id=", getattr(message, "id", None),
        "conversation_message_id=", getattr(message, "conversation_message_id", None),
        "text=", repr(text),
    )


# ------------------------------------------------------------
# PERIODIC TASK
# ------------------------------------------------------------

@bot.loop_wrapper.interval(seconds=REPORT_INTERVAL_SECONDS)
async def periodic_report():
    await send_report(period="day", force=False)


# ------------------------------------------------------------
# STARTUP
# ------------------------------------------------------------

async def startup():
    init_db()
    await analyzer.load_model()
    await safety_analyzer.load_model()

    print("Sentiment model loaded")
    print("Safety model loaded")
    print(f"WORK_CHAT_PEER_ID = {WORK_CHAT_PEER_ID}")
    print(f"REPORT_INTERVAL_SECONDS = {REPORT_INTERVAL_SECONDS}")
    print(f"MODEL_DIR = {MODEL_DIR}")
    print(f"SAFETY_MODEL_DIR = {SAFETY_MODEL_DIR}")
    print(f"SAFETY_DANGEROUS_THRESHOLD = {SAFETY_DANGEROUS_THRESHOLD}")
    print(f"DB_PATH = {DB_PATH}")
    print(f"ADMIN_IDS = {sorted(ADMIN_IDS)}")
    print("SSL verification is DISABLED for VKBottle HTTP client")

    try:
        await bot.api.messages.send(
            peer_id=WORK_CHAT_PEER_ID,
            message=(
                "✅ Бот анализа настроения и safety-рисков запущен.\n\n"
                "Команды:\n"
                "/help — информация о работе бота\n"
                "/report — отчёт за сегодня\n"
                "/safety — ИБ-отчёт по dangerous-сообщениям\n"
                "/day — отчёт за сегодня\n"
                "/week — отчёт за последние 7 дней\n"
                "/month — отчёт за последние 30 дней\n"
                "/clearstats — удалить всю статистику, только для админа\n"
                "/chatid — ID текущего чата\n"
                "/ping — проверка работы\n\n"
            ),
            random_id=random.randint(1, 2_147_483_647),
        )

    except Exception as error:
        print(f"Не удалось отправить стартовое сообщение: {error}")


bot.loop_wrapper.on_startup.append(startup())


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

if __name__ == "__main__":
    bot.run_forever()
