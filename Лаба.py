# Лабораторная работа: очистка сообщений университетского сервиса.
# Скрипт читает сообщения, убирает служебные данные и готовит текст для анализа.
import re
import os
import json
import pandas as pd
from collections import Counter

# nltk содержит список стоп-слов, а pymorphy3 приводит слова к начальной форме.
import nltk
from nltk.corpus import stopwords
import pymorphy3


# Все файлы лежат рядом со скриптом, поэтому программа не зависит от папки запуска.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "02_universitetskiy_servis.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "clean_messages.csv")

# UTF-8-SIG позволяет без проблем открыть результат в Excel.
# Дату сразу преобразуем в тип datetime: так её удобнее проверять и сортировать.
df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", parse_dates=["created_at"])

REQUIRED_COLUMNS = {"message_id", "channel", "created_at", "message"}
# Если преподаватель или пользователь передал файл с другой структурой,
# лучше остановиться здесь, чем получить непонятную ошибку позже.
missing_columns = REQUIRED_COLUMNS - set(df.columns)
if missing_columns:
    raise ValueError(f"В исходном файле нет столбцов: {sorted(missing_columns)}")

# Перед обработкой выводим короткую сводку, чтобы заметить проблемы во входном файле.
print("Размер датасета:", df.shape)
print("Столбцы:", list(df.columns))
print("Типы данных:\n", df.dtypes)


# В этом столбце находится текст, который будем очищать.
TEXT_COL = "message"
df = df.dropna(subset=[TEXT_COL]).reset_index(drop=True)
# После удаления пропусков оставляем текст строкой, даже если он состоит из цифр.
df[TEXT_COL] = df[TEXT_COL].astype(str)


# На этом шаге приводим текст к одному виду, но сохраняем символы,
# которые ещё нужны для распознавания ссылок, почты, телефонов и цен.
def basic_clean(text):
    # str() защищает от ошибок, если в тексте всё-таки встретится необычное значение.
    text = str(text).lower()
    # В URL могут встречаться = и &, а символы валют пригодятся при поиске цен.
    text = re.sub(
        r"[^\w\s.,!?\-+@:/\[\]=&%₽€$£¥]",
        " ",
        text,
        flags=re.UNICODE,
    )
    # Несколько одинаковых знаков подряд заменяем одним: "!!!" -> "!".
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Разные записи одной валюты приводим к короткому единому обозначению.
def normalize_currency(text):
    # Не смешиваем валюты: у каждой остаётся своё текстовое обозначение.
    text = text.replace("₽", " руб ")
    text = text.replace("€", " евро ")
    text = text.replace("$", " долл ")
    text = text.replace("£", " фунт ")
    text = text.replace("¥", " иен ")
    text = re.sub(r"\bRUB\b",     " руб ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bрублей\b",  " руб ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bруб\.",     " руб ", text, flags=re.IGNORECASE)
    # Например, "124535Р" превращается в "124535 руб".
    text = re.sub(r"(\d)\s*[Рр]\b", r"\1 руб ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Эти шаблоны находят данные, которые нужно сохранить отдельно и скрыть в тексте.
# Регулярные выражения компилируются один раз, потому что затем применяются к каждой строке.
# Поддерживаются ссылки с http/https и варианты, начинающиеся с www.
RE_URL   = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
# В адресе электронной почты разрешены точки, дефисы и знак плюс.
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Шаблон покрывает распространённые российские форматы: +7-999-123-45-67 и 8 (999) 123-45-67.
RE_PHONE = re.compile(r"(?:\+7|8)[\s\-\(\)]*\d{3}[\s\-\(\)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}")
# Даты могут быть записаны через точку, слеш, дефис или в ISO-формате.
RE_DATE  = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
RE_PRICE = re.compile(
    r"\d[\d\s]*(?:[.,]\d+)?\s*(?:руб|евро|долл|фунт|иен)\b",
    re.IGNORECASE,
)


def extract_entities(text):
    # findall возвращает список: в одном сообщении может быть несколько ссылок или телефонов.
    return {
        "urls":   RE_URL.findall(text),
        "emails": RE_EMAIL.findall(text),
        "phones": RE_PHONE.findall(text),
        "dates":  RE_DATE.findall(text),
        "prices": RE_PRICE.findall(text),
    }


# Вместо персональных и служебных данных оставляем понятные метки.
def anonymize(text):
    # Порядок важен: сначала заменяем целые ссылки и почту, затем более короткие сущности.
    text = RE_URL.sub("[URL]", text)
    text = RE_EMAIL.sub("[EMAIL]", text)
    text = RE_PHONE.sub("[PHONE]", text)
    text = RE_DATE.sub("[DATE]", text)
    text = RE_PRICE.sub("[PRICE]", text)
    return text


# Стоп-слова не несут темы сообщения, поэтому исключаем их из анализа.
def get_stop_words():
    try:
        russian = stopwords.words("russian")
        english = stopwords.words("english")
    except LookupError as error:
        raise RuntimeError(
            "Не найден ресурс NLTK stopwords. Выполните один раз: "
            "python -m nltk.downloader stopwords"
        ) from error

    # Объединяем списки в set: поиск слова в множестве выполняется быстро.
    return set(russian) | set(english)


# Помимо стандартных стоп-слов убираем слова из ссылок и служебные метки.
STOP = get_stop_words()
STOP |= {
    "url", "email", "phone", "date", "price",
    "mail", "http", "https", "www",
    "тел", "тело", "руб", "rub",
    "com", "ru", "org", "net", "example", "csu",
}

# Берём русские слова от трёх букв: короткие предлоги и частицы нам не нужны.
TOKEN_RE = re.compile(r"[а-яё]{3,}", re.IGNORECASE)


def tokenize(text):
    # findall сразу отбрасывает числа и пунктуацию, а фильтр убирает стоп-слова.
    return [w for w in TOKEN_RE.findall(text.lower()) if w not in STOP]


# Лемматизация приводит формы слова к начальной форме: "справки" -> "справка".
morph = pymorphy3.MorphAnalyzer()


def lemmatize(tokens):
    # parse() может вернуть несколько вариантов разбора; первый — наиболее вероятный.
    return [morph.parse(t)[0].normal_form for t in tokens]


# Последовательно применяем каждый этап к каждому сообщению.
df["basic_clean"] = df[TEXT_COL].apply(basic_clean)
df["normalized"]  = df["basic_clean"].apply(normalize_currency)

# Сначала сохраняем найденные сущности в отдельные столбцы.
ents = df["normalized"].apply(extract_entities)
for key in ["urls", "emails", "phones", "dates", "prices"]:
    # Для каждого типа сущности создаём отдельный столбец с найденными значениями.
    df[key] = ents.apply(lambda d, k=key: d[k])

# Затем скрываем эти сущности в тексте и готовим слова для анализа.
df["anonymized"] = df["normalized"].apply(anonymize)

df["tokens"]     = df["anonymized"].apply(tokenize)
df["lemmas"]     = df["tokens"].apply(lemmatize)
df["clean_text"] = df["lemmas"].apply(lambda x: " ".join(x))


def join_entities(row):
    # Этот столбец удобен для быстрого просмотра всех сущностей одной строки.
    parts = []
    if row["urls"]:   parts.append(f"url: {json.dumps(row['urls'], ensure_ascii=False)}")
    if row["emails"]: parts.append(f"email: {json.dumps(row['emails'], ensure_ascii=False)}")
    if row["phones"]: parts.append(f"phone: {json.dumps(row['phones'], ensure_ascii=False)}")
    if row["dates"]:  parts.append(f"date: {json.dumps(row['dates'], ensure_ascii=False)}")
    if row["prices"]: parts.append(f"price: {json.dumps(row['prices'], ensure_ascii=False)}")
    return " | ".join(parts)

df["entities"] = df.apply(join_entities, axis=1)


# Сравниваем частые слова в исходных сообщениях и после обработки.
def top_words(texts, n=20):
    # Counter считает, сколько раз встретилось каждое слово во всех сообщениях.
    c = Counter()
    for t in texts:
        c.update(TOKEN_RE.findall(str(t).lower()))
    return c.most_common(n)


print("\n=== TOP-20 ДО очистки ===")
for w, n in top_words(df[TEXT_COL]):
    print(f"{w:20s} {n}")

print("\n=== TOP-20 ПОСЛЕ очистки и лемматизации ===")
c_after = Counter(w for lst in df["lemmas"] for w in lst)
# Здесь считаем уже леммы, поэтому "оформить" и "оформлено" попадают в одно значение.
for w, n in c_after.most_common(20):
    print(f"{w:20s} {n}")


# Для записи в CSV списки превращаем в одну строку через "|".
# Разделитель ";" удобен при открытии файла в Excel с русской локалью.
out = df.copy()
# Копию используем, чтобы в памяти исходный DataFrame сохранил списки, а не строки.

# CSV не умеет хранить списки, поэтому записываем их в текстовом виде.
for col in ["urls", "emails", "phones", "dates", "prices", "tokens", "lemmas"]:
    out[col] = out[col].apply(lambda lst: "|".join(map(str, lst)))

# Задаём понятный порядок столбцов в готовом файле.
IDENT = ["message_id", "channel", "created_at"]
CLEAN = ["basic_clean"]
NORM  = ["normalized"]
ANON  = ["anonymized"]
LEX   = ["tokens", "lemmas", "clean_text"]
ENT   = ["entities", "urls", "emails", "phones", "dates", "prices"]

COLUMNS_ORDER = IDENT + CLEAN + NORM + ANON + LEX + ENT

missing = [c for c in COLUMNS_ORDER if c not in out.columns]
# Эта проверка помогает заметить опечатку в названии столбца до записи файла.
if missing:
    raise ValueError(f"Нет столбцов: {missing}. Есть: {list(out.columns)}")

out = out[COLUMNS_ORDER]
# Оставляем только нужные для сдачи поля и располагаем их в логичном порядке.

SEP = ";"
out.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
    sep=SEP,
    date_format="%Y-%m-%d %H:%M:%S",
)

print(f"\nСохранено: {OUTPUT_FILE}")
print(f"Разделитель: '{SEP}' | Кодировка: utf-8-sig")
print("Строк:", len(out), "| Столбцов:", len(out.columns))
print("\nБлоки столбцов:")
print("  1. ИДЕНТИФИКАЦИЯ:", IDENT)
print("  2. ЧИСТКА:        ", CLEAN)
print("  3. ВАЛЮТЫ:        ", NORM)
print("  4. ОБЕЗЛИЧЕНО:    ", ANON)
print("  5. ТОКЕНЫ/ЛЕММЫ:  ", LEX)
print("  6. СУЩНОСТИ:      ", ENT)
