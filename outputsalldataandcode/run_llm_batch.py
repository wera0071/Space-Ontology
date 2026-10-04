"""
Пример:
    # OpenAI-compatible (ohmylama)
    export OHMYLAMA_API_KEY=sk-lama-...
    python run_llm_batch.py \\
        --provider openai \\
        --base_url https://ohmylama.ru/v1 \\
        --model gpt-5.4 \\
        --terms rkt_dev_terms_50.txt \\
        --template ontology \\
        --out gpt54_ontology.txt \\
        --concurrency 5

    # GigaChat
    export GIGACHAT_AUTH_KEY=<base64 client_id:secret>
    python run_llm_batch.py \\
        --provider gigachat \\
        --model GigaChat-Max \\
        --terms rkt_dev_terms_50.txt \\
        --template ontology \\
        --out giga_max_ontology.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx


TEMPLATES = {
    "zero": (
        "Перечисли гиперонимы для русского существительного.\n"
        "Ответ: только гиперонимы через \"; \" (точка с запятой и пробел). "
        "Никаких пояснений.\n\n"
        "гипоним: {WORD}\n"
        "гиперонимы:"
    ),
    "cot": (
        "Сначала подумай, какие гиперонимы наиболее вероятны, "
        "но не показывай ход рассуждений.\n"
        "Затем выведи только итоговый список (максимум 15) через \"; \".\n\n"
        "гипоним: {WORD}\n"
        "гиперонимы:"
    ),
    "ontology": (
        "Перечисли гиперонимы для русского существительного так, "
        "как их обычно задают в тезаурусе (краткие словарные наименования).\n\n"
        "Требования:\n"
        "- только существительные или именные группы;\n"
        "- без пояснений, примеров и комментариев;\n"
        "- от более общего к менее общему;\n"
        "- избегай слишком частных и случайных ассоциаций;\n"
        "- максимум 15;\n"
        "- ответ только через \"; \" (точка с запятой и пробел).\n\n"
        "гипоним: {WORD}\n"
        "гиперонимы:"
    ),
}


def read_terms(path: Path) -> list[str]:
    return [t.strip() for t in path.read_text(encoding="utf-8").splitlines() if t.strip()]


def normalize_answer(raw: str) -> str:
    """Чистим ответ модели: убираем переносы, лишние пробелы, шум."""
    if not raw:
        return ""
    txt = raw.strip()
    # Если модель пишет "гиперонимы: a; b; c" — отрезаем префикс
    txt = re.sub(r"^(гиперонимы\s*:\s*)", "", txt, flags=re.IGNORECASE)
    # Убираем переносы строк
    txt = re.sub(r"\s*\n+\s*", "; ", txt)
    # Лишние пробелы вокруг ;
    txt = re.sub(r"\s*;\s*", "; ", txt)
    # Сводим повторяющиеся пробелы
    txt = re.sub(r"\s+", " ", txt).strip()
    # Убираем висящий ; на конце
    txt = txt.rstrip("; ").strip()
    return txt


# ---------- OpenAI-совместимый провайдер ----------

async def call_openai(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = await client.post(url, headers=headers, json=payload, timeout=90.0)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


# ---------- GigaChat провайдер ----------

async def gigachat_get_token(client: httpx.AsyncClient, auth_key: str, scope: str) -> str:
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Basic {auth_key}",
    }
    data = {"scope": scope}
    r = await client.post(url, headers=headers, data=data, timeout=30.0)
    r.raise_for_status()
    return r.json()["access_token"]


async def call_gigachat(
    client: httpx.AsyncClient,
    access_token: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = await client.post(url, headers=headers, json=payload, timeout=90.0)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ---------- Основной прогон ----------

async def process_term(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    args,
    template: str,
    term: str,
    state: dict,
) -> tuple[str, str, Optional[str]]:
    prompt = template.replace("{WORD}", term)
    err: Optional[str] = None
    raw = ""
    async with sem:
        for attempt in range(1, args.retries + 1):
            try:
                if args.provider == "openai":
                    raw = await call_openai(
                        client, args.base_url, args.api_key,
                        args.model, prompt, args.max_tokens, args.temperature,
                    )
                else:
                    raw = await call_gigachat(
                        client, state["gigachat_token"],
                        args.model, prompt, args.max_tokens, args.temperature,
                    )
                err = None
                break
            except httpx.HTTPStatusError as e:
                err = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                # GigaChat: если access_token истёк — обновляем
                if args.provider == "gigachat" and e.response.status_code in (401, 403):
                    try:
                        state["gigachat_token"] = await gigachat_get_token(
                            client, args.api_key, args.gigachat_scope,
                        )
                    except Exception as ex:
                        err = f"token refresh failed: {ex}"
                await asyncio.sleep(min(2 ** attempt, 10))
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                await asyncio.sleep(min(2 ** attempt, 10))
    cleaned = normalize_answer(raw)
    return term, cleaned, err


async def main_async(args) -> None:
    template = TEMPLATES[args.template]
    terms = read_terms(Path(args.terms))
    print(f"[i] {len(terms)} terms; provider={args.provider}; "
          f"model={args.model}; template={args.template}; "
          f"concurrency={args.concurrency}")

    state: dict = {}
    async with httpx.AsyncClient(verify=args.verify_ssl) as client:
        if args.provider == "gigachat":
            print("[i] obtaining GigaChat access token...")
            state["gigachat_token"] = await gigachat_get_token(
                client, args.api_key, args.gigachat_scope,
            )
            print("[i] token ok")

        sem = asyncio.Semaphore(args.concurrency)
        t0 = time.time()
        tasks = [
            process_term(sem, client, args, template, term, state)
            for term in terms
        ]
        results = []
        done = 0
        for coro in asyncio.as_completed(tasks):
            term, ans, err = await coro
            results.append((term, ans, err))
            done += 1
            status = "OK" if not err else f"ERR[{err[:60]}]"
            print(f"  [{done}/{len(terms)}] {term[:40]:<40}  {status}")
        dt = time.time() - t0

    # Сохраняем в исходном порядке терминов
    by_term = {term: (ans, err) for term, ans, err in results}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    with out_path.open("w", encoding="utf-8") as f_main:
        for term in terms:
            ans, err = by_term.get(term, ("", "missing"))
            if ans and not err:
                n_ok += 1
            f_main.write(f"{term}\t{ans}\n")

    # Лог с ошибками отдельно
    log_path = out_path.with_suffix(out_path.suffix + ".log")
    with log_path.open("w", encoding="utf-8") as f_log:
        f_log.write(f"# provider={args.provider} model={args.model} "
                    f"template={args.template}\n")
        f_log.write(f"# terms={len(terms)} ok={n_ok} elapsed={dt:.1f}s\n")
        for term, ans, err in results:
            if err:
                f_log.write(f"{term}\tERR\t{err}\n")

    print(f"[i] done: {n_ok}/{len(terms)} ok in {dt:.1f}s")
    print(f"[i] wrote {out_path}")
    print(f"[i] log:   {log_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["openai", "gigachat"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--terms", required=True)
    ap.add_argument("--template", required=True, choices=sorted(TEMPLATES.keys()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--api_key", default=None,
                    help="API key. По умолчанию берётся из env: "
                         "OHMYLAMA_API_KEY/OPENAI_API_KEY (для openai) или "
                         "GIGACHAT_AUTH_KEY (для gigachat).")
    ap.add_argument("--base_url", default="https://ohmylama.ru/v1",
                    help="OpenAI-compat base URL (только для provider=openai).")
    ap.add_argument("--gigachat_scope", default="GIGACHAT_API_PERS",
                    help="GIGACHAT_API_PERS / GIGACHAT_API_B2B / GIGACHAT_API_CORP")
    ap.add_argument("--concurrency", type=int, default=5,
                    help="Параллельных запросов. Для GigaChat Freemium ставь 1.")
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--verify_ssl", action="store_true",
                    help="Включить проверку SSL (для GigaChat иногда нужно False, "
                         "т.к. они используют собственный CA).")
    args = ap.parse_args()

    if args.api_key is None:
        if args.provider == "openai":
            args.api_key = (os.environ.get("OHMYLAMA_API_KEY")
                            or os.environ.get("OPENAI_API_KEY"))
        else:
            args.api_key = os.environ.get("GIGACHAT_AUTH_KEY")
    if not args.api_key:
        sys.exit("ERROR: api_key not provided (use --api_key or env var)")

    if args.provider == "gigachat" and args.concurrency != 1:
        print(f"[!] WARN: GigaChat Freemium supports 1 concurrent stream. "
              f"Forcing concurrency=1 (was {args.concurrency})")
        args.concurrency = 1

    return args


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
