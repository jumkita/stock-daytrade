# -*- coding: utf-8 -*-
"""
Gemini API の初期化と定性監査（投資適格ランク判定）の汎用モジュール。
特定のアプリに依存せず、環境変数または secrets 辞書で API キーを取得する。
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ----- Gemini API 初期化（google.genai 新SDK） -----

try:
    from google import genai
    from google.genai import types as genai_types
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    genai = None
    genai_types = None


def get_gemini_api_key(secrets: Optional[Any] = None) -> Optional[str]:
    """
    GEMINI_API_KEY を取得する（ハイブリッド仕様）。
    優先順位: 1) 環境変数 GEMINI_API_KEY（ローカル・.env）
             2) secrets.get("GEMINI_API_KEY")（デプロイ時などで渡す辞書相当）
    いずれも取得できない場合のみ None を返す。呼び出し側でエラー表示すること。
    """
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    if secrets is not None:
        try:
            key = secrets.get("GEMINI_API_KEY") if hasattr(secrets, "get") else None
            if key and isinstance(key, str) and key.strip():
                return key.strip()
        except Exception:
            pass
    return None


def is_gemini_available() -> bool:
    """google-genai（新SDK）が利用可能かどうか。"""
    return _GEMINI_AVAILABLE and genai is not None


def generate_one_shot(
    prompt: str,
    api_key: Optional[str] = None,
    secrets: Optional[Any] = None,
) -> str:
    """
    プロンプトを1回送り、応答の先頭1行を返す汎用関数。
    キー未設定・エラー時はエラーメッセージを返す。
    """
    if not _GEMINI_AVAILABLE or genai is None:
        return "google-genai がインストールされていません。pip install google-genai"
    key = api_key or get_gemini_api_key(secrets)
    if not key:
        return "GEMINI_API_KEY が設定されていません。（環境変数または secrets で設定してください）"
    model_ids = ("gemini-2.5-flash", "gemini-2.0-flash")  # v1betaで利用可能なモデル（1.5-flashは404）
    last_error: Optional[str] = None
    try:
        client = genai.Client(api_key=key)
        for model_id in model_ids:
            try:
                response = client.models.generate_content(model=model_id, contents=prompt)
                if response and getattr(response, "text", None):
                    return response.text.strip().split("\n")[0]
                return "（応答なし）"
            except Exception as e:
                err_str = str(e)
                if "404" in err_str or "not found" in err_str.lower():
                    last_error = err_str
                    continue
                raise
        return f"API エラー（いずれのモデルも利用できません）: {last_error}"
    except Exception as e:
        logger.exception("Gemini generate_one_shot: %s", e)
        return f"API エラー: {e}"


# ----- 定性監査（SOTP乖離の「質」に特化・数値の重複評価禁止） -----

QUALITATIVE_AUDIT_PROMPT_TEMPLATE = """あなたの任務は、SOTP理論株価との乖離率の「質」を問う定性分析に特化することである。

【ハードリミット（絶対遵守ルール）】
1. 乖離率が +100% を超える銘柄は、市場が倒産リスクや深刻な構造的欠陥（不祥事等）を織り込んでいるバリュートラップ（罠）である。いかにチャートの買いサイン（逆三尊など）が点灯していても、**絶対に Rank A や B を付与してはならない。原則として Rank D (Avoid) とせよ。**
2. Rank A (Strong Buy) は、乖離率が +20% 〜 +80% の現実的な範囲にあり、かつ稼ぐ力（ROEやEBITDA）が健全で、チャートに反転サインが出ている「本物の是正期待株」にのみ付与すること。

【禁止事項】
「PBRが低いから割安」「PERが低いから買い」など、SOTPで既に計算済みの数値を再評価してはならない。数字そのものの羅列は禁止する。

【主要任務】
「なぜ市場はこの銘柄をSOTPより安く放置しているのか？」という **Why** の解明に集中せよ。入力の数値が示唆するのは「企業の体質」であり、数字そのものではない。その体質を読み解け。

【判定基準（4段階・グラデーションを意識せよ）】
- **Rank A (Strong Buy)**: 財務健全かつ割安（乖離率 +20%〜+80%）で、明確な強気材料（構造改革、高ROE、テクニカル好転）がある「本命」。
- **Rank B (Buy on Dip)**: 基礎体力は高いが、直近の悪材料で売られすぎている、または上昇トレンドの押し目待ち。「打診買い」の水準。
- **Rank C (Watch)**: 割安だが、成長触媒が不足している、または決算待ち。監視リストに入れるべき銘柄。
- **Rank D (Avoid)**: 構造的な赤字、ガバナンス欠如、乖離率 +100% 超、または回復の見込みが薄い「死に体」。

【思考プロセス】
- 迷ったらDにするのではなく、**致命的な欠陥がなければC以上を検討する**。
- わずかな懸念点（例: 海外リスク）だけでDにしない。それが株価に織り込み済みならBやCとするよう柔軟に評価せよ。
- ただし、乖離率 +100% 超はバリュートラップの強いシグナルであり、柔軟に評価する対象外である。

【出力の質】
「割安です」「回復期待です」などの定型文は禁止。マンネリを避け、**ライブ感のある表現**を優先せよ。
例: 「昨日の決算好決算を受け、見直し買いが入る」「節目を抜けたためトレンド発生」「〇〇報道で一時売られたが織り込み済み」。

銘柄データ:
{fundamental_line}

最後に、必ず1行で「Rank X:」に続けて、上記の質的結論を40文字以内で述べよ。"""


def _parse_upside_to_float(upside: Any) -> float:
    """乖離率を数値に変換。文字列 '+134.7%' や float に対応。"""
    if upside is None:
        return 0.0
    if isinstance(upside, (int, float)):
        return float(upside)
    s = re.sub(r"[^\d.\-+]", "", str(upside).strip())
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _trap_override(upside: Any, result: dict[str, Any]) -> dict[str, Any]:
    """
    乖離率100%超の場合は A/B を強制 D に上書き（LLM暴走防止）。
    """
    num = _parse_upside_to_float(upside)
    if num <= 100.0:
        return result
    rank = (result.get("rank") or "—").strip().upper()
    if rank not in ("A", "B"):
        return result
    return {
        **result,
        "rank": "D",
        "conclusion": "【システム強制判定】乖離率100%超えの異常値。倒産リスクや構造的欠陥（罠）とみなし除外。",
        "verdict": "AVOID",
    }


def qualitative_audit(
    ticker_symbol: str,
    upside: float,
    api_key: Optional[str] = None,
    secrets: Optional[Any] = None,
    fundamental_line: Optional[str] = None,
) -> dict[str, Any]:
    """
    銘柄に対して Gemini（Google Search 利用可）で定性監査を行い、
    投資適格ランク（A-D）とストラテジストの結論を返す。
    fundamental_line があれば PER/PBR/ROE 等を含む1行として渡す。

    戻り値: {"rank": "A"|"B"|"C"|"D", "conclusion": str (25文字以内), "verdict": "OK"|"AVOID"}
    """
    default_result = {
        "rank": "—",
        "conclusion": "（API未設定またはエラー）",
        "verdict": "OK",
    }
    if not _GEMINI_AVAILABLE or genai is None:
        return default_result

    key = api_key or get_gemini_api_key(secrets)
    if not key:
        return default_result

    line = fundamental_line if fundamental_line else f"{ticker_symbol} | Upside +{upside}%"
    prompt = QUALITATIVE_AUDIT_PROMPT_TEMPLATE.format(fundamental_line=line)

    model_ids = ("gemini-2.5-flash", "gemini-2.0-flash")  # v1betaで利用可能なモデル（1.5-flashは404）
    last_error: Optional[str] = None
    config_with_search = None
    if genai_types and hasattr(genai_types, "Tool") and hasattr(genai_types, "GoogleSearch"):
        try:
            config_with_search = genai_types.GenerateContentConfig(
                tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())]
            )
        except Exception:
            config_with_search = None

    try:
        client = genai.Client(api_key=key)
        for model_id in model_ids:
            try:
                response = None
                if config_with_search:
                    try:
                        response = client.models.generate_content(
                            model=model_id, contents=prompt, config=config_with_search
                        )
                    except Exception:
                        response = None
                if not response:
                    response = client.models.generate_content(model=model_id, contents=prompt)
                if not response or not getattr(response, "text", None):
                    continue
                text = response.text.strip()
                rank = "—"
                for r in ("A", "B", "C", "D"):
                    if f"Rank {r}" in text or f"rank {r}" in text or f"RANK {r}" in text:
                        rank = r
                        break
                conclusion = ""
                for line in text.split("\n"):
                    line = line.strip()
                    if "Rank " in line and ":" in line:
                        parts = line.split(":", 1)
                        if len(parts) >= 2:
                            conclusion = parts[1].strip()[:50]
                            break
                if not conclusion:
                    conclusion = text[:50].replace("\n", " ")
                verdict = "AVOID" if rank == "D" else "OK"
                out = {"rank": rank, "conclusion": conclusion or text[:50], "verdict": verdict}
                return _trap_override(upside, out)
            except Exception as e:
                err_str = str(e)
                if "404" in err_str or "not found" in err_str.lower() or "tools" in err_str.lower():
                    last_error = err_str
                    continue
                logger.warning("qualitative_audit (%s): %s", model_id, e)
                return {
                    "rank": "—",
                    "conclusion": f"API エラー: {str(e)[:30]}",
                    "verdict": "OK",
                }
        return {
            "rank": "—",
            "conclusion": f"利用可能なモデルなし: {last_error or 'unknown'}"[:25],
            "verdict": "OK",
        }
    except Exception as e:
        logger.exception("qualitative_audit: %s", e)
        return {
            "rank": "—",
            "conclusion": str(e)[:25],
            "verdict": "OK",
        }


# ----- 定性監査（バッチ・RPM削減） -----

DEFAULT_BATCH_SIZE = 5


def qualitative_audit_batch(
    items: list[dict[str, Any]],
    api_key: Optional[str] = None,
    secrets: Optional[Any] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> list[dict[str, Any]]:
    """
    複数銘柄を1リクエストで監査し、RPM制限を回避する。
    items: [{"ticker": str, "upside": float}, ...]（ticker と upside 必須）
    戻り値: [{"rank": "A"|"B"|"C"|"D", "conclusion": str, "verdict": "OK"|"AVOID"}, ...]
    入力順序を保持する。パース失敗時は1銘柄ずつフォールバック。
    progress_callback(done_count, total_count, message) でバッチごとに進捗通知可能。
    """
    default_one = {"rank": "—", "conclusion": "（未取得）", "verdict": "OK"}
    if not items:
        return []
    if not _GEMINI_AVAILABLE or genai is None:
        return [default_one.copy() for _ in items]
    key = api_key or get_gemini_api_key(secrets)
    if not key:
        return [default_one.copy() for _ in items]

    client = genai.Client(api_key=key)
    total = len(items)
    chunks: list[list[dict[str, Any]]] = []
    for i in range(0, len(items), batch_size):
        chunks.append(items[i : i + batch_size])

    model_ids = ("gemini-2.5-flash", "gemini-2.0-flash")  # v1betaで利用可能なモデル（1.5-flashは404）
    out: list[dict[str, Any]] = []
    for batch_idx, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(len(out), total, f"監査バッチ {batch_idx + 1}/{len(chunks)}")
        ticker_upside_lines = "\n".join(
            x.get("line") or f"{x['ticker']} 乖離+{x['upside']}%" for x in chunk
        )
        prompt = _build_batch_audit_prompt(chunk, ticker_upside_lines)
        parsed = _call_batch_and_parse(client, key, chunk, prompt, model_ids, secrets=secrets)
        out.extend(parsed)
        # 1バッチ終了ごとに5秒待機し、APIの1分あたりリクエスト数を制限内に収める
        if batch_idx < len(chunks) - 1:
            time.sleep(5)
    if progress_callback:
        progress_callback(total, total, "監査完了")
    return out


def _build_batch_audit_prompt(chunk: list[dict[str, Any]], ticker_upside_lines: str) -> str:
    """SOTP乖離の質に特化・4段階判定・New Signal・ライブ感のあるコメント。"""
    n = len(chunk)
    has_new_signal = any(
        "逆三尊" in str(x.get("line", "")) or "ピンバー" in str(x.get("line", ""))
        for x in chunk
    )
    new_signal_instruction = ""
    if has_new_signal:
        new_signal_instruction = "パターンに[NEW SIGNAL]が付いている銘柄は、そのサインが**今まさに点灯した**と仮定し、エントリーの緊急度を評価せよ。\n"
    return f"""任務: SOTP乖離の「質」の定性分析に特化。禁止: PBR/PERが低いから割安など数値の再評価。「なぜ市場はSOTPより安く放置しているか」のWhyに集中し、数値が示唆する企業の体質を読み解け。
{new_signal_instruction}【4段階】A(Strong Buy)=本命・強気材料明確。B(Buy on Dip)=打診買い・売られすぎor押し目待ち。C(Watch)=監視リスト・成長触媒不足or決算待ち。D(Avoid)=死に体・構造的赤字orガバナンス欠如。迷ったらDではなく、致命的欠陥がなければC以上を検討。わずかな懸念（海外リスク等）は織り込み済みならB/Cとする。
コメント禁止: 「割安です」「回復期待です」等の定型文。優先: ライブ感のある表現（例: 昨日の〇〇を受け見直し買い、節目抜けでトレンド発生）。
JSON以外の文字は1文字も出力するな。
以下の{n}銘柄を判定し、必ずこの形式のJSON配列のみ返せ: [{{"ticker":"コード","rank":"A","comment":"40文字以内の質的結論"}}, ...]

銘柄リスト:
{ticker_upside_lines}
"""


def _call_batch_and_parse(
    client: Any,
    key: str,
    chunk: list[dict[str, Any]],
    prompt: str,
    model_ids: tuple[str, ...],
    secrets: Optional[Any] = None,
) -> list[dict[str, Any]]:
    default_one = {"rank": "—", "conclusion": "（未取得）", "verdict": "OK"}

    def _fallback_single_audits() -> list[dict[str, Any]]:
        """バッチパース失敗時: 1銘柄ずつ qualitative_audit で再試行する。429回避のため2秒間隔を挟む。"""
        result_list: list[dict[str, Any]] = []
        for item in chunk:
            time.sleep(2)  # 429エラーを物理的に回避
            ticker = item.get("ticker", "")
            upside = item.get("upside", 0.0)
            single = qualitative_audit(
                ticker, upside, api_key=key, secrets=secrets,
                fundamental_line=item.get("line"),
            )
            result_list.append({
                "rank": single.get("rank", "—"),
                "conclusion": (single.get("conclusion") or "—")[:50],
                "verdict": single.get("verdict", "OK"),
            })
        return result_list

    def _is_rate_limit_error(err: Exception) -> bool:
        s = str(err).lower()
        return "429" in str(err) or "rate limit" in s or "resource exhausted" in s

    try:
        for model_id in model_ids:
            for attempt in range(3):
                try:
                    time.sleep(10)  # 超安全運転: API呼び出し直前に必ず10秒冷却
                    response = client.models.generate_content(model=model_id, contents=prompt)
                    if not response or not getattr(response, "text", None):
                        break
                    text = response.text.strip()
                    arr = _extract_json_array(text)
                    if not arr or not isinstance(arr, list):
                        break
                    result_list: list[dict[str, Any]] = []
                    for i, item in enumerate(chunk):
                        ticker = item.get("ticker", "")
                        row = _find_by_ticker(arr, ticker)
                        if not row and i < len(arr) and isinstance(arr[i], dict):
                            row = arr[i]  # インデックスベースのバックアップ照合
                        if not row or not isinstance(row, dict):
                            result_list.append(default_one.copy())
                            continue
                        r = str(row.get("rank", "—")).strip().upper()
                        rank = r if r in ("A", "B", "C", "D") else "—"
                        comment = str(row.get("comment", ""))[:50]
                        verdict = "AVOID" if rank == "D" else "OK"
                        one = {"rank": rank, "conclusion": comment or "—", "verdict": verdict}
                        upside_val = chunk[i].get("upside", 0) if i < len(chunk) else 0
                        result_list.append(_trap_override(upside_val, one))
                    if all(
                        r.get("conclusion") == "（未取得）" or r.get("rank") == "—"
                        for r in result_list
                    ):
                        logger.info("batch parse failed for chunk, falling back to single audits")
                        return _fallback_single_audits()
                    return result_list
                except Exception as e:
                    err_str = str(e)
                    if _is_rate_limit_error(e):
                        if attempt < 2:
                            logger.warning("429 Resource exhausted: 1分間完全停止して再試行 (%s/%s)", attempt + 1, 3)
                            time.sleep(60)  # 1分間完全停止
                            continue
                        logger.warning("429 が3回続いたためフォールバックへ")
                        return _fallback_single_audits()
                    if "404" in err_str or "not found" in err_str.lower():
                        break
                    logger.warning("qualitative_audit_batch (%s): %s", model_id, e)
                    break
    except Exception as e:
        logger.exception("qualitative_audit_batch: %s", e)
    # バッチ全体が失敗した場合も1銘柄ずつ再試行
    logger.info("batch API failed, falling back to single audits for %s items", len(chunk))
    return _fallback_single_audits()


def _extract_json_array(text: str) -> Optional[list]:
    """
    応答テキストから JSON 配列を強制抽出。
    マークダウン(```json...```)でもプレーンテキストでも、r'\[.*\]' (DOTALL) で [ ] の塊を抜き出す。
    """
    text = text.strip()
    # 正規表現で [ から ] までの塊を強制抽出（DOTALL で改行も含む）
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # マークダウンブロック内を優先して再試行
    mm = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if mm:
        try:
            return json.loads(mm.group(1))
        except json.JSONDecodeError:
            pass
    return None


def _find_by_ticker(arr: list, ticker: str) -> Optional[dict]:
    """配列要素の 'ticker' が一致するものを返す。"""
    for x in arr:
        if isinstance(x, dict) and str(x.get("ticker", "")).strip() == str(ticker).strip():
            return x
    return None
