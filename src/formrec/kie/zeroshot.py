"""손글씨 KIE 영점(zero-shot) 시험: 로컬 VLM 게이트웨이(OpenAI 호환 /chat/completions)에 서식 영역 이미지 1장 + 키 목록을 주고
키 순서대로 값을 받는다. 학습 없음 · 검출 없음.
접속 정보: 환경변수 VLM_BASE_URL · VLM_API_KEY · VLM_MODEL, 없으면 .env 파일(FORMREC_ENV_FILE, 작업 공간/.env, 저장소/.env 순)에서 읽는다.
  값은 출력하지 않는다(오류도 종류만 남긴다). 저장소에는 .env.example만 둔다.
채점: 키별 완전 일치(공백 정규화, 생년월일은 숫자만), 질의 전체 완전 일치.
출력: results/xk/zs_{tag}_{subset}.parquet (qid, keys, target, raw, pred, ok, all_ok, n_keys)
사용: python -m formrec.kie.zeroshot --subset test --n 200 --workers 4 --tag qwen
"""
import argparse
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from formrec.common import workspace as cfg
from formrec.kie.data import blob_bytes
from formrec.kie.fields import norm, parse_loose, target_dict

ROOT = cfg.ROOT
ENV_KEYS = ("VLM_BASE_URL", "VLM_API_KEY", "VLM_MODEL")
PROMPT = ("이 이미지는 손글씨로 작성된 한국어 서식의 일부입니다. 아래 항목의 값을 이미지에서 찾아, 주어진 항목 순서대로 한 줄에 하나씩 "
          "'항목: 값' 형식으로만 답하세요. 설명은 쓰지 마세요.\n"
          "- 이름: 성명 칸의 손글씨\n- 생년월일: 주민등록번호 앞 6자리 숫자\n- 주소: 주소 칸의 손글씨 전체(시 · 도부터, 띄어쓰기 유지)\n"
          "- 시도: 주소의 첫 단어\n- 시군구: 주소의 두 번째 단어\n\n항목: {keys}")


def read_env_file(path):
    e = {}
    for line in open(path, encoding="utf-8"):
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.strip().split("=", 1)
            e[k] = v.strip().strip('"').strip("'")
    return e


def connection():
    """VLM 접속 정보 {VLM_BASE_URL, VLM_API_KEY, VLM_MODEL}. 환경변수가 .env보다 우선."""
    e = {}
    files = [os.environ.get("FORMREC_ENV_FILE"), ROOT / ".env", cfg.REPO / ".env"]
    for f in files:
        if f and Path(f).is_file():
            e = read_env_file(f)
            break
    e.update({k: os.environ[k] for k in ENV_KEYS if os.environ.get(k)})
    missing = [k for k in ENV_KEYS if not e.get(k)]
    if missing:
        raise SystemExit(f"VLM 접속 정보 없음: {missing} (.env.example 참고)")
    return e


def sample_queries(subset, n):
    Q = pd.read_parquet(ROOT / f"meta/x/xk_queries_{subset}.parquet")
    return Q.iloc[np.sort(np.random.default_rng(0).choice(len(Q), min(n, len(Q)), replace=False))].reset_index(drop=True)


def score_rows(Q, raws):
    rows = []
    for i, raw in enumerate(raws):
        keys, tg = json.loads(Q["keys"][i]), target_dict(Q.target[i])
        pr = parse_loose(re.sub(r"<think>.*?</think>", "", raw, flags=re.S), keys)     # 추론형 모델의 생각 구간 제거
        ok = {k: norm(k, pr.get(k, "")) == norm(k, tg[k]) for k in keys}
        rows.append(dict(qid=Q.qid[i], keys=Q["keys"][i], target=Q.target[i], raw=raw, pred=json.dumps(pr, ensure_ascii=False),
                         ok=json.dumps(ok, ensure_ascii=False), all_ok=all(ok.values()), n_keys=len(keys)))
    return pd.DataFrame(rows)


def main(argv=None):
    import requests
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="qwen")
    a = ap.parse_args(argv)
    E = connection()
    url = E["VLM_BASE_URL"].rstrip("/") + "/chat/completions"
    hd = {"Authorization": f"Bearer {E['VLM_API_KEY']}"}
    Q = sample_queries(a.subset, a.n)
    imgs = [base64.b64encode(blob_bytes("all", int(r))).decode() for r in Q.row]

    def ask(i):
        keys = json.loads(Q["keys"][i])
        body = {"model": E["VLM_MODEL"], "temperature": 0, "max_tokens": 4096,
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + imgs[i]}},
                                                          {"type": "text", "text": PROMPT.format(keys=", ".join(keys))}]}]}
        err = ""
        for t in range(3):
            try:
                r = requests.post(url, json=body, headers=hd, timeout=300)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except Exception as e:                       # 일시 오류만 재시도 (오류 내용에 접속 정보가 섞이지 않게 종류만 남김)
                err = type(e).__name__
                time.sleep(5 * (t + 1))
        return f"[오류 {err}]"

    t0, raws = time.time(), []
    with ThreadPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(ask, range(len(Q)))):
            raws.append(r)
            if (k + 1) % 20 == 0:
                print(f"{k + 1}/{len(Q)} {time.time() - t0:.0f}s", flush=True)
    R = score_rows(Q, raws)
    out = ROOT / "results/xk"
    out.mkdir(parents=True, exist_ok=True)
    R.to_parquet(out / f"zs_{a.tag}_{a.subset}.parquet", index=False)
    per = {}
    for o in R.ok:
        for k, v in json.loads(o).items():
            per.setdefault(k, []).append(v)
    print("키별 정확도", {k: f"{np.mean(v):.3f} (n={len(v)})" for k, v in per.items()})
    print(f"질의 완전 일치 {R.all_ok.mean():.3f} (n={len(R)}) · 시간 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
