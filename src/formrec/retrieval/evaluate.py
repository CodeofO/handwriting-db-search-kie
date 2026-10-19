"""제안 모델 평가 — 단계별 레코드 검색(1등 레코드 = 정답, Exact Match와 같은 기준).

추론: 1단 검색(슬롯 벡터 내적) 상위 K개 → 2단 재채점(학습된 1단 · PLI 결합 + (mix_ctc + λ)·글자 검증) → 1등.
  확신도 conf = 수락 판정 머리 점수(1·2등 점수 차는 margin_conf로 보관). open_* = 정답을 DB에서 뺀 조건(DB 밖 질의 모의).
  s1 = 1등 후보의 글자 검증 점수(단위당 −CTC NLL, 비교군 CTC 재채점의 s1과 같은 성격), 결합 점수는 fused_s1.
단계(질의 = 그 세트의 레코드)
  xa : L1 이름 · L2~L6 주소 k토큰 · F2 이름+생년월일 · F3 +주소4, 서브 = F3@20k · F3@2M · F3#d0~2
  xh : H2 · H3 · H4 · H5+ (이름 줄 단어 수)      xi : I1(≤5단어) · I2(6~7) · I3(8~9) · I4(10~11) · I5(≥12)      xc : C2 · C3
DB
  xa : meta/kie_db_{단계}.parquet (단계마다 20만, 절반은 토큰 1개만 다른 이웃)
  xh : HANA 전체 이름 줄의 고유 문자열 — meta/x/xh_db.parquet       xi : IAM 전체 줄 + 단어 1개만 바꾼 이웃 — meta/x/xi_db.parquet
  xc : xa F2 · F3 DB의 칼럼 구분 기호를 공백으로
모드
  select : tune에서 2단 후보 수 K · 글자 검증 추가 가중치 λ를 고른다(단계 평균 정확도 최대, 동률이면 작은 K · 작은 λ) → runs/x/{name}/sel.json
  eval   : sel.json 설정으로 tune · test 1회 → results/{ds}/scores/P-{name}__{단계}__{세트}.parquet (+ _t1 = 1단만), latency
  epochs : 에폭 체크포인트마다 val 단계 평균 정확도(K 16, λ 0) → runs/x/{name}/epochs_metric.csv (부록)
  db     : xh · xi DB 파일만 만든다(formrec.data_prep.search_db, 없을 때만)
사용: python -m formrec.retrieval.evaluate --ds xa --name xa_prop --mode select|eval|epochs [--smoke N --cap N]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from formrec.common.levels import LEVELS, SEP, blank, level_table, parse_level
from formrec.common.levels import db_texts as kie_db_texts
from formrec.common.workspace import ROOT
from formrec.data_prep.search_db import build_dbs
from formrec.retrieval import data as X
from formrec.retrieval.augment import augment
from formrec.retrieval.model import RecordEncoder3
from formrec.retrieval.tokenizer import UnitTok3, field_char_lengths

SUBLV = {"xa": ["F3@20k", "F3@2M", "F3#d0", "F3#d1", "F3#d2"], "xh": [], "xi": [], "xc": []}
KS = (8, 16, 32, 64)
LAMS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
EPOCH_K = 16


# ---------------------------------------------------------------- DB
class LevelDB:
    """단계 → DB 문자열(정렬된 고유값). smoke > 0이면 DB = 질의 정답 + 앞 smoke×10개(CPU 스모크)."""
    def __init__(self, ds, smoke=0):
        self.ds, self.smoke, self._c = ds, smoke, {}

    def base(self, level):
        if self.ds == "xa":
            t = kie_db_texts("kie", level)
            return self.fix20k(t) if "@20k" in level else t
        if self.ds == "xc":                                      # 한 줄 여러 칼럼 DB = 같은 내용의 F2 · F3 DB(칼럼 구분 → 공백)
            k = "xc_" + level
            if k not in self._c:
                t = pd.read_parquet(ROOT / f"meta/kie_db_{ {'C2': 'F2', 'C3': 'F3'}[level]}.parquet").text.values
                self._c[k] = np.unique(np.array([x.replace(SEP, " ") for x in t]))
            return self._c[k]
        if self.ds not in self._c:
            self._c[self.ds] = pd.read_parquet(ROOT / f"meta/x/{self.ds}_db.parquet").text.values
        return self._c[self.ds]

    def fix20k(self, t):
        """meta/kie_db_F3_20k는 이전 분할(val · test 정답 + 이웃)로 만들어 새 tune 정답 일부가 빠져 있다.
        새 tune · test F3 정답을 모두 넣고, 같은 수만큼 정답이 아닌 항목을 무작위로 빼서 규모 2만을 유지한다(tune · test 같은 DB)."""
        if "20k" in self._c:
            return self._c["20k"]
        gt = set()
        for sub in ("tune", "test"):
            L = level_table("xa", sub, X.items("xa", sub)); gt |= set(L[L.level == "F3"].text)
        miss = sorted(gt - set(t)); others = np.array(sorted(set(t) - gt))
        drop = set(np.random.default_rng(0).choice(len(others), len(miss), replace=False).tolist())
        keep = [x for i, x in enumerate(others) if i not in drop]
        self._c["20k"] = np.array(sorted(set(keep) | gt)); print(f"F3@20k DB: 새 tune 정답 {len(miss):,}개 추가 · 같은 수 제거 → {len(self._c['20k']):,}", flush=True)
        return self._c["20k"]

    def __call__(self, level, query_text):
        t = self.base(level)
        if self.smoke:
            t = np.unique(np.concatenate([np.asarray(query_text, dtype=object), t[:self.smoke * 10]]).astype(str))
        return t


# ---------------------------------------------------------------- 질의
class XQueries:
    """세트의 단계별 질의: get(level) → (pid, 정답 문자열, 이미지 행 (n, 3)). 이미지는 그 세트 전부를 메모리에(64×1280 uint8).
    cap: 큰 세트(xh) 질의 표본 상한, smoke: 단계마다 앞 N개만."""
    def __init__(self, ds, sub, cap=0, smoke=0):
        D = X.items(ds, sub)
        if cap and len(D) > cap and ds != "xa":
            D = D.iloc[np.sort(np.random.default_rng(123).choice(len(D), cap, replace=False))].reset_index(drop=True)
        self.X = X.XImages(ds).get(list(D.src), list(D.row))
        self.row = {i: r for r, i in enumerate(D.item_id)}
        self.L = level_table(ds, sub, D)
        self.smoke = smoke

    def get(self, level):
        base, _, drop = parse_level(level)
        g = self.L[self.L.level == base]
        rows = np.full((len(g), 3), -1, np.int64)
        for i, its in enumerate(g["items"]):
            r = [self.row[x] for k, x in enumerate(its) if k != drop]
            rows[i, :len(r)] = r
        text = g.text.values if drop is None else np.array([blank(t, drop) for t in g.text.values])
        pid = g.pid.values
        if self.smoke:
            pid, text, rows = pid[:self.smoke], text[:self.smoke], rows[:self.smoke]
        return pid, text, rows


# ---------------------------------------------------------------- 모델 · 추론
def load_model(path, dev):
    """FT 체크포인트 → (모델 eval, 토크나이저, 학습 args). 최종 프로토콜 구조만 받는다(strict)."""
    ck = torch.load(path, map_location=dev, weights_only=False)
    a = ck["args"]
    assert a.get("vfeat", 14) == RecordEncoder3.NVF and not a.get("dynw", 0) and not a.get("glyph_sum", 0) and not a.get("len_w", 0), \
        "최종 프로토콜 밖의 구조(vfeat · dynw · glyph_sum · len_w)"
    tok = UnitTok3.from_vocab(ck["vocab"])
    m = RecordEncoder3(len(tok.vocab), tok.maxu, a["slots"], 3, a["H"], a["W"], d=a["d"], dim=a["dim"], layers=a["layers"],
                       read_layers=a.get("read_layers", 0)).to(dev)
    m.load_state_dict(ck["model"], strict=True)
    m.eval()
    return m, tok, a


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def enc_text_slots(m, tok, texts, dev, bs=2048):
    """DB 문자열 → 검색 벡터 (N, slots·dim) fp16."""
    out = []
    for i in range(0, len(texts), bs):
        ids, fld, pos, role = tok.batch(list(texts[i:i + bs]), dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out.append(RecordEncoder3.flat(m.slots_text(m.text_tokens(ids, fld, pos, role), ids).float()).half())
    return torch.cat(out)


CHECKED = []          # 입력 1회 점검을 한 번만


@torch.no_grad()
def run_level(m, tok, a, Q, lv, dev, db_fn, K=64, qbs=64, lam=0.0, keep_raw=False):
    """단계 하나 평가 → (결과 {"final", "t1"} 프레임, 지연 시간, DB 크기[, λ 선택용 원점수]).
    db_fn(level, 질의 정답) → DB 문자열. lam: 글자 검증 점수 추가 가중치(tune에서 고름, 학습된 mix_ctc와 합해 쓴다)."""
    use_v = a.get("verify_w", 0) > 0                                         # 수락 판정 머리로 확신도
    use3 = a.get("ctc_w", 0) > 0 and (keep_raw or lam != 0 or float(m.mix_ctc) != 0 or use_v)
    raw = dict(base=[], s3=[], top=[], gt=[])
    H, W = a["H"], a["W"]
    pid, text, rows = Q.get(lv)
    db = db_fn(lv, text)
    idx = {t: i for i, t in enumerate(db)}
    gt = np.array([idx[t] for t in text])
    t0 = time.time(); zdb = enc_text_slots(m, tok, db, dev); t_db = time.time() - t0
    n = len(rows); res = {k: [] for k in ("t1_top", "t1_s1", "t1_s2", "t1_op", "t1_os1", "t1_os2", "top", "s1", "s2", "op", "os1", "os2", "rec",
                                          "p1", "op1", "g1", "og1", "vc", "ovc")}
    tm = dict(img=0.0, s1=0.0, s2=0.0)
    for i in range(0, n, qbs):
        r_ = torch.from_numpy(rows[i:i + qbs]); mk = r_ >= 0
        x = torch.ones(len(r_), r_.size(1), 1, H, W)
        x[mk] = augment(torch.from_numpy(np.stack([Q.X[j] for j in r_[mk].tolist()])), H, W, False)
        if not CHECKED:
            from formrec.common import input_check
            chk = input_check.kie_eval_inputs(x[:4].numpy(), list(text[i:i + 4]), f"kie_v3_eval_{lv}", train_shape=(H, W))
            if not chk["pass"]:
                raise SystemExit("입력 점검 실패 — logs/checks/ 확인")
            CHECKED.append(lv)
        x = x.to(dev); mk = mk.to(dev)
        _sync(dev); t0 = time.time()
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            hi = m.image_tokens(x, mk); zq = RecordEncoder3.flat(m.slots_image(hi, mk).float()).half()
        _sync(dev); tm["img"] += time.time() - t0; t0 = time.time()
        # 1단: 전수 내적 검색, 후보 K+1개(DB 밖 조건용 여유 1)
        s = zq.float() @ zdb.float().T                                        # (q, N)
        g = torch.as_tensor(gt[i:i + qbs], device=dev)
        v, top = s.topk(min(K + 1, s.size(1)), 1)
        s_open = s.scatter(1, g[:, None], -1e9); ov, otop = s_open.topk(2, 1)
        res["t1_top"].append(top[:, 0].cpu()); res["t1_s1"].append(v[:, 0].cpu()); res["t1_s2"].append(v[:, 1].cpu())
        res["t1_op"].append(otop[:, 0].cpu()); res["t1_os1"].append(ov[:, 0].cpu()); res["t1_os2"].append(ov[:, 1].cpu())
        res["rec"].append((top[:, :K] == g[:, None]).any(1).cpu())
        _sync(dev); tm["s1"] += time.time() - t0; t0 = time.time()
        # 2단: 후보 텍스트 토큰 → PLI · 글자 검증
        cand_txt = [db[j] for j in top.flatten().tolist()]
        cand = torch.arange(len(cand_txt), device=dev).view(top.shape)
        ids, fld, pos, role = tok.batch(cand_txt, dev); fl = field_char_lengths(cand_txt).to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            ht = m.text_tokens(ids, fld, pos, role)
            ext = m.ink_extent(x, mk)
            r_ = m.pli(hi, ext, mk, ht, fld, pos, ids, fl, cand, per_slot=use_v)
        sc, pp, ppm = (r_[0].float(), r_[1].float(), r_[2]) if use_v else (r_.float(), None, None)
        sp = sc
        sc = m.final_score(v.float(), sc)                                      # 학습된 1단 · 2단 결합
        if use3:
            r3 = m.cand_ctc(hi.float(), mk, [tok.field_units(t) for t in cand_txt], cand, per_slot=use_v)
            s3, pg, pgm = r3 if use_v else (r3, None, None)
            if keep_raw:
                raw["base"].append((sc + float(m.mix_ctc) * s3).cpu()); raw["s3"].append(s3.cpu()); raw["top"].append(top.cpu()); raw["gt"].append(g.cpu())
            sc = sc + (float(m.mix_ctc) + lam) * s3
        else:
            s3 = torch.zeros_like(sc)
        kc = min(K, top.size(1))
        sc_c = sc[:, :kc]                                                      # 닫힌 조건: 1단 상위 K
        v2, o2 = sc_c.topk(2, 1)
        res["top"].append(top.gather(1, o2[:, :1])[:, 0].cpu()); res["s1"].append(v2[:, 0].cpu()); res["s2"].append(v2[:, 1].cpu())
        sc_o = sc.masked_fill(top == g[:, None], -1e9)                         # 열린 조건: 정답 제외
        ov2, oo2 = sc_o.topk(2, 1)
        res["op"].append(top.gather(1, oo2[:, :1])[:, 0].cpu()); res["os1"].append(ov2[:, 0].cpu()); res["os2"].append(ov2[:, 1].cpu())
        T_ = float(m.pli_scale.exp().clamp(max=100))                           # 학습 온도의 후보 소프트맥스 1등 확률, 1등 후보의 글자 검증 점수
        res["p1"].append(torch.softmax(T_ * sc_c, 1).max(1).values.cpu()); res["op1"].append(torch.softmax(T_ * sc_o, 1).max(1).values.cpu())
        res["g1"].append(s3.gather(1, o2[:, :1])[:, 0].cpu()); res["og1"].append(s3.gather(1, oo2[:, :1])[:, 0].cpu())
        if use_v:                                                              # 닫힌 · 열린 1등의 수락 판정 점수
            sf = sc - lam * s3                                                 # 특징 = 학습 때와 같은 점수(학습 mix_ctc만), 1·2등 선택 = 최종 점수(λ 포함)
            fc, _ = m.verify_feats(sf[:, :kc], sp[:, :kc], s3[:, :kc], pp[:, :kc], ppm[:, :kc], pg[:, :kc], pgm[:, :kc], v.float()[:, :kc], rank=sc_c)
            fo, _ = m.verify_feats(sf.masked_fill(top == g[:, None], -1e9), sp, s3, pp, ppm, pg, pgm, v.float(), rank=sc_o)
            res["vc"].append(m.verify(fc).float().cpu()); res["ovc"].append(m.verify(fo).float().cpu())
        _sync(dev); tm["s2"] += time.time() - t0
    R = {k: torch.cat(v).numpy() for k, v in res.items() if v}
    out = {}
    for tag, pre in (("final", ""), ("t1", "t1_")):
        pred = R[pre + "top"]
        out[tag] = pd.DataFrame(dict(pid=pid, text=text, pred=db[pred], correct=pred == gt, conf=R[pre + "s1"] - R[pre + "s2"], s1=R[pre + "s1"],
                                     open_pred=db[R[pre + "op"]], open_conf=R[pre + "os1"] - R[pre + "os2"], open_s1=R[pre + "os1"]))
        if tag == "final":
            out[tag]["p1"], out[tag]["open_p1"], out[tag]["g1"], out[tag]["open_g1"] = R["p1"], R["op1"], R["g1"], R["og1"]
            if use3:   # 1등 절대 점수 = 1등 후보의 글자 검증 점수, 결합 점수는 fused_s1로
                out[tag]["fused_s1"], out[tag]["fused_open_s1"] = out[tag]["s1"], out[tag]["open_s1"]
                out[tag]["s1"], out[tag]["open_s1"] = R["g1"], R["og1"]
            if use_v:  # 확신도 = 수락 판정 머리 점수 (마진은 margin_conf로)
                out[tag]["margin_conf"], out[tag]["margin_open_conf"] = out[tag]["conf"], out[tag]["open_conf"]
                out[tag]["conf"], out[tag]["open_conf"] = R["vc"], R["ovc"]
    lat = dict(ms_img=1000 * tm["img"] / n, ms_search=1000 * tm["s1"] / n, ms_rerank=1000 * tm["s2"] / n, s_db=t_db, recall_at_K=float(R["rec"].mean()))
    if keep_raw:
        return out, lat, len(db), {k: torch.cat(v) for k, v in raw.items()} if raw["top"] else None
    return out, lat, len(db)


def select_grid(raws, ks=KS, lams=LAMS):
    """단계별 원점수 → {(K, λ): 단계 평균 정확도}, 고른 (K, λ) (최대, 동률이면 작은 K · 작은 λ)."""
    grid = {}
    for K in ks:
        for lam in lams:
            accs = [float((r["top"].gather(1, (r["base"] + lam * r["s3"])[:, :K].argmax(1, keepdim=True)) == r["gt"][:, None]).float().mean()) for r in raws]
            grid[(K, lam)] = float(np.mean(accs))
    best = max(grid, key=lambda k: (round(grid[k], 6), -k[0], -k[1]))
    return grid, best


# ---------------------------------------------------------------- 모드
def mode_select(a, run, dev, db_fn, lvs):
    m, tok, args = load_model(run / "final.pt", dev)
    Q = XQueries(a.ds, "tune", a.cap, a.smoke); raws = []
    for lv in lvs:
        o, lat, ndb, raw = run_level(m, tok, args, Q, lv, dev, db_fn, max(KS), keep_raw=True); raws.append(raw)
        print(f"tune {lv}: n={len(o['final'])} K{max(KS)} 정확도 {o['final'].correct.mean():.4f}", flush=True)
    grid, (K, lam) = select_grid(raws)
    json.dump(dict(K=K, lam=lam, criterion="tune 단계 평균 정확도", grid={f"{k[0]}|{k[1]}": v for k, v in grid.items()}), open(run / "sel.json", "w"), indent=1)
    print("tune 격자 (K별 최고)", {k_: max(v for (kk, _), v in grid.items() if kk == k_) for k_ in KS}, "→ K", K, "λ", lam, flush=True)
    print("DONE x_eval select", a.name)


def mode_eval(a, run, dev, db_fn, lvs):
    m, tok, args = load_model(run / "final.pt", dev)
    S = json.load(open(run / "sel.json")); K, lam = S["K"], S["lam"]
    sd = ROOT / f"results/{a.ds}/scores"; sd.mkdir(parents=True, exist_ok=True)
    latp = ROOT / f"results/{a.ds}/latency/P-{a.name}.json"; latp.parent.mkdir(parents=True, exist_ok=True)
    L_all = {}
    for sub in ("tune", "test"):
        Q = XQueries(a.ds, sub, smoke=a.smoke)
        for lv in lvs + SUBLV[a.ds]:
            o, lat, ndb = run_level(m, tok, args, Q, lv, dev, db_fn, K, lam=lam)
            o["final"].to_parquet(sd / f"P-{a.name}__{lv}__{sub}.parquet", index=False)
            o["t1"].to_parquet(sd / f"P-{a.name}_t1__{lv}__{sub}.parquet", index=False)
            L_all[f"{sub}_{lv}"] = dict(lat, ms_search=lat["ms_search"] + lat["ms_rerank"])
            print(f"{sub} {lv}: n={len(o['final'])} DB={ndb} 1단 {o['t1'].correct.mean():.4f} → 2단 {o['final'].correct.mean():.4f} "
                  f"(후보 {K} 재현율 {lat['recall_at_K']:.4f}) img {lat['ms_img']:.2f} · 검색+재채점 {lat['ms_search'] + lat['ms_rerank']:.2f} ms", flush=True)
    json.dump(L_all, open(latp, "w"), indent=1)
    print("DONE x_eval eval", a.name)


def mode_epochs(a, run, dev, db_fn, lvs):
    Q = XQueries(a.ds, "val", a.cap, a.smoke); rows = []
    eps = sorted((run / "epochs").glob("ep*.pt"), key=lambda p: int(p.stem[2:]))
    for p in eps:
        m, tok, args = load_model(p, dev); accs = {}
        for lv in lvs:
            o, lat, ndb = run_level(m, tok, args, Q, lv, dev, db_fn, EPOCH_K)
            accs[lv] = float(o["final"].correct.mean())
        rows.append(dict(epoch=int(p.stem[2:]), acc=float(np.mean(list(accs.values()))), **accs)); print("epoch", rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(run / "epochs_metric.csv", index=False)
    R = pd.DataFrame(rows); b = R.sort_values(["acc", "epoch"]).iloc[-1]
    print(f"DONE x_eval epochs {a.name}: 지표 기준 최고 에폭 {int(b.epoch)} (val 단계 평균 {b.acc:.4f})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=list(X.DSS) + ["xc"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", required=True, choices=["select", "eval", "epochs", "db"])
    ap.add_argument("--cap", type=int, default=20000, help="큰 세트(HANA) 질의 표본 상한: select · epochs")
    ap.add_argument("--smoke", type=int, default=0, help="CPU 스모크: 단계마다 질의 N개 · DB = 질의 정답 + 앞 N×10개")
    a = ap.parse_args(argv)
    build_dbs()
    if a.mode == "db":
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    db_fn = LevelDB(a.ds, a.smoke)
    run = ROOT / "runs/x" / a.name
    {"select": mode_select, "eval": mode_eval, "epochs": mode_epochs}[a.mode](a, run, dev, db_fn, LEVELS[a.ds])


if __name__ == "__main__":
    main()
