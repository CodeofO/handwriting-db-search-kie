"""1단계: 제안 모델 사전학습 — 글자 인식(CTC)으로 이미지 인코더 + 인식 전용 층 + CTC 머리만 학습(비교군의 OCR 사전학습에 해당).

데이터: xa · xh · xi의 train 분할만(val · tune · test 이미지는 쓰지 않음). 이미지 1장 ↔ 그 이미지에 쓰인 문자열.
표집: 데이터셋 비율 ∝ (train 이미지 수)^α (α=0.5: 큰 데이터셋이 압도하지 않게), 데이터셋 안에서는 균등.
모델: 최종 구조(d 384 · 6층 · 인식 전용 3층 · 슬롯 16 · 64×1280), 어휘 = meta/x/vocab.json. 텍스트 쪽 · 검색 · 비교기는 학습하지 않는다.
증강: FT와 같은 이미지 증강(keep_ink).
선정: eval_every 스텝마다 세 데이터 val(큰 데이터는 고정 표본 val_cap장)의 CTC loss, 데이터셋 평균(macro) 최소 = final.pt.
      patience번 연속 개선이 없으면 조기 종료. 체크포인트는 모두 저장(가중치만, 이어 학습용 last.pt만 옵티마이저 포함).
출력: runs/x/{name}/{args.json, epochs/ep{k}.pt, final.pt, log.jsonl, best.json}
사용: torchrun --nproc_per_node 4 -m formrec.retrieval.pretrain --name pretrain_ctc [--steps 30000 --eval_every 2000] --resume
"""
import argparse
import json
import math
import queue
import shutil
import threading
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from formrec.common.workspace import ROOT
from formrec.retrieval import data as X
from formrec.retrieval.augment import augment
from formrec.retrieval.distributed import average_gradients, enable_tf32, finish, is_main, log, setup
from formrec.retrieval.model import RecordEncoder3

TRAINED = ("img.", "img_id.", "read.", "ctc_head.")       # 사전학습에서 학습하는 파라미터 접두사


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pretrain_ctc")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=128, help="GPU당 이미지 수")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--val_cap", type=int, default=20000)
    ap.add_argument("--read_layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dss", default="xa,xh,xi", help="스모크용으로 일부만")
    ap.add_argument("--cpu", action="store_true", help="CPU 스모크(gloo)")
    return ap.parse_args(argv)


def make_model(vocab_size, maxu, read_layers):
    return RecordEncoder3(vocab_size, maxu, 16, 3, X.H, X.W, d=384, dim=64, layers=6, read_layers=read_layers)


def dataset_probs(n, alpha):
    """데이터셋 표집 비율 ∝ n^α."""
    n = np.asarray(n, float)
    return n ** alpha / (n ** alpha).sum()


def val_samples(dss, cap):
    """val 고정 표본(모든 rank 같은 표본): 큰 데이터셋은 시드 123으로 cap장."""
    vrng = np.random.default_rng(123)
    val = {}
    for ds in dss:
        V = X.items(ds, "val")
        if len(V) > cap:
            V = V.iloc[np.sort(vrng.choice(len(V), cap, replace=False))].reset_index(drop=True)
        val[ds] = V
    return val


def sample_batch(rng, dss, p_ds, tr, imgs, bs):
    """→ uint8 (bs, H, W), 문자열 bs개, 데이터셋 번호 (bs,)."""
    dsi = rng.choice(len(dss), bs, p=p_ds)
    arr = np.empty((bs, X.H, X.W), np.uint8); txt = [None] * bs
    for k, ds in enumerate(dss):
        sel = np.where(dsi == k)[0]
        if len(sel) == 0:
            continue
        ii = rng.integers(len(tr[ds]), size=len(sel)); T_ = tr[ds]
        arr[sel] = imgs[ds].get(list(T_.src.values[ii]), list(T_.row.values[ii]))
        for s_, i_ in zip(sel, ii):
            txt[s_] = T_.text.values[i_]
    return arr, txt, dsi


def greedy_decode(row):
    """CTC 최선 경로: 반복 합치고 blank(0) 제거."""
    return [int(u) for t_, u in enumerate(row) if u != 0 and (t_ == 0 or u != row[t_ - 1])]


@torch.no_grad()
def val_loss(model, val, imgs, units, dev, rank=0, world=1, amp=False, bs=256):
    """데이터셋별 val CTC loss(단위당 정규화, 학습과 같은 단위) · 완전 일치율. rank별로 나눠 계산하고 all_reduce."""
    model.eval(); res = {}
    for ds, V in val.items():
        idx = np.arange(rank, len(V), world)
        tot = torch.zeros(3, device=dev)                    # nll 합, 이미지 수, 완전 일치 수
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            xb = torch.from_numpy(imgs[ds].get(list(V.src.values[j]), list(V.row.values[j]))).to(dev)
            x = augment(xb, X.H, X.W, False)[:, None]
            mk = torch.ones(len(j), 1, dtype=torch.bool, device=dev)
            tg = [units(t) for t in V.text.values[j]]
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                hi = model.image_tokens(x, mk)
            h = model.read_feats(hi[:, 0]).float()
            lp = F.log_softmax(model.ctc_head(h).float(), -1)
            ok = [k_ for k_, t in enumerate(tg) if t]
            if not ok:
                continue
            lpt = lp[ok].transpose(0, 1)
            n_u = torch.tensor([len(tg[k_]) for k_ in ok], device=dev)
            nll = F.ctc_loss(lpt, torch.tensor([u for k_ in ok for u in tg[k_]], device=dev), torch.full((len(ok),), lp.size(1), device=dev, dtype=torch.long),
                             n_u, blank=0, reduction="none", zero_infinity=True)
            nll = nll / n_u
            pr = lp.argmax(-1).cpu().numpy()
            em = sum(greedy_decode(pr[k_]) == tg[k_] for k_ in ok)
            tot += torch.tensor([float(nll.sum()), len(ok), em], device=dev)
        dist.all_reduce(tot)
        res[ds] = dict(loss=float(tot[0] / tot[1]), em=float(tot[2] / tot[1]), n=int(tot[1]))
    model.train()
    return res


def main(argv=None):
    a = parse_args(argv)
    DSS = tuple(a.dss.split(","))
    dev, rank, world = setup(a.cpu, 29917)
    if not a.cpu:
        enable_tf32()
    out = ROOT / "runs/x" / a.name
    if is_main():
        (out / "epochs").mkdir(parents=True, exist_ok=True); json.dump(vars(a), open(out / "args.json", "w"), indent=1)
    tok = X.load_tok()
    tr = {ds: X.items(ds, "train") for ds in DSS}
    imgs = {ds: X.XImages(ds) for ds in DSS}
    n = np.array([len(tr[ds]) for ds in DSS], float); p_ds = dataset_probs(n, a.alpha)
    log("train 이미지", dict(zip(DSS, n.astype(int))), "표집 비율", dict(zip(DSS, p_ds.round(3))), "어휘", len(tok.vocab))
    ucache = {}

    def units(t):
        if t not in ucache:
            ucache[t] = tok.units(t)
        return ucache[t]
    val = val_samples(DSS, a.val_cap)
    torch.manual_seed(a.seed)
    model = make_model(len(tok.vocab), tok.maxu, a.read_layers).to(dev)
    model = model.to(memory_format=torch.channels_last) if not a.cpu else model
    for p_ in model.parameters():
        dist.broadcast(p_.data, 0)
    train_names = [n_ for n_, _ in model.named_parameters() if n_.startswith(TRAINED)]
    params = [p_ for n_, p_ in model.named_parameters() if n_ in train_names]
    for n_, p_ in model.named_parameters():
        p_.requires_grad_(n_ in train_names)
    log(f"학습 파라미터 {sum(p.numel() for p in params) / 1e6:.2f}M / 전체 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd, fused=not a.cpu)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    start, best, bad, hist = 0, float("inf"), 0, []
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best, bad, hist = ck["step"], ck["best"], ck["bad"], ck["hist"]; log("resume", start)

    # 배치 미리 읽기 (CPU mmap 모으기를 GPU 계산과 겹침)
    rng = np.random.default_rng(a.seed * 1000 + rank + start)
    q = queue.Queue(maxsize=6)

    def producer():
        while True:
            arr, txt, dsi = sample_batch(rng, DSS, p_ds, tr, imgs, a.bs)
            t_ = torch.from_numpy(arr)
            q.put((t_ if a.cpu else t_.pin_memory(), txt, dsi))
    for _ in range(2):
        threading.Thread(target=producer, daemon=True).start()

    t0 = time.time()
    for step in range(start, a.steps):
        arr, txt, dsi = q.get()
        x = augment(arr.to(dev, non_blocking=True), X.H, X.W, True, keep_ink=True)[:, None]
        mk = torch.ones(a.bs, 1, dtype=torch.bool, device=dev)
        tg = [units(t) for t in txt]
        if step == start and is_main():                          # 입력 1회 점검: 실제 배치 그림 · 글자 단위 수 대비 열 토큰 수
            from formrec.common import input_check
            r = input_check.kie_batch(x.float().cpu().numpy(), mk.cpu().numpy(), txt, f"x_{a.name}_train_s{step}")
            T_ = X.W // 4; mx = max(len(t) for t in tg)
            log(f"[input_check] 열 토큰 {T_} · 이미지당 최대 글자 단위 {mx} · 표본 데이터셋 {np.bincount(dsi, minlength=3).tolist()}")
            if not r["pass"] or mx * 2 + 1 > T_:
                raise SystemExit("입력 점검 실패 — logs/checks/ 확인")
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=not a.cpu):
            hi = model.image_tokens(x, mk)
            loss = model.ctc_loss(hi, mk, tg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        average_gradients(params, world)                         # 1프로세스여도 거친다(기울기 없는 파라미터도 0 기울기로 갱신 — 원래 동작)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % 100 == 0 or step < 3:
            log(f"step {step}/{a.steps} ctc {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
        if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
            ep = (step + 1) // a.eval_every
            vr = val_loss(model, val, imgs, units, dev, rank, world, amp=not a.cpu); macro = float(np.mean([vr[ds]["loss"] for ds in DSS]))
            hist.append(dict(epoch=ep, step=step + 1, val_macro=macro, **{f"{ds}_{k}": v for ds in DSS for k, v in vr[ds].items()}))
            improved = macro < best
            if improved:
                best, bad = macro, 0
            else:
                bad += 1
            log(f"eval ep{ep}: val CTC loss macro {macro:.4f} " + " · ".join(f"{ds} {vr[ds]['loss']:.4f}/EM {vr[ds]['em']:.3f}" for ds in DSS)
                + (" ← 최소" if improved else f" (개선 없음 {bad}/{a.patience})"))
            if is_main():
                sd = model.state_dict()
                torch.save(dict(model=sd, vocab=tok.vocab, args=vars(a), epoch=ep, step=step + 1), out / "epochs" / f"ep{ep}.pt")
                if improved:
                    shutil.copy(out / "epochs" / f"ep{ep}.pt", out / "final.pt")
                    json.dump(dict(best_epoch=ep, best_step=step + 1, criterion="val CTC loss (데이터셋 평균)", val=hist[-1]), open(out / "best.json", "w"), indent=1)
                with open(out / "log.jsonl", "a") as f:
                    f.write(json.dumps(hist[-1]) + "\n")
                torch.save(dict(model=sd, opt=opt.state_dict(), sched=sched.state_dict(), step=step + 1, best=best, bad=bad, hist=hist), out / "last.pt")
            if bad >= a.patience:
                log(f"조기 종료: {a.patience}회 연속 개선 없음")
                break
    if is_main():
        (out / "last.pt").unlink(missing_ok=True)
        log("DONE", a.name, "best", json.load(open(out / "best.json"))["best_epoch"])
    finish()


if __name__ == "__main__":
    main()
