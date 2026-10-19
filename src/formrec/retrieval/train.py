"""2단계: 제안 모델 데이터셋별 FT — xa(AI Hub) · xh(HANA) · xi(IAM) · xc(xa 한 줄 이어 붙임).

손실(전부 한 모델 안에서 함께 학습)
  1단 대칭 InfoNCE(전 GPU 배치, 같은 문자열 = 다중 양성, 이미지→텍스트 쪽은 합성 음성 포함) + 슬롯 다양성
  + 2단 PLI 교차 엔트로피(후보 = 정답 · 합성 음성 3 · 배치 안 1단 최상위 pli_hard개; 1단 코사인 · 글자 검증 점수 결합)
  + 수락 판정 머리 BCE(닫힌 1등 = 정답 여부, 정답을 뺀 1등 = 음성) + 보조 CTC.
합성 음성: 글자 1개 치환(데이터셋 글자 집합에서 같은 종류) · 필드 교체 · 토큰 1개 교체(train 같은 자리 토큰).
초기화: --init = 사전학습(pretrain) final.pt → 이미지 인코더 · 인식 가지 · CTC 머리, 나머지는 같은 시드의 새 값. 비우면 처음부터(절제).
에폭 = 고정 스텝(--spe)으로 데이터셋 간 학습량을 맞춘다. 에폭마다 val 고정 표본(시드 고정 · 증강 없음 · eval 모드)으로 학습과 같은 손실식.
선정 = val loss 최소 에폭(final.pt, best.json). --patience번 연속 개선 없으면 조기 종료. 에폭 가중치는 모두 저장.
출력: runs/x/{name}/{args.json, epochs/ep{k}.pt, final.pt, best.json, log.jsonl}
사용: torchrun --nproc_per_node 4 -m formrec.retrieval.train --data xa --name xa_prop --init runs/x/pretrain_ctc/final.pt --resume
"""
import argparse
import json
import math
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn as dnn
import torch.nn.functional as F

from formrec.common.workspace import ROOT
from formrec.retrieval import data as X
from formrec.retrieval.augment import augment
from formrec.retrieval.distributed import average_gradients, enable_tf32, finish, is_main, log, setup
from formrec.retrieval.model import RecordEncoder3
from formrec.retrieval.tokenizer import field_char_lengths

MAX_IMAGES = 3
N_NEG = 3


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True, choices=list(X.DSS) + ["xc"])
    ap.add_argument("--init", default="")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--spe", type=int, default=401, help="에폭당 스텝(고정)")
    ap.add_argument("--bs", type=int, default=64, help="GPU당 레코드 수")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=1.0, help="워밍업 에폭 수")
    ap.add_argument("--shuffle_fields", type=int, default=1)
    ap.add_argument("--field_drop", type=float, default=0.15)
    ap.add_argument("--slot_drop", type=float, default=0.1)
    ap.add_argument("--div_w", type=float, default=0.1)
    ap.add_argument("--pli_w", type=float, default=1.0)
    ap.add_argument("--pli_hard", type=int, default=12)
    ap.add_argument("--ctc_w", type=float, default=1.0)
    ap.add_argument("--ctc_mix", type=int, default=1)
    ap.add_argument("--mix", type=int, default=1)
    ap.add_argument("--verify_w", type=float, default=1.0)
    ap.add_argument("--read_layers", type=int, default=3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--val_recs", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0, help="스모크용 전체 스텝")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="CPU 스모크(gloo)")
    a = ap.parse_args(argv)
    a.H, a.W, a.d, a.layers, a.slots, a.dim = X.H, X.W, 384, 6, 16, 64      # evaluate.load_model이 읽는 구조 정보
    return a


def make_model(a, vocab_size, maxu):
    return RecordEncoder3(vocab_size, maxu, a.slots, MAX_IMAGES, a.H, a.W, d=a.d, dim=a.dim, layers=a.layers, read_layers=a.read_layers)


class FieldUnits:
    """레코드 문자열 → 필드별 CTC 단위 목록 (캐시)."""
    def __init__(self, tok):
        self.tok, self.cache = tok, {}

    def __call__(self, t):
        if t not in self.cache:
            self.cache[t] = self.tok.field_units(t)
        return self.cache[t]


def build_batch(D, recs, rng, train, imgs, pools, tokv, dev):
    """레코드 목록 → (정답 문자열 B, 합성 음성 3B, 이미지 (B, 3, 1, H, W), 이미지 마스크 (B, 3))."""
    built = [D.record(p, f) for p, f in recs]
    keys = [X.field_kinds(f) for _, f in recs]
    txt = [b[0] for b in built]
    B = len(recs)
    img_rows = np.full((B, MAX_IMAGES), -1, np.int64)
    for i, b in enumerate(built):
        img_rows[i, :len(b[1])] = b[1]
    negs = X.synth_negatives(txt, keys, rng, pools, tokv)
    mask = torch.from_numpy(img_rows >= 0)
    flat = img_rows[img_rows >= 0]
    arr = torch.from_numpy(imgs.get(list(D.src[flat]), list(D.rows[flat]))).to(dev, non_blocking=True)
    x = torch.full((B, MAX_IMAGES, 1, X.H, X.W), 1.0, device=dev)
    x[mask.to(dev)] = augment(arr, X.H, X.W, train, keep_ink=True)
    return txt, negs, x, mask.to(dev)


def batch_loss(model, tok, a, field_units, txt, negs, x, mask, train, world=1):
    """학습 · val 공통 손실식 → (loss, 부분 손실 사전). train이고 world > 1이면 1단 InfoNCE는 전 GPU 배치."""
    dev = x.device
    ac = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    B = len(txt)
    allt = txt + negs
    ids, fld, pos, role = tok.batch(allt, dev)
    fl = field_char_lengths(allt).to(dev)
    with torch.autocast(**ac):
        hi = model.image_tokens(x, mask)
        zi = model.slots_image(hi, mask)
        ht = model.text_tokens(ids, fld, pos, role)
        zt_all = model.slots_text(ht, ids)
    zt, zn = zt_all[:B], zt_all[B:]
    sm = (torch.rand(B, model.N, device=dev) >= a.slot_drop) if (a.slot_drop > 0 and train) else None
    if sm is not None:
        sm[sm.sum(1) == 0, 0] = True
    fi = RecordEncoder3.flat(zi.float(), sm); ft = RecordEncoder3.flat(zt.float()); fn = RecordEncoder3.flat(zn.float())
    hid = torch.tensor([hash(t) & 0x7FFFFFFFFFFF for t in txt], device=dev)       # 같은 문자열 = 같은 id (다중 양성)
    if train and world > 1:
        gi = torch.cat(dnn.all_gather(fi), 0); gt = torch.cat(dnn.all_gather(ft), 0)
        gid = torch.cat([torch.zeros_like(hid)] * world, 0); dist.all_gather_into_tensor(gid, hid)
    else:                                                    # val: rank 안 배치만(결정적)
        gi, gt, gid = fi, ft, hid
    # 1단: 대칭 InfoNCE
    pos_m = (hid[:, None] == gid[None, :]).float()
    scale = model.logit_scale.exp().clamp(max=100)
    li = torch.cat([scale * fi @ gt.T, scale * fi @ fn.T], 1)
    pos_i = torch.cat([pos_m, torch.zeros(B, len(fn), device=dev)], 1)
    l_i2t = -(F.log_softmax(li, 1) * (pos_i / pos_i.sum(1, keepdim=True))).sum(1).mean()
    l_t2i = -(F.log_softmax(scale * ft @ gi.T, 1) * (pos_m / pos_m.sum(1, keepdim=True))).sum(1).mean()
    l1 = (l_i2t + l_t2i) / 2
    div = RecordEncoder3.diversity(zi.float()) + RecordEncoder3.diversity(zt.float())
    loss = l1 + a.div_w * div
    l2 = lv = vacc = torch.zeros((), device=dev)
    # 2단: 후보 = 정답 · 합성 음성 3 · 배치 안 1단 최상위
    if a.pli_w > 0:
        with torch.no_grad():
            same = hid[:, None] == hid[None, :]
            hard = (fi @ ft.T).masked_fill(same, -1e9).topk(min(a.pli_hard, B - 1), 1).indices
        b_ = torch.arange(B, device=dev)
        cand = torch.cat([b_[:, None], B + N_NEG * b_[:, None] + torch.arange(N_NEG, device=dev)[None], hard], 1)
        ext = model.ink_extent(x, mask)
        vw = a.verify_w > 0
        with torch.autocast(**ac):
            r_ = model.pli(hi, ext, mask, ht, fld, pos, ids, fl, cand, per_slot=vw)
        sc, pp, ppm = (r_[0].float(), r_[1], r_[2]) if vw else (r_.float(), None, None)
        sp = sc
        fall = torch.cat([ft, fn], 0)
        s1c = (fi[:, None, :] * fall[cand]).sum(-1)
        if a.mix:
            sc = model.final_score(s1c, sc)
        s3 = pg = pgm = None
        if a.ctc_mix:
            r3 = model.cand_ctc(hi.float(), mask, [field_units(t) for t in allt], cand, per_slot=vw)
            s3, pg, pgm = r3 if vw else (r3, None, None)
            sc = sc + model.mix_ctc * s3
        l2 = F.cross_entropy(model.pli_scale.exp().clamp(max=100) * sc, torch.zeros(B, dtype=torch.long, device=dev))
        loss = loss + a.pli_w * l2
        if vw:                                               # 수락 판정 머리: 특징은 떼어 낸(detach) 점수에서
            d_ = lambda t: t.detach()
            fc, c1 = model.verify_feats(d_(sc), d_(sp), d_(s3), d_(pp), ppm, d_(pg), pgm, d_(s1c))
            fo, _ = model.verify_feats(d_(sc).index_fill(1, torch.zeros(1, dtype=torch.long, device=dev), -1e9), d_(sp), d_(s3), d_(pp), ppm, d_(pg), pgm, d_(s1c))
            yv = torch.cat([(c1 == 0).float(), torch.zeros(B, device=dev)])
            lvv = model.verify(torch.cat([fc, fo]))
            lv = F.binary_cross_entropy_with_logits(lvv, yv)
            vacc = ((lvv > 0).float() == yv).float().mean()
            loss = loss + a.verify_w * lv
    lc = torch.zeros((), device=dev)
    if a.ctc_w > 0:                                          # 보조 CTC (이미지 순서 = 비지 않은 필드 순서)
        tgts = [u for t in txt for u in field_units(t)]
        with torch.autocast(**ac):
            lc = model.ctc_loss(hi, mask, tgts)
        loss = loss + a.ctc_w * lc
    return loss, dict(l1=l1, pli=l2, ctc=lc, ver=lv, vacc=vacc, div=div, scale=scale)


def init_from(model, a, tok, dev):
    """--init 체크포인트 로드 + 결합 가중치 초기값. 사전학습에서 오면 mix = [1, 1](텍스트 쪽 · 결합은 아직 학습 안 됨)."""
    from_pre = True
    if a.init:
        ck0 = torch.load(ROOT / a.init, map_location=dev, weights_only=False)
        assert ck0["vocab"] == tok.vocab, "어휘가 다름"
        from_pre = "eval_every" in ck0.get("args", {})
        miss = model.load_state_dict(ck0["model"], strict=False)
        log("init from", a.init, "사전학습" if from_pre else "FT", "missing", miss.missing_keys, "unexpected", miss.unexpected_keys)
    if a.mix and from_pre:
        model.mix.data = torch.tensor([1.0, 1.0], device=dev)
    elif not a.mix:
        model.mix.requires_grad_(False)
    if a.ctc_mix:
        if float(model.mix_ctc) == 0:
            model.mix_ctc.data = torch.tensor(0.1, device=dev)
    else:
        model.mix_ctc.requires_grad_(False)


def make_optimizer(model, a, cpu):
    """수락 판정 머리는 lr × 10 · weight decay 0."""
    vh = [p_ for n_, p_ in model.named_parameters() if n_.startswith("vhead.")]
    rest = [p_ for n_, p_ in model.named_parameters() if not n_.startswith("vhead.")]
    return torch.optim.AdamW([dict(params=rest), dict(params=vh, lr=a.lr * 10, weight_decay=0.0)], lr=a.lr, weight_decay=a.wd, fused=not cpu)


def val_records(va, a):
    """val 고정 표본: 모든 rank가 같은 레코드 목록 → bs 단위 묶음."""
    vrecs = va.sample(np.random.default_rng(123), a.val_recs, a.shuffle_fields, a.field_drop)
    return [vrecs[i:i + a.bs] for i in range(0, len(vrecs), a.bs)]


def main(argv=None):
    a = parse_args(argv)
    dev, rank, world = setup(a.cpu, 29923)
    if not a.cpu:
        enable_tf32()
    out = ROOT / "runs/x" / a.name
    if is_main():
        (out / "epochs").mkdir(parents=True, exist_ok=True); json.dump(vars(a), open(out / "args.json", "w"), indent=1)
    tok = X.load_tok()
    tr, va, imgs = X.XData(a.data, "train"), X.XData(a.data, "val"), X.XImages(a.data)
    pools = X.char_pools(tr.text); tokv = X.token_vocab(tr)
    log(f"data {a.data}: train 레코드 {len(tr.pids)} 이미지 {len(tr.text)} · val 레코드 {len(va.pids)} · 어휘 {len(tok.vocab)} · world {world}")
    torch.manual_seed(a.seed)
    model = make_model(a, len(tok.vocab), tok.maxu).to(dev)
    for p_ in model.parameters():
        dist.broadcast(p_.data, 0)
    init_from(model, a, tok, dev)
    field_units = FieldUnits(tok)
    log(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    model = model.to(memory_format=torch.channels_last) if not a.cpu else model
    opt = make_optimizer(model, a, a.cpu)
    spe = a.spe; total = a.steps or spe * a.epochs; warm = int(spe * a.warmup) if not a.steps else 2
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    start, best, bad, hist = 0, float("inf"), 0, []
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best, bad, hist = ck["step"], ck["best"], ck["bad"], ck["hist"]; log("resume", start)
    vb = val_records(va, a)

    def val_loss():
        model.eval(); acc = torch.zeros(7, device=dev)
        with torch.no_grad():
            for bi in range(rank, len(vb), world):
                txt, negs, x, mask = build_batch(va, vb[bi], np.random.default_rng(1000 + bi), False, imgs, pools, tokv, dev)
                l, p = batch_loss(model, tok, a, field_units, txt, negs, x, mask, False, world)
                acc += torch.tensor([float(l), float(p["l1"]), float(p["pli"]), float(p["ctc"]), float(p["ver"]), float(p["vacc"]), 1.0], device=dev)
        dist.all_reduce(acc); model.train()
        k = acc[-1].clamp(min=1)
        return dict(zip(["loss", "l1", "pli", "ctc", "ver", "vacc"], (acc[:6] / k).tolist()))

    rng = np.random.default_rng(a.seed * 1000 + rank + start)
    t0 = time.time()
    for step in range(start, total):
        recs = tr.sample(rng, a.bs, a.shuffle_fields, a.field_drop)
        txt, negs, x, mask = build_batch(tr, recs, rng, True, imgs, pools, tokv, dev)
        if step == start and is_main():                        # 입력 1회 점검: 실제 배치 그림 · 이미지 수 = 비지 않은 필드 수 · 값 범위
            from formrec.common import input_check
            r = input_check.kie_batch(x.float().cpu().numpy(), mask.cpu().numpy(), txt, f"x_{a.name}_train_s{step}")
            if not r["pass"]:
                raise SystemExit("입력 점검 실패 — logs/checks/ 확인")
        loss, p = batch_loss(model, tok, a, field_units, txt, negs, x, mask, True, world)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if world > 1:
            average_gradients(list(model.parameters()), world)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0 or step == total - 1 or step < 3:
            log(f"step {step}/{total} loss {loss.item():.4f} l1 {p['l1'].item():.4f} pli {p['pli'].item():.4f} ctc {p['ctc'].item():.3f} "
                f"ver {p['ver'].item():.4f}/{p['vacc'].item():.3f} div {p['div'].item():.3f} scale {p['scale'].item():.1f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
        if (step + 1) % spe == 0 or step == total - 1:
            ep = (step + 1 + spe - 1) // spe
            v = val_loss()
            improved = v["loss"] < best
            best, bad = (v["loss"], 0) if improved else (best, bad + 1)
            hist.append(dict(epoch=ep, step=step + 1, train_loss=float(loss.item()), **{f"val_{k}": x_ for k, x_ in v.items()}, time=time.time() - t0))
            log(f"epoch {ep}: val loss {v['loss']:.4f} (l1 {v['l1']:.4f} · pli {v['pli']:.4f} · ctc {v['ctc']:.3f} · ver {v['ver']:.4f}/{v['vacc']:.3f})"
                + (" ← 최소" if improved else f" (개선 없음 {bad}/{a.patience})"))
            if is_main():
                sd = model.state_dict()
                torch.save(dict(model=sd, args=vars(a), epoch=ep, vocab=tok.vocab), out / "epochs" / f"ep{ep}.pt")
                if improved:
                    shutil.copy(out / "epochs" / f"ep{ep}.pt", out / "final.pt")
                    json.dump(dict(best_epoch=ep, criterion="val loss (학습과 같은 손실식, 고정 표본)", val=hist[-1]), open(out / "best.json", "w"), indent=1)
                with open(out / "log.jsonl", "a") as f:
                    f.write(json.dumps(hist[-1]) + "\n")
                torch.save(dict(model=sd, opt=opt.state_dict(), sched=sched.state_dict(), step=step + 1, best=best, bad=bad, hist=hist), out / "last.pt")
            if bad >= a.patience:
                log(f"조기 종료: {a.patience}에폭 연속 개선 없음")
                break
    if is_main():
        (out / "last.pt").unlink(missing_ok=True)
        log("DONE", a.name, "best epoch", json.load(open(out / "best.json"))["best_epoch"])
    finish()


if __name__ == "__main__":
    main()
