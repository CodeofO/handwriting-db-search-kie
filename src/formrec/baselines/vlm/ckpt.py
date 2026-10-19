"""VLM 파인튜닝 체크포인트 → vLLM 폴더 변환(export), val loss 최소 체크포인트 선정(select_loss).

체크포인트 = runs/{ds}/{run}/checkpoint-{step} (step 오름차순 = 1, 2, 3 …번째 저장). run 기본 = {kind}_ft.
export: 가중치(safetensors)만 체크포인트에서, 가중치가 아닌 파일(설정·코드·토크나이저·템플릿)은 원본 모델에서 가져온다.
  pvl : PaddleFormers 전체 FT 결과에서 빠지는 비전 텐서(packing_position_embedding, head.*)를 원본에서 보충
        (PaddleFormers issue #3680: vLLM이 "weights not initialized"로 실패. 추론에 쓰이지 않는 텐서라 출력 불변)
  ovis: transformers 5가 preprocessor_config.json·vocab.json·merges.txt를 빼고 저장하므로 원본 파일을 쓴다. dtype은 원본(bf16)으로
select_loss: 체크포인트별 val loss(trainer_state.json, 없으면 학습 로그 workerlog의 eval_loss 순서) 최소 → export/best (심볼릭 링크).
  체크포인트 · 다른 내보내기는 지우지 않는다(지표 기준 선정은 부록).
사용: python -m formrec.baselines.vlm.ckpt export pvl 2 [run 이름] | select_loss pvl [run 이름]
"""
import json
import re
import shutil
import sys

from formrec.common import workspace as cfg

BASE = {"pvl": cfg.MODELS / "PaddleOCR-VL-1.6", "ovis": cfg.MODELS / "OvisOCR2"}
RUN = None   # 기본 runs/{ds}/{kind}_ft, 스모크 · 다른 실행은 인자로 이름


def run_dir(kind):
    return cfg.RUNS / (RUN or f"{kind}_ft")


def checkpoints(kind):
    return sorted(run_dir(kind).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))


def export(kind, ep):
    from safetensors import safe_open
    from safetensors.torch import save_file
    ck = checkpoints(kind)
    assert len(ck) >= ep, f"{kind}: 체크포인트 {len(ck)}개 < 에폭 {ep}"
    src, base = ck[ep - 1], BASE[kind]
    out = run_dir(kind) / f"export/ep{ep}"
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    for f in base.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and not f.name.endswith(".index.json") and f.suffix not in (".png", ".md"):
            shutil.copy2(f, tmp / f.name)
    with safe_open(base / "model.safetensors", "pt") as g:
        base_keys = set(g.keys()); dtype = g.get_tensor(next(iter(base_keys))).dtype
    ix = src / "model.safetensors.index.json"   # PaddleFormers unified_checkpoint는 master_weights·optimizer도 safetensors로 둔다
    sts = sorted({src / f for f in json.load(open(ix))["weight_map"].values()}) if ix.exists() else sorted(src.glob("model*.safetensors"))
    assert sts, f"{src}: safetensors 없음 ({[p.name for p in src.iterdir()][:10]})"
    tensors = {}
    for sf in sts:
        with safe_open(sf, "pt") as g:
            for k in g.keys():
                tensors[k] = g.get_tensor(k)
    if "lm_head.weight" in tensors and "lm_head.weight" not in base_keys:   # tie_word_embeddings: 원본처럼 저장하지 않음
        del tensors["lm_head.weight"]
    have = set(tensors)
    unknown = sorted(have - base_keys)
    missing = sorted(base_keys - have)
    assert not unknown, f"원본에 없는 키 {len(unknown)}개: {unknown[:5]}"
    if kind == "pvl":
        assert all("packing_position_embedding" in k or ".head." in k for k in missing), missing[:10]
    else:
        assert not missing, missing[:10]
    with safe_open(base / "model.safetensors", "pt") as g:
        for k in missing:
            tensors[k] = g.get_tensor(k)
    tensors = {k: (v.to(dtype) if v.is_floating_point() else v).contiguous() for k, v in tensors.items()}
    save_file(tensors, tmp / "model.safetensors", metadata={"format": "pt"})
    shutil.rmtree(out, ignore_errors=True); tmp.rename(out)
    print(f"DONE export {kind} ep{ep} <- {src.name}: {len(tensors)} tensors (보충 {len(missing)}) dtype {dtype}")


def losses_from_log(text, n_ck):
    """학습 로그 본문 → {체크포인트 번호: eval_loss} (나온 순서대로, 체크포인트 수까지)."""
    ls = [float(x) for x in re.findall(r"eval_loss: ([0-9.eE+-]+)", text)]
    return {i + 1: l for i, l in enumerate(ls[:n_ck])} if ls else {}


def eval_losses(kind):
    """체크포인트별 val loss: 마지막 체크포인트의 trainer_state.json(log_history의 eval_loss · step). 없으면 학습 로그에서."""
    ck = checkpoints(kind)
    st = ck[-1] / "trainer_state.json"
    hist = json.load(open(st))["log_history"] if st.exists() else []
    ev = sorted((int(h["step"]), float(h["eval_loss"])) for h in hist if "eval_loss" in h)
    steps = [int(c.name.split("-")[1]) for c in ck]
    if not ev:                                           # PaddleFormers는 trainer_state에 eval을 남기지 않음 → 이 실행의 학습 로그(workerlog.0)
        rd = str(run_dir(kind))
        for lg in sorted(cfg.LOGS.glob("pdist/*/workerlog.0"), key=lambda p: -p.stat().st_mtime):
            t = lg.read_text(errors="ignore")
            if re.search(r"output_dir\s*:\s*" + re.escape(rd) + r"\b", t) or re.search(r"output_dir\s*:\s*" + re.escape(rd.replace(str(cfg.ROOT) + "/", "")) + r"\b", t):
                return losses_from_log(t, len(ck))
        return {}
    out = {}
    for i, s_ in enumerate(steps):                       # 체크포인트 i(1부터) ↔ 그 스텝의 val loss (에폭 저장 · 스텝 저장 모두)
        m = [l for st_, l in ev if st_ == s_]
        if m:
            out[i + 1] = m[-1]
    assert out, f"{kind}: eval_loss 기록 없음 ({st})"
    return out


def select_loss(kind):
    """val loss 최소 체크포인트(동률이면 앞) → export/best."""
    L = eval_losses(kind)
    best = min(L, key=lambda e: (L[e], e))
    rd = run_dir(kind)
    if not (rd / f"export/ep{best}").exists():
        export(kind, best)
    b = rd / "export/best"
    if b.is_symlink() or b.exists():
        b.unlink() if b.is_symlink() else shutil.rmtree(b)
    b.symlink_to(f"ep{best}")
    json.dump({"best_epoch": best, "criterion": "val loss (프레임워크 평가)", "val_loss": L}, open(rd / "best.json", "w"), indent=1)
    print(f"DONE select_loss {kind}: best ep{best} {L}")


def main(argv=None):
    global RUN
    argv = sys.argv[1:] if argv is None else argv
    if argv[0] == "select_loss":
        RUN = argv[2] if len(argv) > 2 else None
        select_loss(argv[1])
    elif argv[0] == "export":
        RUN = argv[3] if len(argv) > 3 else None
        export(argv[1], int(argv[2]))
    else:
        raise SystemExit(f"알 수 없는 명령 {argv[0]} (export | select_loss)")


if __name__ == "__main__":
    main()
