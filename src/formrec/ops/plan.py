"""실험 계획 생성기 — 실제(plan_full)와 스모크(plan_smoke)를 같은 함수로 만든다(같은 단계 · 같은 명령 · 같은 순서).

학습: CTC 사전학습 1 + 파인튜닝 12(제안 · PP-OCRv5 · PaddleOCR-VL-1.6 · OvisOCR2 × AI Hub · HANA · IAM) + 절제(사전학습 없음) 1
      + 한 줄 여러 칼럼(xc) 4 + 손글씨 KIE(xk · xkp · 데이터량 곡선 · 두 번째 VLM) + 일반 KIE 인쇄 글자 OCR.
데이터셋마다 학습 → 평가 묶음(선정 → 내보내기 → tune · test → 채점 → 요약).
작업 사전: name · cmd(bash -c, 작업 공간에서 실행) · gpus(0 / 1 / 4) · deps(산출물 의존, 실패하면 건너뜀) · after(순서만) · io(대용량 IO 잠금) ·
  stall(로그 정지 알림 분) · disk_gb(예상 쓰기 GB — 디스크 보호가 시작 여부를 정함, 스모크는 1/10).
명령은 `python -m formrec.<패키지>.<모듈>` 형태이고 Python 환경은 $FORMREC_PY_{BASE,TRAIN,PADDLE,VLLM,PF}, 셸 스크립트는 $FORMREC_REPO/scripts
  (scripts/run_job.sh가 기본값을 채운다).
스모크: 최소 데이터 · 몇 스텝으로 학습 → 체크포인트 저장 → 선정 → 평가 → 다음 단계. 이름 접두 S_ · 산출물 접미 _smoke, 끝나면 산출물을 지운다.
"""
import re

# 다른 영역 모듈(이름 약속) — 바뀌면 여기만 고친다
MODULES = dict(
    data="formrec.data_prep.build",             # iam_blob · hana_blob · cache · xc_blob
    ft_data="formrec.baselines.vlm.ft_data",    # VLM 학습 데이터
    pf_train="formrec.baselines.vlm.pf_train",  # PaddleFormers 학습 실행
    vlm_ckpt="formrec.baselines.vlm.ckpt",      # 체크포인트 선정 · 내보내기
    vlm_infer="formrec.baselines.vlm.infer",    # VLM OCR 생성
    pp_select="formrec.baselines.ppocr.bn_select",
    pp_infer="formrec.baselines.ppocr.infer",
    base_eval="formrec.baselines.evaluate",
    pretrain="formrec.retrieval.pretrain",
    train="formrec.retrieval.train",
    evaluate="formrec.retrieval.evaluate",
    summary="formrec.retrieval.summary",
    kie_vlm="formrec.kie.vlm",
    kie_eval="formrec.kie.evaluate",
    kie_oov="formrec.kie.oov",
    kie_order="formrec.kie.order",
    kie_ocr="formrec.kie_general.ocr",
)
PY = {k: f'"$FORMREC_PY_{k.upper()}"' for k in ("base", "train", "paddle", "vllm", "pf")}
SH = '"$FORMREC_REPO"/scripts'


def m(key):
    return "-m " + MODULES[key]


PB, PT = PY["base"], PY["train"]
TORCHRUN = f"env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {PT} -m torch.distributed.run --nproc_per_node 4"
XT = f"{TORCHRUN} --master_port 29751 {m('train')}"
# PP-OCRv5 입력: 폭 비례 풀링, 글자가 촘촘한 HANA · IAM은 가로 2배로 늘려 글자당 출력 칸을 확보(폭 1280이면 글자당 1.35칸 → 부족)
PPENV = {"xa": "KIE_PPOCR_POOL=prop", "xc": "KIE_PPOCR_POOL=prop", "xh": "KIE_PPOCR_POOL=prop KIE_PPOCR_XSTRETCH=2 PPOCR_CFG=en_v5_ft.yml",
         "xi": "KIE_PPOCR_POOL=prop KIE_PPOCR_XSTRETCH=2 PPOCR_CFG=en_v5_ft.yml"}
PPCFG = {"xa": "korean_v5_ft.yml", "xc": "korean_v5_ft.yml", "xh": "en_v5_ft.yml", "xi": "en_v5_ft.yml"}
PPNAME = {"xa": "korean_PP-OCRv5_mobile_rec", "xc": "korean_PP-OCRv5_mobile_rec", "xh": "en_PP-OCRv5_mobile_rec", "xi": "en_PP-OCRv5_mobile_rec"}
VLM = (("pvl", "paddlevl"), ("ovis", "ovis"))
# 예상 쓰기 GB(체크포인트 · 전처리 캐시). 없으면 선정 · 내보내기 = 2, 그 밖 = 0
DISK = {"pretrain": 3, "xa_prop": 6, "xh_prop": 6, "xi_prop": 6, "xa_scratch": 6, "xa_ppocr": 2, "xh_ppocr": 2, "xi_ppocr": 2,
        "xa_pvl": 6, "xh_pvl": 6, "xi_pvl": 19, "xa_ovis": 8, "xh_ovis": 25, "xi_ovis": 19,
        "xc_prop": 6, "xc_ppocr": 2, "xc_pvl": 6, "xc_ovis": 8, "xk_pvl": 6, "xk_pvl_smoke": 2, "xk2_pvl": 9, "xk2_pvl_smoke": 2,
        "xkp_pvl": 6, "xkp10_pvl": 6, "xkp30_pvl": 6, "xkp_ovis": 8}


def KVX(ds):
    return f"env -u OMP_NUM_THREADS HW_DATASET={ds} VLLM_USE_FLASHINFER_SAMPLER=0 {PY['vllm']}"


def build(smoke=False):
    JOBS = []
    P = "S_" if smoke else "X_"                                      # 작업 이름 접두
    sx = "_smoke" if smoke else ""                                   # 실행 · 결과 이름 접미

    def J(name, cmd, gpus=0, deps=(), after=(), io=False, stall=20):
        g = DISK.get(name, 2 if name.endswith("_sel") or re.search(r"_exp\d+$", name) else 0)
        JOBS.append(dict(name=P + name, cmd=cmd, gpus=gpus, deps=[P + d for d in deps], after=[P + d for d in after], io=io, stall=stall,
                         disk_gb=g / 10 if smoke else g))

    NCK = {"xa": 2, "xh": 2, "xi": 2, "xc": 2} if smoke else {"xa": 3, "xh": 3, "xi": 10, "xc": 3}   # VLM 체크포인트 수(에폭)
    # ── 0. 데이터 (실제: 이미 끝났으면 .done으로 건너뜀)
    if not smoke:
        J("iam_blob", f"{PB} {m('data')} iam_blob", 0, io=True, stall=30)
        J("hana_blob", f"{PB} {m('data')} hana_blob", 0, ["iam_blob"], io=True, stall=60)
        J("cache_xi", f"{PB} {m('data')} cache --ds xi", 0, ["iam_blob"], io=True, stall=30)
        J("cache_xh", f"{PB} {m('data')} cache --ds xh", 0, ["hana_blob", "cache_xi"], io=True, stall=60)
        J("xa_vlmdata", f"env HW_DATASET=xa {PB} {m('ft_data')} vlm", 0, ["hana_blob"], io=True, stall=30)
        J("xh_vlmdata", f"env HW_DATASET=xh {PB} {m('ft_data')} vlm --save_splits 3", 0, ["cache_xh"], io=True, stall=40)
        J("xi_vlmdata", f"env HW_DATASET=xi {PB} {m('ft_data')} vlm --vlm_epochs 10", 0, ["xh_vlmdata"], io=True, stall=40)
    else:                                                             # 스모크 데이터: VLM 학습 512 · val 64 (실제 파일의 앞부분)
        for ds in ("xa", "xh", "xi"):
            J(f"{ds}_vlmdata", " && ".join(f"head -n {n} ft/{ds}/{k}/{s}.jsonl > ft/{ds}/{k}/{s}_smoke.jsonl"
                                           for k in ("pvl", "ovis") for s, n in (("train", 512), ("val", 64))), 0, io=True)

    # ── 학습 명령
    def ppocr_train(ds):
        e = f"env HW_DATASET={ds} {PPENV[ds]}"
        return f"{e} SMOKE=1 SMOKE_EPOCHS=2 bash {SH}/ppocr_ft.sh" if smoke else f"{e} bash {SH}/ppocr_ft.sh"

    def pf_launch(ds, y, extra=""):
        return f"env HW_DATASET={ds} {PY['pf']} {m('pf_train')} launch {y}{extra}"

    def pvl_train(ds):
        y = f"ft/{ds}/pvl/pvl16_full.yaml"
        if not smoke:
            return pf_launch(ds, y)
        o = f"runs/{ds}/pvl_ft_smoke"
        return (f"rm -rf {o} && " + pf_launch(ds, y, f" max_steps=20 save_strategy=steps save_steps=10 "
                f"evaluation_strategy=steps eval_steps=10 train_dataset_path=ft/{ds}/pvl/train_smoke.jsonl eval_dataset_path=ft/{ds}/pvl/val_smoke.jsonl "
                f"output_dir={o} logging_dir={o}/vdl logging_steps=1"))

    def ovis_train(ds):
        if not smoke:
            extra = "--eval_strategy epoch --num_train_epochs $(cat ft/xa/ovis/epochs.txt)" if ds == "xa" else f"$(cat ft/{ds}/ovis/swift_args.txt)"
            return f"env HW_DATASET={ds} bash {SH}/ovis_train.sh --val_dataset ft/{ds}/ovis/val.jsonl --per_device_eval_batch_size 32 {extra}"
        return (f"rm -rf runs/{ds}/ovis_ft_smoke && env HW_DATASET={ds} OVIS_TRAIN=ft/{ds}/ovis/train_smoke.jsonl bash {SH}/ovis_train.sh "
                f"--val_dataset ft/{ds}/ovis/val_smoke.jsonl --per_device_eval_batch_size 8 --max_steps 20 --save_strategy steps --save_steps 10 "
                f"--eval_strategy steps --eval_steps 10")

    def prop_train(ds, name, init=True):
        i = (f"--init runs/x/pretrain_ctc{'' if ds == 'xc' else sx}/final.pt " if init else "")   # xc는 스모크도 실제 사전학습 가중치에서
        if smoke:
            return f"rm -rf runs/x/{name} && {XT} --data {ds} --name {name} {i}--steps 40 --spe 20 --val_recs 256"
        spe = "--spe 46 --epochs 348 " if ds == "xi" else ""
        return f"{XT} --data {ds} --name {name} {i}{spe}--resume"

    def ckpt(ds, cmd):
        return f"env HW_DATASET={ds} {PT} {m('vlm_ckpt')} {cmd}"

    # ── 평가 묶음
    def eval_block(ds, prop_runs, after, baselines=True):
        fin = []
        sm = " --smoke 40 --cap 200" if smoke else ""
        for r in prop_runs:
            J(f"{r}_sel", f"{PT} {m('evaluate')} --ds {ds} --name {r}{sx} --mode select{sm}", 1, [r], [after], stall=40)
            J(f"{r}_eval", f"{PT} {m('evaluate')} --ds {ds} --name {r}{sx} --mode eval{sm}", 1, [f"{r}_sel"], stall=40)
            J(f"{r}_ep", f"{PT} {m('evaluate')} --ds {ds} --name {r}{sx} --mode epochs{sm}", 1, [f"{r}_eval"], [] if smoke else ["xc_ovis"], stall=40)   # 부록용 — 학습을 다 끝낸 뒤
            fin.append(f"{r}_eval")
        if baselines:
            E_ = f"env HW_DATASET={ds} {PPENV[ds]}"
            run = f"ppocrv5_ft{sx}"
            lim = " --val_limit 512 --calib 512 --check_warn_only" if smoke else ""   # 2에폭 스모크 모델은 BN 통계가 불안정해 모드 차이 점검이 무의미
            J(f"{ds}_ppocr_sel", f"{E_} {PY['paddle']} {m('pp_select')} --run runs/{ds}/{run} --cfg ft/{ds}/ppocr/{PPCFG[ds]} --select loss --last 0{lim}",
              1, [f"{ds}_ppocr"], [after], stall=60)
            J(f"{ds}_ppocr_post", f"{E_} bash {SH}/ppocr_ft_post.sh {run}", 0, [f"{ds}_ppocr_sel"])
            gl = " --limit 300" if smoke else ""
            for s in ("tune", "test"):
                J(f"{ds}_ppocr_g1_{s}", f"{E_} {PY['paddle']} {m('pp_infer')} --subset {s} --tag ppocrv5ft --out_tag ppocrv5ft{sx} --model_name {PPNAME[ds]} "
                  f"--model_dir runs/{ds}/{run}/infer{gl}", 1, [f"{ds}_ppocr_post"], stall=40)
            part = " --partial" if smoke else ""
            J(f"{ds}_ppocr_base", f"env HW_DATASET={ds} {PB} {m('base_eval')} --tag ppocrv5ft{sx}{part}", 0, [f"{ds}_ppocr_g1_{s}" for s in ("tune", "test")])
            fin.append(f"{ds}_ppocr_base")
            vl = " --limit 100" if smoke else ""
            for k, mdl in VLM:
                rd = f"runs/{ds}/{k}_ft{sx}"
                J(f"{ds}_{k}_sel", ckpt(ds, f"select_loss {k} {k}_ft{sx}"), 0, [f"{ds}_{k}"], [after], io=True, stall=40)
                for s in ("tune", "test"):
                    J(f"{ds}_{k}_g1_{s}", f"{KVX(ds)} {m('vlm_infer')} --model {mdl} --subset {s} --model_path {rd}/export/best --out_name {k}ft{sx}_{s}{vl}", 1,
                      [f"{ds}_{k}_sel"], stall=40)
                J(f"{ds}_{k}_base", f"env HW_DATASET={ds} {PB} {m('base_eval')} --tag {k}ft{sx}{part}", 0, [f"{ds}_{k}_g1_{s}" for s in ("tune", "test")])
                fin.append(f"{ds}_{k}_base")
                for i in range(1, NCK[ds] + 1):                         # 부록 · 지표 기준 선정: 체크포인트별 val 완전 일치 → 채점 뒤 그 내보내기는 지움(최선만 남김)
                    J(f"{ds}_{k}_exp{i}", f"[ -d {rd}/export/ep{i} ] || [ $(ls -d {rd}/checkpoint-* | wc -l) -lt {i} ] || "
                      + ckpt(ds, f"export {k} {i} {k}_ft{sx}"), 0, [f"{ds}_{k}_base"], io=True, stall=40)
                    J(f"{ds}_{k}_val{i}", f"[ ! -d {rd}/export/ep{i} ] || ( {KVX(ds)} {m('vlm_infer')} --model {mdl} --subset val --model_path {rd}/export/ep{i} "
                      f"--out_name {k}ft{sx}_ep{i}_val{vl} && {{ [ \"$(readlink {rd}/export/best)\" = ep{i} ] || rm -rf {rd}/export/ep{i}; }} )", 1, [f"{ds}_{k}_exp{i}"], stall=40)
        tag = "_".join(prop_runs) if not baselines else ds
        J(f"{tag}_summary", f"{PB} {m('summary')} --data {ds}", 0, after=fin)
        return f"{tag}_summary"

    # ── 1~3. AI Hub: PP-OCRv5 → 사전학습 → 제안 FT → PaddleOCR-VL → OvisOCR2 → 평가 묶음
    J("xa_ppocr", ppocr_train("xa"), 4, [], [], stall=30)
    pre = (f"rm -rf runs/x/pretrain_ctc_smoke && " if smoke else "") + f"{TORCHRUN} --master_port 29741 {m('pretrain')} " + \
        ("--name pretrain_ctc_smoke --steps 60 --eval_every 30 --val_cap 200 --warmup 10" if smoke else "--name pretrain_ctc --steps 30000 --eval_every 2000 --resume")
    J("pretrain", pre, 4, [] if smoke else ["cache_xh", "cache_xi"], ["xa_ppocr"], stall=30)
    J("xa_prop", prop_train("xa", f"xa_prop{sx}"), 4, ["pretrain"], stall=30)
    J("xa_pvl", pvl_train("xa"), 4, ["xa_vlmdata"], ["xa_prop"], stall=40)
    J("xa_ovis", ovis_train("xa"), 4, ["xa_vlmdata"], ["xa_pvl"], stall=40)
    J("xa_ovis_clean", f"rm -f ft/xa/ovis/train{sx}.jsonl" if not smoke else "true", 0, after=["xa_ovis"])
    eval_block("xa", ["xa_prop"], "xa_ovis")
    # ── 4. HANA
    J("xh_prop", prop_train("xh", f"xh_prop{sx}"), 4, ["pretrain"], ["xa_ovis"], stall=30)
    J("xh_ppocr", ppocr_train("xh"), 4, [], ["xh_prop"], stall=40)
    J("xh_pvl", pvl_train("xh"), 4, ["xh_vlmdata"], ["xh_ppocr"], stall=40)
    J("xh_ovis", ovis_train("xh"), 4, ["xh_vlmdata"], ["xh_pvl"], stall=40)
    J("xh_ovis_clean", f"rm -f ft/xh/ovis/train{sx}.jsonl" if not smoke else "true", 0, after=["xh_ovis"])
    eval_block("xh", ["xh_prop"], "xh_ovis")
    # ── 5. IAM
    J("xi_prop", prop_train("xi", f"xi_prop{sx}"), 4, ["pretrain"], ["xh_ovis"], stall=30)
    J("xi_ppocr", ppocr_train("xi"), 4, [], ["xi_prop"], stall=40)
    J("xi_pvl", pvl_train("xi"), 4, ["xi_vlmdata"], ["xi_ppocr"], stall=40)
    J("xi_ovis", ovis_train("xi"), 4, ["xi_vlmdata"], ["xi_pvl"], stall=40)
    J("xi_ovis_clean", f"rm -f ft/xi/ovis/train{sx}.jsonl" if not smoke else "true", 0, after=["xi_ovis"])
    eval_block("xi", ["xi_prop"], "xi_ovis")
    # ── 6. 절제: 사전학습 없이 AI Hub 처음부터
    J("xa_scratch", prop_train("xa", f"xa_scratch{sx}", init=False), 4, [], ["xi_ovis"], stall=30)
    eval_block("xa", ["xa_scratch"], "xa_scratch", baselines=False)
    # ── 7. 한 줄 여러 칼럼(xc): AI Hub F2 · F3 칼럼 이미지를 가로로 이어 붙인 이미지 1장 — 네 모델 학습 · 평가
    if not smoke:
        J("xc_blob", f"[ -f meta/x/xc_items.parquet ] && [ -f crops/xc/all.blob ] || {PB} {m('data')} xc_blob", 0, io=True, stall=30)
        J("cache_xc", f"[ -f cache/xc/all_64x1280.npy ] || {PB} {m('data')} cache --ds xc", 0, ["xc_blob"], io=True, stall=30)
        J("xc_vlmdata", f"[ -f ft/xc/ovis/val.jsonl ] || env HW_DATASET=xc {PB} {m('ft_data')} vlm --vlm_epochs 3", 0, ["xc_blob"], io=True, stall=40)
    else:
        J("xc_vlmdata", " && ".join(f"head -n {n} ft/xc/{k}/{s}.jsonl > ft/xc/{k}/{s}_smoke.jsonl" for k in ("pvl", "ovis") for s, n in (("train", 512), ("val", 64))), 0, io=True)
    xd = ["cache_xc"] if not smoke else []
    J("xc_prop", prop_train("xc", f"xc_prop{sx}"), 4, xd, [] if smoke else ["xa_scratch"], stall=30)
    if not smoke:
        kie_jobs_person(J)
    J("xc_ppocr", ppocr_train("xc"), 4, xd, ["xc_prop"] if smoke else ["xk_pvl"], stall=40)
    J("xc_pvl", pvl_train("xc"), 4, ["xc_vlmdata"], ["xc_ppocr"] if smoke else ["xc_ppocr", "xk2_pvl"], stall=40)
    J("xc_ovis", ovis_train("xc"), 4, ["xc_vlmdata"], ["xc_pvl"], stall=40)
    J("xc_ovis_clean", f"rm -f ft/xc/ovis/train{sx}.jsonl" if not smoke else "true", 0, after=["xc_ovis"])
    eval_block("xc", ["xc_prop"], "xc_ovis")
    if not smoke:
        kie_jobs_page(J)
        for k in range(4):                                             # 일반 KIE(xkg): 서식 페이지 전체의 인쇄 글자 OCR(4 조각, GPU 1장씩)
            J(f"xkg_ocr{k}", f"{PY['paddle']} {m('kie_ocr')} --shard {k} --n 4", 1, [], [], stall=60)
    if smoke:                                                          # 스모크 산출물 정리
        J("cleanup", f"bash {SH}/smoke_clean.sh", 0, after=[j["name"][2:] for j in JOBS])
    return JOBS


# ── 손글씨 KIE (실제 계획에만)
KV = f"env HW_DATASET=xk {PY['base']} {m('kie_vlm')}"
KVI = f"{KVX('xk')} {m('kie_vlm')} infer"


def _pf_smoke(ft_yaml, run, steps, val):
    """PaddleOCR-VL 짧은 학습(스모크) 인자."""
    return (f"env HW_DATASET=xk {PY['pf']} {m('pf_train')} launch {ft_yaml} max_steps={steps} save_strategy=steps save_steps={steps} "
            f"evaluation_strategy=steps eval_steps={steps} eval_dataset_path={val} output_dir={run} logging_dir={run}/vdl logging_steps=1")


def _ckpt_xk(cmd):
    return f"env HW_DATASET=xk {PT} {m('vlm_ckpt')} {cmd}"


def kie_jobs_person(J):
    """사람 영역 KIE(xk): 스모크 1단계 → 학습 → 선정 → tune · test 생성 → 채점, 그리고 입력 해상도를 올린 v2(폭 1860, all3)."""
    kv = "ft/xk/pvl"
    J("xk_pvl_smoke", f"head -n 64 {kv}/val.jsonl > {kv}/val_smoke.jsonl && rm -rf runs/xk/pvl_ft_smoke && "
      + _pf_smoke(f"{kv}/pvl16_full.yaml", "runs/xk/pvl_ft_smoke", 10, f"{kv}/val_smoke.jsonl")
      + f" && {_ckpt_xk('export pvl 1 pvl_ft_smoke')} && {KVI} --subset test --model_path runs/xk/pvl_ft_smoke/export/ep1 --out smoke --limit 32 && rm -rf runs/xk/pvl_ft_smoke",
      4, [], ["xc_prop"], stall=40)
    J("xk_pvl", f"env HW_DATASET=xk {PY['pf']} {m('pf_train')} launch {kv}/pvl16_full.yaml", 4, ["xk_pvl_smoke"], stall=40)
    J("xk_pvl_sel", _ckpt_xk("select_loss pvl pvl_ft"), 0, ["xk_pvl"], io=True, stall=40)
    for s_ in ("tune", "test"):
        J(f"xk_pvl_infer_{s_}", f"{KVI} --subset {s_} --model_path runs/xk/pvl_ft/export/best --out pvlkie", 1, ["xk_pvl_sel"], stall=40)
    rd = "runs/xk/pvl_ft"
    for i in range(1, 4):                                              # 부록 · 지표 기준 선정: 체크포인트별 val 질의 완전 일치 → 디스크 보호가 두 선정본만 남김
        J(f"xk_pvl_exp{i}", f"[ -d {rd}/export/ep{i} ] || {_ckpt_xk(f'export pvl {i} pvl_ft')}", 0, ["xk_pvl_sel"], io=True, stall=40)
        J(f"xk_pvl_val{i}", f"{KVI} --subset val --model_path {rd}/export/ep{i} --out pvlkie_ep{i} --val_em_out pvlft_ep{i}_val "
          f"&& {{ [ \"$(readlink {rd}/export/best)\" = ep{i} ] || rm -rf {rd}/export/ep{i}; }}", 1, [f"xk_pvl_exp{i}"], stall=40)
    J("xk_eval", f"{PB} {m('kie_eval')} --subset test --methods ocr:ppocrv5ft,ocr:pvlft,ocr:ovisft,vlm:pvlkie --db", 0, ["xk_pvl_infer_test"])
    # v2: 입력 해상도 올림(v1은 가로 절반 축소 → 주소 한 글자 오독이 대부분). 원본 해상도는 val 평가 메모리 초과 → 폭 1860(all3) · val 800 · 2에폭
    k2, r2 = "ft/xk/pvl3", "runs/xk/pvl_ft3"
    J("xk2_data", f"until [ -f crops/xk/all3.blob ]; do sleep 30; done; {KV} data --src all3 --ft pvl3 --run pvl_ft3 --bs 16 --epochs 2 --val_max 800", 0, [], ["xk_eval"])
    J("xk2_pvl_smoke", f"head -n 64 {k2}/val.jsonl > {k2}/val_smoke.jsonl && rm -rf {r2}_smoke && "
      + _pf_smoke(f"{k2}/pvl16_full.yaml", f"{r2}_smoke", 6, f"{k2}/val_smoke.jsonl")
      + f" && {_ckpt_xk('export pvl 1 pvl_ft3_smoke')} && {KVI} --subset test --src all3 --model_path {r2}_smoke/export/ep1 --out smoke2 --limit 32 && rm -rf {r2}_smoke",
      4, ["xk2_data"], stall=40)
    J("xk2_pvl", f"env HW_DATASET=xk {PY['pf']} {m('pf_train')} launch {k2}/pvl16_full.yaml", 4, ["xk2_pvl_smoke"], stall=60)
    J("xk2_pvl_sel", _ckpt_xk("select_loss pvl pvl_ft3"), 0, ["xk2_pvl"], io=True, stall=40)
    for s_ in ("tune", "test"):
        J(f"xk2_pvl_infer_{s_}", f"{KVI} --subset {s_} --src all3 --model_path {r2}/export/best --out pvlkie2", 1, ["xk2_pvl_sel"], stall=40)
    J("xk2_eval", f"{PB} {m('kie_eval')} --subset test --methods ocr:ppocrv5ft,ocr:pvlft,ocr:ovisft,vlm:pvlkie,vlm:pvlkie2 --db --auto", 0,
      ["xk2_pvl_infer_test", "xk2_pvl_infer_tune"])


def kie_jobs_page(J):
    """검출 없는 전체 페이지 KIE(xkp): 실험 1 본 실험 · 2 DB 밖 · 3 키 순서 · 4 두 번째 VLM · 5 데이터량 곡선."""
    VI = f"{KVI} --qset xkp --src page"
    ocrm = "ocr:ppocrv5ft,ocr:pvlft,ocr:ovisft"

    def pvl_page(tag, frac, after):                                    # PaddleOCR-VL-1.6 전체 페이지 KIE (frac = 학습 페이지 비율)
        ft, run = f"pvlp{tag}", f"pvl_ftp{tag}"
        J(f"xkp{tag}_data", f"{KV} data --qset xkp --src page --ft {ft} --run {run} --bs 4 --epochs 3 --val_max 300 --train_frac {frac}", 0, [], after)
        J(f"xkp{tag}_smoke", f"head -n 32 ft/xk/{ft}/val.jsonl > ft/xk/{ft}/val_smoke.jsonl && rm -rf runs/xk/{run}_smoke && "
          + _pf_smoke(f"ft/xk/{ft}/pvl16_full.yaml", f"runs/xk/{run}_smoke", 4, f"ft/xk/{ft}/val_smoke.jsonl")
          + f" && {_ckpt_xk(f'export pvl 1 {run}_smoke')} && {VI} --model_path runs/xk/{run}_smoke/export/ep1 --out smokep --limit 16 && rm -rf runs/xk/{run}_smoke",
          4, [f"xkp{tag}_data"], stall=60)
        J(f"xkp{tag}_pvl", f"env HW_DATASET=xk {PY['pf']} {m('pf_train')} launch ft/xk/{ft}/pvl16_full.yaml", 4, [f"xkp{tag}_smoke"], stall=90)
        J(f"xkp{tag}_sel", _ckpt_xk(f"select_loss pvl {run}"), 0, [f"xkp{tag}_pvl"], io=True)
        for s_ in (("tune", "test") if not tag else ("test",)):
            J(f"xkp{tag}_infer_{s_}", f"{VI} --subset {s_} --model_path runs/xk/{run}/export/best --out pvlkiep{tag}", 1, [f"xkp{tag}_sel"], stall=40)
        return f"xkp{tag}_infer_test"
    # 실험 1: 전체 페이지 KIE (PaddleOCR-VL-1.6) + 대조군 · DB 조회 · 자동화율
    last = pvl_page("", 1.0, ["xk2_eval"])
    J("xkp_eval", f"{PB} {m('kie_eval')} --qset xkp --subset test --methods {ocrm},vlm:pvlkiep --db --auto", 0, [last, "xkp_infer_tune"])
    # 실험 2: DB 커버리지(DB 밖 0 · 5 · 20%) + 되돌림 규칙
    J("xkp_oov", f"{PB} {m('kie_oov')} --qset xkp --methods ocr:pvlft,ocr:ppocrv5ft,vlm:pvlkiep", 0, ["xkp_eval"])
    # 실험 3: 키 순서 일관성 · 결합 오류
    J("xkp_order_infer", f"{KVI} --qset xkpperm --src page --subset test --model_path runs/xk/pvl_ftp/export/best --out pvlkiepperm", 1, ["xkp_sel"], stall=40)
    J("xkp_order", f"{PB} {m('kie_order')} score --qset xkp --out pvlkiep", 0, ["xkp_order_infer", "xkp_infer_test"])
    # 실험 5: 학습 데이터량 곡선 (학습 페이지 10% · 30%)
    l10 = pvl_page("10", 0.1, ["xkp_pvl"])
    l30 = pvl_page("30", 0.3, ["xkp10_pvl"])
    # 실험 4: 두 번째 VLM (OvisOCR2) 전체 페이지 KIE
    J("xkp_ovisdata", f"{KV} data_ovis --qset xkp --src page --ft ovisp --val_max 300", 0, [], ["xkp30_pvl"], io=True, stall=40)
    OV = (f"env HW_DATASET=xk OVIS_TRAIN=ft/xk/ovisp/train.jsonl bash {SH}/ovis_train.sh --val_dataset ft/xk/ovisp/val.jsonl --per_device_eval_batch_size 4 "
          "--max_length 4096 --per_device_train_batch_size 4 --gradient_accumulation_steps 8")
    J("xkp_ovis_smoke", f"head -n 64 ft/xk/ovisp/train.jsonl > ft/xk/ovisp/train_smoke.jsonl && head -n 16 ft/xk/ovisp/val.jsonl > ft/xk/ovisp/val_smoke.jsonl && "
      f"rm -rf runs/xk/ovis_ft_smoke && env HW_DATASET=xk OVIS_TRAIN=ft/xk/ovisp/train_smoke.jsonl bash {SH}/ovis_train.sh --val_dataset ft/xk/ovisp/val_smoke.jsonl "
      f"--per_device_eval_batch_size 4 --max_length 4096 --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --max_steps 4 --save_strategy steps --save_steps 4 "
      f"--eval_strategy steps --eval_steps 4 && {_ckpt_xk('export ovis 1 ovis_ft_smoke')} && "
      f"{VI} --model ovis --model_path runs/xk/ovis_ft_smoke/export/ep1 --out smokepo --limit 16 && rm -rf runs/xk/ovis_ft_smoke", 4, ["xkp_ovisdata"], stall=60)
    J("xkp_ovis", f"env OUT=runs/xk/ovis_ftp {OV} --eval_strategy epoch --num_train_epochs 3", 4, ["xkp_ovis_smoke"], stall=90)
    J("xkp_ovis_sel", _ckpt_xk("select_loss ovis ovis_ftp"), 0, ["xkp_ovis"], io=True)
    for s_ in ("tune", "test"):
        J(f"xkp_ovis_infer_{s_}", f"{VI} --model ovis --subset {s_} --model_path runs/xk/ovis_ftp/export/best --out ovkiep", 1, ["xkp_ovis_sel"], stall=40)
    J("xkp_eval_all", f"{PB} {m('kie_eval')} --qset xkp --subset test --methods {ocrm},vlm:pvlkiep,vlm:pvlkiep10,vlm:pvlkiep30,vlm:ovkiep --db --auto", 0,
      ["xkp_ovis_infer_test", "xkp_ovis_infer_tune", l10, l30])


def only(jobs, prefix):
    """계획의 일부만(예: 'S_xc' = xc 단계 스모크): 이름이 prefix로 시작하는 작업, 의존은 그 안에서만."""
    keep = [dict(j) for j in jobs if j["name"].startswith(prefix)]
    names = {j["name"] for j in keep}
    for j in keep:
        j["deps"] = [d for d in j["deps"] if d in names]
        j["after"] = [d for d in j["after"] if d in names]
    return keep
