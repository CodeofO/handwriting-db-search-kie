"""위치 정렬 레코드 검색 모델 (Position-Aligned Record Retrieval Model).

구조
  이미지: 필드 이미지 1~3장(64×1280) → 보폭 합성곱 CNN → 세로 8줄을 채널로 접기(세로 평균 없음) → 열 토큰 320개 → Transformer 6층.
  텍스트: 단위 토큰 + (필드 번호 · 필드 안 글자 순번 · 자모 역할 · 레코드 안 순번) 임베딩 → Transformer 6층.
  1단 검색: 공용 학습 쿼리 16개로 양쪽을 슬롯 16개(× 64차원)로 요약, 이어 붙인 벡터의 내적 = 슬롯 코사인 평균.
  2단 재채점
    - PLI(위치 정렬 정밀 비교기): 텍스트 단위(필드 f, 글자 순번 p)는 필드 f 이미지의 예상 위치
      μ = 잉크 시작 + (p + 0.5)/n_f × 잉크 폭 근처 열 토큰하고만 비교(가우시안 위치 사전, σ = 1.5 글자 폭).
    - 글자 검증: 인식 전용 층 + CTC 머리로 후보 문자열의 −CTC NLL / 단위 수.
    - 최종 점수 = mix0·1단 코사인 + mix1·PLI + mix_ctc·글자 검증 (모두 학습).
  수락 판정 머리(vhead): 1·2등 점수 · 필드별 최소 일치 특징 14개 → 1등이 정답일 로짓(확신도).
체크포인트 state_dict 키 이름은 원본 그대로다(기존 가중치 strict 로드).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- 부품
class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, 1, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sc = nn.Identity() if cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.gelu(self.b1(self.c1(x)))
        return F.gelu(self.b2(self.c2(y)) + self.sc(x))


def transformer(d, layers, heads=8):
    """pre-LN Transformer 인코더 (GELU, FFN 4d, dropout 0.1)."""
    layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class AttnPool(nn.Module):
    """학습 쿼리 k개로 시퀀스를 요약 → (B, k, d)."""
    def __init__(self, d, k=1, heads=8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, k, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln = nn.LayerNorm(d)

    def forward(self, x, key_padding_mask=None):
        q = self.q.expand(x.size(0), -1, -1)
        o, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return self.ln(o)


class ImageEnc3(nn.Module):
    """필드 이미지 (N, 1, 64, 1280) → 열 토큰 (N, 320, d). 줄이기는 보폭 합성곱으로만, 세로 8줄은 채널로 접는다."""
    def __init__(self, H=64, W=1280, d=384, layers=6, c=256):
        super().__init__()
        assert H % 8 == 0 and W % 4 == 0
        cb = lambda i, o, st: nn.Sequential(nn.Conv2d(i, o, 3, st, 1, bias=False), nn.BatchNorm2d(o), nn.GELU())
        self.cnn = nn.Sequential(
            cb(1, 64, 2), ResBlock(64, 64),                                    # H/2, W/2
            cb(64, 128, 2), ResBlock(128, 128), ResBlock(128, 128),            # H/4, W/4
            cb(128, c, (2, 1)), ResBlock(c, c), ResBlock(c, c))                # H/8, W/4
        self.rows = H // 8
        self.fold = nn.Sequential(nn.Linear(c * self.rows, d), nn.LayerNorm(d))   # 세로 접기: 줄 × 채널 → d
        self.pos = nn.Parameter(torch.zeros(1, W // 4, d))
        self.tf = transformer(d, layers)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        f = self.cnn(x)                                     # (N, c, H/8, W/4)
        N, C, R, T = f.shape
        f = f.permute(0, 3, 2, 1).reshape(N, T, R * C)      # 열마다 모든 줄의 채널을 이어 붙임
        return self.ln(self.tf(self.fold(f) + self.pos[:, :T]))


# ---------------------------------------------------------------- 모델
class RecordEncoder3(nn.Module):
    BUCKET = 32          # 이미지 수를 이 배수로 채워 인코딩(cuDNN 튜닝 재사용). 학습 모드 BN 통계에도 들어가므로 바꾸지 않는다.
    NVF = 14             # 수락 판정 머리 입력 특징 수
    LMAX = 48            # 글자 수 머리(len_head) 출력 크기 − 1 (최종 프로토콜에서는 쓰지 않음, 체크포인트 호환용)

    def __init__(self, vocab, maxu=160, n_slots=16, max_images=3, H=64, W=1280, d=384, dim=64, layers=6, pdim=128, read_layers=0):
        super().__init__()
        # 모듈 생성 순서 = 원본 순서(같은 시드면 같은 초기값)
        self.read = transformer(d, read_layers) if read_layers else None   # 인식 전용 층: 글자 인식(CTC)만 거친다
        self.N, self.dim, self.max_images, self.W = n_slots, dim, max_images, W
        self.img = ImageEnc3(H, W, d, layers)
        self.img_id = nn.Embedding(max_images, d)
        self.ipool = AttnPool(d, n_slots)
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.tpos = nn.Embedding(64, d)                     # 필드 안 글자 순번
        self.tfld = nn.Embedding(8, d)
        self.trole = nn.Embedding(6, d)
        self.tabs = nn.Embedding(maxu + 1, d)               # 레코드 안 단위 순번(보조)
        self.ttf = transformer(d, layers)
        self.tln = nn.LayerNorm(d)
        self.tpool = AttnPool(d, n_slots)
        self.iproj, self.tproj = nn.Linear(d, dim), nn.Linear(d, dim)
        self.iproj2, self.tproj2 = nn.Linear(d, pdim), nn.Linear(d, pdim)   # PLI용 토큰 투영
        self.ctc_head = nn.Linear(d, vocab)                 # 글자(자모 단위) 인식 CTC 머리 (0 = blank = pad)
        self.mix_ctc = nn.Parameter(torch.tensor(0.0))      # 최종 점수의 글자 검증 점수 가중치
        self.vhead = nn.Sequential(nn.Linear(self.NVF, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))   # 수락 판정 머리
        self.len_head = nn.Sequential(nn.Linear(d + 2, d), nn.GELU(), nn.Linear(d, self.LMAX + 1))      # 미사용(체크포인트 호환)
        self.mix_len = nn.Parameter(torch.tensor(0.0))                                                   # 미사용(체크포인트 호환)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.pli_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.pli_tau = 0.1
        self.mix = nn.Parameter(torch.tensor([0.0, 1.0]))   # 최종 점수 = mix0·(1단 코사인) + mix1·(2단 PLI)
        self.sigma_k = 1.5
        for pool in (self.ipool, self.tpool):               # 쿼리 대칭 깨기
            nn.init.orthogonal_(pool.q.data[0]); pool.q.data *= d ** 0.5

    # ------------------------------------------------------------ 인코더
    def image_tokens(self, x, img_mask):
        """x: (B, M, 1, H, W), img_mask: (B, M) → 토큰 (B, M, T, d) (패드 이미지는 0)."""
        B, M = img_mask.shape
        T = self.W // 4
        h = torch.zeros(B * M, T, self.img.pos.size(2), device=x.device, dtype=x.dtype)
        v = img_mask.flatten()
        xv = x.flatten(0, 1)[v]
        n = xv.size(0); nb = -(-n // self.BUCKET) * self.BUCKET
        if nb > n:
            xv = torch.cat([xv, xv.new_ones(nb - n, *xv.shape[1:])])
        xv = xv.contiguous(memory_format=torch.channels_last) if xv.is_cuda else xv
        h[v] = self.img(xv)[:n].to(h.dtype)
        return h.view(B, M, T, -1)

    def slots_image(self, h, img_mask):
        """이미지 토큰 → 정규화 슬롯 (B, N, dim). 패드 이미지 토큰은 가린다."""
        B, M, T, d = h.shape
        h = h + self.img_id.weight[:M].view(1, M, 1, -1)
        kpm = (~img_mask[:, :, None].expand(B, M, T)).flatten(1)
        z = self.ipool(h.flatten(1, 2), key_padding_mask=kpm)
        return F.normalize(self.iproj(z), dim=-1)

    def text_tokens(self, ids, fld, pos, role):
        L = ids.size(1)
        h = (self.emb(ids) + self.tpos(pos.clamp(max=63)) + self.tfld(fld.clamp(max=7)) + self.trole(role)
             + self.tabs(torch.arange(1, L + 1, device=ids.device)))
        return self.tln(self.ttf(h, src_key_padding_mask=ids == 0))

    def slots_text(self, ht, ids):
        z = self.tpool(ht, key_padding_mask=ids == 0)
        return F.normalize(self.tproj(z), dim=-1)

    @staticmethod
    def flat(z, slot_mask=None):
        """(B, N, dim) → 검색 벡터 (B, N·dim), 내적 = 슬롯 코사인 평균. slot_mask(학습 슬롯 드롭아웃): True = 사용."""
        if slot_mask is not None:
            z = z * slot_mask.unsqueeze(-1)
            n = slot_mask.sum(1, keepdim=True).clamp(min=1).unsqueeze(-1)
            return (z / n.sqrt()).flatten(1)
        return (z / z.size(1) ** 0.5).flatten(1)

    @staticmethod
    def diversity(z):
        """슬롯 다양성 손실: 같은 샘플 안 슬롯 쌍 코사인 제곱 평균."""
        N = z.size(1)
        if N < 2:
            return z.new_zeros(())
        g = z @ z.transpose(1, 2)
        off = ~torch.eye(N, dtype=torch.bool, device=z.device)
        return (g[:, off] ** 2).mean()

    # ------------------------------------------------------------ 글자 인식 (CTC)
    def read_feats(self, hi):
        """글자 인식용 토큰: (…, T, d) → 인식 전용 층(있으면) 통과."""
        if self.read is None:
            return hi
        sh = hi.shape
        return self.read(hi.reshape(-1, sh[-2], sh[-1])).reshape(sh)

    def ctc_loss(self, hi, img_mask, targets):
        """실제 이미지마다 열 토큰 → 자모 단위 확률, 그 필드 문자열을 정답으로. targets: 이미지 순서의 단위 id 목록들."""
        v = img_mask.flatten()
        h = self.read_feats(hi.flatten(0, 1)[v])                           # (n, T, d)
        lp = F.log_softmax(self.ctc_head(h).float(), -1).transpose(0, 1)   # (T, n, C)
        tl = torch.tensor([len(t) for t in targets], device=h.device)
        tg = torch.tensor([u for t in targets for u in t], device=h.device)
        il = torch.full((h.size(0),), h.size(1), dtype=torch.long, device=h.device)
        return F.ctc_loss(lp, tg, il, tl, blank=0, reduction="mean", zero_infinity=True)

    def cand_ctc(self, hi, img_mask, units, cand, per_slot=False):
        """후보 레코드 글자 검증 점수(미분 가능): 질의 b의 이미지 j ↔ 후보 텍스트의 j번째 비지 않은 필드.
        units: 텍스트별 [필드별 단위 id 목록](빈 필드 제외), cand: (B, K) → (B, K) = −Σ CTC NLL / Σ 단위 수.
        per_slot이면 필드(이미지)별 점수 (B, K, M)와 유효 마스크도 돌려준다."""
        B, M, T, _ = hi.shape; K = cand.size(1)
        lp = F.log_softmax(self.ctc_head(self.read_feats(hi)).float(), -1)  # (B, M, T, C)
        nimg = img_mask.sum(1).tolist(); cl = cand.tolist()
        seqs, tg, tl, owner, slot = [], [], [], [], []
        for b in range(B):
            for k in range(K):
                for j, u in enumerate(units[cl[b][k]][:nimg[b]]):
                    if u:
                        seqs.append(b * M + j); tg += u; tl.append(len(u)); owner.append(b * K + k); slot.append((b * K + k) * M + j)
        if not seqs:
            z = hi.new_zeros(B, K)
            return (z, hi.new_zeros(B, K, M), torch.zeros(B, K, M, dtype=torch.bool, device=hi.device)) if per_slot else z
        L = lp.view(B * M, T, -1)[torch.tensor(seqs, device=hi.device)].transpose(0, 1)   # (T, N, C)
        nll = F.ctc_loss(L, torch.tensor(tg, device=hi.device), torch.full((len(seqs),), T, dtype=torch.long, device=hi.device),
                         torch.tensor(tl, device=hi.device), blank=0, reduction="none", zero_infinity=True)
        ow = torch.tensor(owner, device=hi.device)
        tot = torch.zeros(B * K, device=hi.device).index_add(0, ow, nll)
        nu = torch.zeros(B * K, device=hi.device).index_add(0, ow, torch.tensor(tl, device=hi.device, dtype=torch.float))
        out = (-tot / nu.clamp(min=1)).view(B, K)
        if not per_slot:
            return out
        sl = torch.tensor(slot, device=hi.device); tlf = torch.tensor(tl, device=hi.device, dtype=torch.float)
        ps = torch.zeros(B * K * M, device=hi.device).index_add(0, sl, nll)
        pn = torch.zeros(B * K * M, device=hi.device).index_add(0, sl, tlf)
        return out, (-ps / pn.clamp(min=1)).view(B, K, M), (pn > 0).view(B, K, M)

    # ------------------------------------------------------------ 결합 · 수락 판정
    def final_score(self, s1, s2):
        """1단 코사인 s1과 2단 PLI 점수 s2의 학습된 결합."""
        return self.mix[0] * s1 + self.mix[1] * s2

    def verify_feats(self, sc, sp, s3, pp, ppm, pg, pgm, s1c, rank=None):
        """수락 판정 특징. sc: 결합 점수 (B, K), sp: PLI, s3: 글자 검증, s1c: 1단 코사인 (B, K),
        pp/pg: 필드별 PLI · 글자 검증 (B, K, M), ppm/pgm: 유효 마스크. 레코드 평균이 가리는 '한 필드만 어긋남'을 필드 최소값으로 드러낸다.
        rank: 1·2등을 고르는 점수(평가 때 λ를 더한 최종 점수). 없으면 sc. 특징 값은 sc에서 읽는다.
        → (B, NVF) 특징, 1등 후보 위치 (B,)"""
        g = lambda t, c: t.gather(1, c[:, None]).squeeze(1)
        o2 = (sc if rank is None else rank).topk(2, 1).indices; c1, c2 = o2[:, 0], o2[:, 1]
        v2 = torch.stack([g(sc, c1), g(sc, c2)], 1)
        gm = lambda t, c: t.gather(1, c[:, None, None].expand(-1, 1, t.size(2))).squeeze(1)

        def mn(t, msk):
            r = torch.where(msk, t, torch.full_like(t, 1e4)).amin(1)
            return torch.where(msk.any(1), r, torch.zeros_like(r))
        m1, n1 = gm(ppm, c1) & gm(ppm, c2), gm(pgm, c1) & gm(pgm, c2)
        p1, p2, q1, q2 = gm(pp, c1), gm(pp, c2), gm(pg, c1), gm(pg, c2)
        f = torch.stack([v2[:, 0], v2[:, 0] - v2[:, 1], g(sp, c1), g(sp, c1) - g(sp, c2), g(s3, c1), g(s3, c1) - g(s3, c2),
                         mn(p1, gm(ppm, c1)), mn(p1 - p2, m1), mn(q1, gm(pgm, c1)), mn(q1 - q2, n1), g(s1c, c1), g(s1c, c1) - g(s1c, c2),
                         torch.where(gm(pgm, c1), q1, torch.zeros_like(q1)).sum(1),   # 필드별 글자 검증 합
                         (gm(pgm, c1).sum(1)).float()], 1)
        return f.float().clamp(-30, 30), c1

    def verify(self, f):
        return self.vhead(f).squeeze(-1)

    # ------------------------------------------------------------ 위치 정렬 정밀 비교기
    @staticmethod
    def ink_extent(x, img_mask, thr=0.0):
        """정규화 입력(흰색 = 1)에서 잉크 열의 처음 · 끝 (토큰 단위, 4px). x: (B, M, 1, H, W) → (B, M, 2)."""
        col = (x.squeeze(2).amin(2) < thr)                                    # (B, M, W)
        W = col.size(-1); idx = torch.arange(W, device=x.device)
        big = torch.full_like(col, W, dtype=torch.long)
        s = torch.where(col, idx, big).amin(-1)
        e = torch.where(col, idx, torch.full_like(big, -1)).amax(-1)
        none = e < 0
        s = torch.where(none, torch.zeros_like(s), s); e = torch.where(none, torch.full_like(e, W - 1), e)
        return torch.stack([s, e + 1], -1).float() / 4.0

    def pli(self, hi, ext, img_mask, ht, fld, pos, ids, flen, cand, per_slot=False):
        """hi: (B, M, T, d) 질의 이미지 토큰, ext: (B, M, 2) 잉크 범위, ht: (C, L, d) 후보 텍스트 토큰,
        fld/pos/ids: (C, L), flen: (C, F) 필드 글자 수(빈 필드 0), cand: (B, K) 질의마다 비교할 후보 텍스트 번호 → 점수 (B, K).
        필드 f의 텍스트 토큰은 그 필드 이미지(비지 않은 필드 순서 = 이미지 순서)의 예상 위치 근처 열 토큰하고만 비교한다.
        per_slot이면 이미지(필드)별 평균 (B, K, M)과 유효 마스크도 돌려준다."""
        B, M, T, _ = hi.shape
        K = cand.size(1); L = ht.size(1)
        I = F.normalize(self.iproj2(hi), dim=-1).reshape(B, M * T, -1)           # (B, M·T, p)
        Tt = F.normalize(self.tproj2(ht), dim=-1)                                 # (C, L, p)
        S = torch.einsum("bklp,bnp->bkln", Tt[cand], I).view(B, K, L, M, T)
        img_of_field = torch.cumsum((flen > 0).long(), 1) - 1                     # (C, F)
        f_ = fld.clamp(max=flen.size(1) - 1)
        img_idx = torch.gather(img_of_field, 1, f_)[cand]                          # (B, K, L)
        n_f = torch.gather(flen, 1, f_).clamp(min=1).float()[cand]                 # (B, K, L)
        sel = img_idx.clamp(min=0, max=M - 1)
        S = torch.gather(S, 3, sel[..., None, None].expand(B, K, L, 1, T)).squeeze(3).float()   # (B, K, L, T)
        e_u = torch.gather(ext[:, None, None].expand(B, K, L, M, 2), 3, sel[..., None, None].expand(B, K, L, 1, 2)).squeeze(3)
        width = (e_u[..., 1] - e_u[..., 0]).clamp(min=1.0)
        mu = e_u[..., 0] + (pos[cand].float() + 0.5) / n_f * width
        sig = (self.sigma_k * width / n_f).clamp(min=1.0)
        tpos = torch.arange(T, device=S.device).float()
        logw = -0.5 * ((tpos - mu[..., None]) / sig[..., None]) ** 2
        has_img = torch.gather(img_mask, 1, sel.view(B, -1)).view(B, K, L) & (img_idx >= 0)
        su = self.pli_tau * torch.logsumexp(S / self.pli_tau + logw, -1)          # (B, K, L)
        su = torch.where(has_img, su, torch.full_like(su, -1.0))
        m = (ids[cand] > 2).float()
        out = (su * m).sum(-1) / m.sum(-1).clamp(min=1)
        if not per_slot:
            return out
        mm = (m * has_img.float())
        ps = torch.zeros(B, K, M, device=su.device).scatter_add(2, sel, su.float() * mm)
        pn = torch.zeros(B, K, M, device=su.device).scatter_add(2, sel, mm)
        return out, ps / pn.clamp(min=1), pn > 0


def build_model(vocab_size, maxu=160, read_layers=3, H=64, W=1280, d=384, dim=64, layers=6, slots=16):
    """최종 프로토콜 구조(d 384 · 6층 · 인식 전용 3층 · 슬롯 16 × 64 · 64×1280)."""
    return RecordEncoder3(vocab_size, maxu, slots, 3, H, W, d=d, dim=dim, layers=layers, read_layers=read_layers)
