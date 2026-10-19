"""위치 정렬 레코드 검색 모델(Position-Aligned Record Retrieval Model) — 제안 모델.

필드 이미지 묶음(레코드)을 슬롯 벡터 1개로 요약해 DB 문자열에서 1단 검색하고, 상위 K개를
위치 정렬 정밀 비교기(PLI) · 글자 검증(CTC) 점수로 다시 매긴다. 수락 판정 머리가 확신도를 낸다.

모듈
  tokenizer : 문자열 → 단위 토큰(한글 = 자모, 그 밖 = 글자)
  model     : RecordEncoder3(이미지 인코더 · 텍스트 인코더 · 슬롯 · PLI · CTC · 수락 판정)
  data      : 분할표 · 통합 어휘 · 이미지 저장소 · 레코드 표집 · 합성 음성
  augment   : 이미지 증강(학습) / 정규화(평가)
  pretrain  : 1단계 CTC 사전학습       train : 2단계 데이터셋별 FT
  evaluate  : 단계별 검색 평가(select · eval · epochs · db)
  summary   : 방법 × 단계 지표 표(제안 모델 · 비교군 공용)
"""
