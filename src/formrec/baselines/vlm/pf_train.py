"""공식 PaddleFormers 학습 실행기. `paddleformers-cli train cfg.yaml`이 하는 일을 그대로 하되
(환경변수 설정 → paddle.distributed.launch → paddleformers/cli/launcher.py train), 각 rank가 launcher 대신 이 파일을 실행해
이미지 읽기만 blob 참조("blob:train/123")도 읽도록 패치한다. 학습 루프·템플릿·전처리·저장은 모두 공식 코드.

사용 (GPU 4장, 작업 공간에서): python -m formrec.baselines.vlm.pf_train launch ft/xa/pvl/pvl16_full.yaml [key=value ...]
"""
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))     # 저장소 src/ (rank가 이 파일을 스크립트로 실행할 때 formrec를 찾게)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]   # 이 폴더의 모듈 이름이 공식 코드를 가리지 않게


def norm_ref(u):
    """PaddleFormers는 상대 경로 앞에 데이터 파일 폴더를 붙인다: ".../ft/xa/pvl/blob:train/5" → "blob:train/5"."""
    from formrec.common import blob
    if isinstance(u, str) and blob.PREFIX in u:
        return u[u.index(blob.PREFIX):]
    return u


def patch():
    import importlib
    from formrec.common import blob
    sft_mod = importlib.import_module("paddleformers.datasets.SFTDataset")
    from paddleformers.datasets.template import mm_plugin
    orig_dl = mm_plugin.BasePlugin._file_download

    def _file_download(self, url):
        if blob.is_ref(norm_ref(url)):
            return io.BytesIO(blob.read(norm_ref(url)))
        return orig_dl(self, url)
    mm_plugin.BasePlugin._file_download = _file_download

    cls = sft_mod.BaseSFTDataset
    if hasattr(cls, "_get_image_size"):
        orig_sz = cls._get_image_size

        def _get_image_size(image):
            if blob.is_ref(norm_ref(image)):
                image = {"bytes": blob.read(norm_ref(image))}
            return orig_sz(image)
        cls._get_image_size = staticmethod(_get_image_size)


def launch_cmd(argv, env):
    """부모: cli.main()과 같은 환경으로 분산 실행할 명령(--host 127.0.0.1: 없으면 공인 IP로 TCP store를 잡아 멈춤)."""
    from formrec.common import workspace as cfg
    cards = env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    log_dir = str(cfg.LOGS / "pdist" / env.get("KIE_JOB", "pf_train"))   # 실행기(run_job.sh)가 rank 로그도 감시
    return [sys.executable, "-m", "paddle.distributed.launch", "--log_dir", log_dir, "--gpus", cards,
            "--master", "127.0.0.1:" + env.get("MASTER_PORT", "8080"), "--host", "127.0.0.1", "--nnodes", "1", "--rank", "0",
            "--run_mode=collective", os.path.abspath(__file__), "train", *argv]


def main():
    if sys.argv[1] == "launch":
        env = dict(os.environ)
        for k, v in {"FLAGS_set_to_1d": "False", "NVIDIA_TF32_OVERRIDE": "0",
                     "FLAGS_dataloader_use_file_descriptor": "False"}.items():
            env.setdefault(k, v)
        import paddleformers
        pf_root = os.path.dirname(os.path.dirname(paddleformers.__file__))
        env["PYTHONPATH"] = os.pathsep.join([pf_root, SRC, env.get("PYTHONPATH", "")])
        sys.exit(subprocess.call(launch_cmd(sys.argv[2:], env), env=env))
    sys.path.insert(0, SRC)
    patch()   # 각 rank
    from paddleformers.cli import launcher
    launcher.launch()


if __name__ == "__main__":
    main()
