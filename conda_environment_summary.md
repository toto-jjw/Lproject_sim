# Conda Environment Summary
> 문서화 날짜: 2026-02-24  
> 환경 경로: `/home/jaewon/Lproject_sim/.conda`  
> 재현용 파일: `conda_environment.yml`

---

## 기본 정보

| 항목 | 값 |
|------|-----|
| Python 버전 | **3.14.2** |
| 채널 | `defaults` |
| 환경 타입 | 프로젝트 로컬 conda 환경 |

---

## 주요 패키지 (Python 라이브러리)

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `python` | 3.14.2 | 인터프리터 |
| `numpy` | 2.4.1 | 수치 연산 |
| `pandas` | 3.0.0 | 데이터 분석 |
| `matplotlib` | 3.10.8 | 데이터 시각화 |
| `seaborn` | 0.13.2 | 통계 시각화 |
| `pillow` | 12.1.0 | 이미지 처리 |
| `lmdb` | 0.9.31 | 데이터셋 DB (NAFNet 등) |
| `ipython` | 9.7.0 | 인터랙티브 Python 셸 |
| `ipykernel` | 6.31.0 | Jupyter 커널 |
| `jupyter_client` | 8.8.0 | Jupyter 클라이언트 |
| `jupyter_core` | 5.9.1 | Jupyter 코어 |
| `pyqt` | 6.9.1 | GUI / Qt 바인딩 |
| `mkl` | 2025.0.0 | Intel 수학 라이브러리 |
| `mkl_fft` | 2.1.1 | FFT (MKL 기반) |
| `mkl_random` | 1.3.0 | 난수 생성 (MKL 기반) |
| `setuptools` | 80.10.2 | 패키지 관리 |
| `pip` | 25.3 | 패키지 설치 |

## Jupyter / 개발 도구

| 패키지 | 버전 |
|--------|------|
| `ipykernel` | 6.31.0 |
| `ipython` | 9.7.0 |
| `jupyter_client` | 8.8.0 |
| `jupyter_core` | 5.9.1 |
| `debugpy` | 1.8.16 |
| `jedi` | 0.19.2 |
| `traitlets` | 5.14.3 |
| `tornado` | 6.5.4 |
| `pyzmq` | 27.1.0 |

---

## 환경 복원 방법

```bash
# 방법 1: yml 파일로 동일 환경 재생성 (로컬 경로)
conda env create -p /home/jaewon/Lproject_sim/.conda -f conda_environment.yml

# 방법 2: 이름 있는 환경으로 복원
conda env create -n lproject_sim -f conda_environment.yml

# 방법 3: 크로스플랫폼 호환 (빌드 해시 제외)
conda env export --no-builds -p /home/jaewon/Lproject_sim/.conda > environment_noBuild.yml
conda env create -n lproject_sim -f environment_noBuild.yml
```

---

## 전체 패키지 목록

전체 패키지 목록(빌드 해시 포함)은 `conda_environment.yml` 파일을 참고하세요.
