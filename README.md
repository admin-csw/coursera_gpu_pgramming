# Coursera GPU Programming — 실습 및 과제

**저장소:** [admin-csw/coursera_gpu_pgramming](https://github.com/admin-csw/coursera_gpu_pgramming)

**2026년 1월**, 북클럽 **나란**에서 진행한 스터디 모임과 맞물려 Coursera **[GPU Programming 스페셜라이제이션](https://www.coursera.org/specializations/gpu-programming)**(존스 홉킨스 대학교, 강사 Chancellor Thomas Pascale) 강의 실습·과제를 정리하는 저장소입니다.

스터디·커뮤니티 관련 안내는 **[사이버서원 (www.cyberseowon.com)](https://www.cyberseowon.com)**에서 확인할 수 있습니다.

## 스페셜라이제이션 소개

일반적으로 구할 수 있는 **NVIDIA GPU**에서 효율적으로 돌아가는 소프트웨어를 만들고 싶은 데이터 과학자·소프트웨어 개발자를 대상으로 합니다. **CUDA**와 대규모 병렬 연산용 라이브러리를 다루며, 머신러닝, 영상·음향 신호 처리, 일반 데이터 처리 등에 응용할 수 있습니다.

- **난이도:** 중급  
- **권장 학습 속도:** 전체 스페셜라이제이션 기준 주당 약 10시간으로 약 2개월(Coursera에서는 과목당 약 1개월 안내도 함께 제시함)  
- **선수 요건:** 프로그래밍 경험 1년 이상; 강의 이해와 과제 수행에는 **C/C++**에 익숙할 것을 권장합니다.

## 과목 순서

아래 순서대로 수강하는 것이 권장됩니다.

| # | 과목명 | 내용 요약 |
|---|--------|-----------|
| 1 | Introduction to Concurrent Programming with GPUs | Python·C/C++ 동시성 소프트웨어; GPU 하드웨어·소프트웨어 아키텍처 입문 |
| 2 | Introduction to Parallel Programming with CUDA | CPU + NVIDIA GPU용 CUDA; 순차 알고리즘을 GPU에서 대규모로 실행하는 커널로 전환 |
| 3 | CUDA at Scale for the Enterprise | 다중 CPU·다중 GPU 환경; 비동기·대화형 GPU 커널; CUDA·메모리·라이브러리를 활용한 영상 처리 등 |
| 4 | CUDA Advanced Libraries | **cuFFT**, **cuBLAS**, **Thrust**; **cuTensor**, **cuDNN**을 활용한 ML |

개별 과목 수강 신청 URL은 변경될 수 있으므로, 각 과목은 **[스페셜라이제이션 페이지](https://www.coursera.org/specializations/gpu-programming)**의 과목 목록에서 선택해 수강하세요.

## 실습 프로젝트

Coursera 스페셜라이제이션 설명에 따르면, 학습자는 보통 **최소 두 개의 프로젝트**(예: 영상·신호 처리를 위한 CUDA 접근 + 본인이 선택한 주제)와 짧은 시연·코드 공유 형태의 활동을 수행합니다.

## 저장소 폴더 구조 제안

과목·주차·모듈 단위로 자유롭게 나누면 됩니다. 예:

```text
course-01-concurrent-gpu/
  labs/
  assignments/
course-02-cuda-intro/
  labs/
  assignments/
course-03-cuda-at-scale/
  labs/
  assignments/
course-04-advanced-libraries/
  labs/
  assignments/
```

각 과목 아래 **`labs/`**에는 강의 실습·스타터 코드를, **`assignments/`**에는 과제·본인 풀이를 두면 나중에 다시 찾기 쉽습니다.

## 환경 안내

- C/C++ CUDA 실습에는 지원 드라이버가 설치된 **NVIDIA GPU**와 강의에서 안내하는 버전에 맞춘 **CUDA Toolkit**이 필요합니다.  
- 1번 과목에서는 동시성 주제 일부에 **Python**을 사용합니다.  
- 이후 과목에서는 **C/C++**, CUDA API, NVIDIA 라이브러리(cuBLAS, cuFFT, Thrust, cuDNN 등)를 실러버스에 따라 다룹니다.

## 학업 윤리

채점되는 과제 풀이를 이 저장소에 둘 경우, 저장소는 **비공개**로 두고 Coursera·존스 홉킨스의 명예 규정을 지키세요. 타인이 학습 과정을 우회할 수 있게 풀이를 공개하지 마세요.

## 라이선스·저작권

강의 영상·공식 과제 문구 등 코스 콘텐츠는 **Coursera**와 **존스 홉킨스 대학교**에 귀속됩니다. 본인이 작성한 코드의 라이선스는 별도로 정하시면 됩니다.
