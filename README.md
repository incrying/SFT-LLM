# SFT-LLM
Supervised Finetuning open-source llms with custom Datasets

## Before you start
- 본 코드는 meta의 llama2 non-instruction model에 최적화되어 있습니다. 다른 모델로 변경하여 사용 시, 코드 내부 수정이 필요함을 알려드립니다.
- 본 코드에서는 양자화 방법을 제공하지 않습니다.
- dataload.py에서 데이터셋 생성 양식이 적절하지 않습니다. 추후 수정하겠습니다.

## Reference
위 작업은 아래의 AI Factory(인공지능팩토리) 강의를 참고하였습니다.
- 코드: https://colab.research.google.com/github/choijhyeok/easy_finetuner/blob/main/개인_데이터셋을_통한_llama2_fine_tune.ipynb
- 영상: https://www.youtube.com/live/4I9AUFuBlFs?feature=shared
