# novelgirls
2021학년도 2학기 자연어처리 프로젝트 (+SM-PBL)

훈련된 모델 다운

- eye color
  - ```BERT/trained_model/bert_eye_color```
  - https://drive.google.com/file/d/1OSlpy4YfCLxg6ng4Z7xhHHkSuLWh3QqE/view?usp=sharing

- hair color
  - ```BERT/trained_model/bert_hair_color```
  - https://drive.google.com/file/d/1rk7TEhwTGJqIOd3gAcV55szrrYEBY-lm/view?usp=sharing

- hair nlg model.pkl
  - ```LSTM/trained_model```
  - https://drive.google.com/file/d/1ONRFnbfAVl_VDZ7IEdApBCGGhZ_OU6F9/view?usp=sharing

- eye nlg model.pkl
  - ```LSTM/trained_model```
  - https://drive.google.com/file/d/1qhvvMX1MMLPNJer_doUZiIF_JcMYQ8ki/view?usp=sharing

* 실행 요령

BERT는 tensorflow==1.15.0, LSTM은 tensorflow==2.X를 써서 같은 환경에서 돌리면 둘 중 하나에서는 에러가 발생합니다.
그래서 BERT와 LSTM의 환경을 분리하여 BERT의 결과를 LSTM으로 넘겨 사용할 예정입니다.


1. BERT와 LSTM의 디렉토리를 분리

<LSTM 디렉토리>
![image](https://user-images.githubusercontent.com/81811255/142639536-82a2fa22-bb0b-4a39-bb89-05fc46552acc.png)


<BERT 디렉토리>
![image](https://user-images.githubusercontent.com/81811255/142639618-69eb233a-4629-4f2a-8619-9baaa068b1ad.png)


2. BERT용 가상환경, LSTM용 가상환경, 총 2개의 가상환경을 생성(nl_bert, nl_lstm)
<nl_bert>

pip install tensorflow==1.15.0

pip install --upgrade tb-nightly

pip install bert-tensorflow==1.0.1

pip install pandas


<nl_lstm>

pip install tensorflow

(저는 따로 버전을 지정해주지 않았는데, 2.X 버전이면 되는 것 같습니다.)

pip install pandas


3. 각 가상환경에서 각 디렉토리를 실행

BERT의 get_output_bert를 실행하면 eye_color_pred와 hair_color_pred가 임포트되어 분류한 label이 txt파일로 저장됨

LSTM의 get_output_lstm에서 이 txt파일을 이용하여 문장 생성.

따라서 txt파일이 LSTM 디렉토리로 들어가야함.

![image](https://user-images.githubusercontent.com/81811255/142640679-46a6f494-1c07-456d-ba33-6bb2f3952e92.png)

그러므로 get_output_bert의 위 코드에 LSTM 디렉토리의 BERT 디렉토리 주소를 넣어야함.

![image](https://user-images.githubusercontent.com/81811255/142640890-827a8327-cb34-4190-828e-5534b9f4c034.png)

