# Verse-only 기타 리프 생성 (VAE + Latent Diffusion)

## 프로젝트 소개
이 프로젝트는 **기타 곡의 Verse 구간 리프**를 생성하는 모델을 구현한 것이다.  
수업시간에 배웠던 Variational Autoencoder(VAE)와 Latent Diffusion Model을 결합하여  
기타 Verse 리프를 MIDI 형태로 생성한다.

곡 전체를 생성하는 대신,  
기타 연주자의 스타일이 가장 잘 드러나고 반복 구조가 뚜렷한  
**Verse 구간(2마디)**만을 데이터로 사용하였다.

이 프로젝트를 시작하게 된 가장 큰 이유는  
제가 Red Hot Chili Peppers의 기타리스트 **John Frusciante의 팬**이기 때문이다.

John Frusciante의 연주는 Jimi Hendrix를 연상시키는 화려한 솔로도 있지만
곡의 분위기를 지탱하는 멜로딕한 리프가 가장 큰 매력이다.
그리고 반복 속에서 미묘하게 변하는 뉘앙스 또한 인상을준다.

이 프로젝트에서 Verse 구간만을 선택한 이유는 

크게 2가지 이유가있다.

1.곡 전체를 모델링하는 경우 
다음과 같은 문제가 있다.
- 시퀀스가 길어져 학습이 불안정해짐
- 많은 데이터가 필요함
- 구조(Verse / Chorus / Bridge 등)가 복잡함

2.VAE의 특성을 생각해보자
 VAE는 개별 샘플을 암기하는 모델이 아니라,  
데이터 전체가 이루는 분포를 연속적인 latent space로 학습하는 모델이다.  
따라서 입력 데이터가 반복적이고 구조적으로 유사할수록  
latent space가 안정적으로 형성된다.

기타 곡에서 Verse 구간은  
코드 진행과 리듬 패턴이 반복되며,  
음의 개수와 변화 폭이 상대적으로 제한되어 있다.  
이는 VAE가 하나의 일관된 분포로 인식하기에 적합한 특성이다.

특히 John Frusciante의 기타 연주에서  
Verse 파트는 화려한 솔로보다는  
반복되는 리프와 미세한 변주로 구성되어 있어,  
VAE가 학습하기에 이상적인 데이터 구조를 가진다고 볼 수 있다.

그래서 이 프로젝트에서는  
**Verse 기타 리프만 사용하면 적은 데이터로도 의미 있는 결과를 낼 수 있다**는 가정 하에  
Verse-only 데이터셋을 구성하였다.

## 데이터 구성

- 각 데이터는 **Verse 구간 2마디**
  
- 16분음표 단위로 양자화
 기타 Verse 리프는 대부분 8분음표 또는 16분음표 기반의 리듬으로 구성되며,  
 16분음표는 기타 연주에서 발생하는 리듬 변화를 충분히 표현하면서도  
 과도하게 시퀀스를 길게 만들지 않는 최소 단위라고 판단하였다.

 8분음표 단위로 양자화할 경우  
 기타의 미묘한 리듬 변화를 표현하기 어렵고
 32분음표 단위는 실제 기타 Verse 연주에서 거의 사용되지 않으며  
 시퀀스 길이만 불필요하게 증가시킨다.

 또한 16분음표 단위로 양자화하면  
 2마디 Verse 리프를 정확히 32 step의 고정 길이 시퀀스로 표현할 수 있어,  
 VAE 학습 시 배치 구성과 모델 설계가 단순해지는 장점이 있다.

 따라서 본 프로젝트에서는  
 리듬 표현력과 모델 학습 안정성 사이의 균형을 고려하여  
 16분음표 단위를 양자화 기준으로 선택하였다.

- 길이 32 step의 단일 음(monophonic) 시퀀스(2마디를 16분음표=32step)
 Verse 구간의 기타 연주는  
 복잡한 코드 스트로크보다는 단일 음 리프나  
 코드톤을 따라가는 멜로디 형태로 구성되는 경우가 많다.  
 특히 John Frusciante의 Verse 연주는  
 과도한 화음을 사용하기보다는  
 한 음 한 음의 위치와 뉘앙스를 강조하는 스타일에 가깝다.

 또한 monophonic 시퀀스로 제한함으로써  
 모델이 동시에 여러 음의 조합을 학습해야 하는 부담을 줄이고  
 음의 진행과 반복 구조에 집중하도록 설계할 수 있었다.

 이는 소규모 데이터셋 환경에서  
 VAE가 보다 안정적인 latent space를 형성하는 데에도 도움이 된다.

- 토큰 정의:
  - `0` : rest
  - `1 이상` : MIDI(mp3같은 소리 데이터라고 생각하시면 됩니다.) pitch
 Verse 구간의 기타 연주는  
 모든 step마다 음이 존재하기보다는,  
 음이 울리며 지속되는 구간과  
 아무 음도 연주되지 않는 구간(rest)이 함께 나타난다.

 따라서 rest를 별도의 토큰으로 명시적으로 표현하지 않으면  
 리프의 리듬 구조를 제대로 모델링하기 어렵다.

 또한 MIDI pitch 값을 그대로 토큰으로 사용함으로써  
 추가적인 인코딩 없이  
 음높이 정보가 직접적으로 보존되며,  
 모델이 음의 상대적 위치와 반복 패턴을 학습하기 쉬워진다.

 이러한 단순한 토큰 정의는  
 소규모 데이터 환경에서  
 VAE가 안정적으로 분포를 학습하는 데에도 도움이 된다.

-데이터는 Guitar Pro / MIDI 파일에서 Verse 구간만 직접 잘라서 구성하였다.(엄청 힘들었어여 수작업이라 기타 tab보면서 잘랐습니다.)

---

## 모델 구조

### 1. Variational Autoencoder (VAE)

VAE는 기타 리프 시퀀스를 연속적인 latent 벡터로 압축한다.

- Encoder: GRU 기반
- Latent dimension: 32(당연하게도 32step이니까)
- Decoder: GRU 기반
- Loss:
  - Reconstruction loss (cross entropy)
  - KL divergence(수업시간에 배운걸로)

이 과정을 통해 이산적인 MIDI(mp3같은 소리 데이터라고 생각하시면 됩니다) 시퀀스를 연속적인 latent 공간으로 변환한다.

---

### 2. Latent Diffusion Model

Diffusion은 MIDI 시퀀스가 아니라  
**VAE latent space에서 수행**한다.
 MIDI 토큰 시퀀스는 이산(discrete)이고,  
또한 시간축(32 step) 위에서 리듬/음높이 구조가 얽혀 있다.  
이 공간에서 바로 diffusion을 수행하려면  
모델이 매우 복잡해지고 학습 안정성이 떨어질 수 있다.

반면 VAE는 입력 시퀀스를 연속적인 latent 벡터 `z`로 압축한다.  
이 latent 벡터는 다음과 같은 성질을 기대할 수 있다.

- 데이터의 공통 구조(Verse 리프의 패턴)가 더 압축되어 있음
- 연속 공간이므로 Gaussian noise를 추가하는 과정이 자연스러움
- 차원이 낮아 모델이 학습해야 할 분포가 단순해짐

따라서 본 프로젝트에서는  
`x(토큰 시퀀스) → z(latent)`로 변환한 뒤,  
`z` 공간에서 diffusion을 적용하는 방식으로 설계하였다.

- Forward process: latent 벡터에 noise 추가(점점 표준정규분포에 가깝게)
- Reverse process: MLP 기반 노이즈 예측(forward 과정에서 추가된 노이즈를 제거하여 거의 원상태로 복원)
- 장점:
  - 차원이 낮아 학습이 안정적임
  - 적은 데이터에서도 동작
  - 샘플링 속도가 빠름
 
-바로 diffusion을 사용안한 이유는 토큰 시퀀스는 이산적이고 구조가 복잡해서 학습이 불안정할 수 있어서
VAE로 연속 latent 표현을 만든 뒤 그 공간에서 diffusion을 수행했습니다.
특히 이번 프로젝트는 데이터 규모가 크지 않기 때문에
직접 시퀀스 공간에서 생성 모델을 돌리기보다는  
저차원 latent 공간에서 학습하는 것이 더 좋은 선택지같았습니다.

---

### 3. Autoregressive 디코딩 (rest 붕괴 문제와 해결)

처음에는 diffusion으로 생성한 latent `z`를 VAE decoder에 넣고,  
한 번에 32 step 전체를 디코딩한 뒤 `argmax`로 토큰을 결정하는 방식으로 구현하였다.

하지만 실제로 실행해보니 결과가 대부분 **rest 토큰(0)**으로만 채워지는 현상이 발생했다.  
즉, 생성된 MIDI가 무음이거나(노트 0개), 거의 연주가 없는 형태로 붕괴되었다.

이 현상은 소규모 데이터셋 환경에서  
decoder가 "가장 안전한 선택"인 rest 토큰으로 확률을 몰아주는 방식으로  
출력을 쉽게 수렴시키는 경우가 있다는 점과 관련이 있다고 판단하였다.  
(특히 `x_in`을 전부 rest로 고정한 채 디코딩하면 이런 문제가 더 심해질 수 있다.)

이를 해결하기 위해 다음과 같은 방법을 적용하였다.

#### 1) Autoregressive decoding (한 step씩 생성)

디코더는 32개의 토큰을 한 번에 결정하는 대신,  
이전 step에서 생성한 토큰을 다음 step의 입력으로 사용하여  
**한 step씩 순차적으로 생성**하도록 변경하였다.

- `t=0`에서 토큰을 생성
- 생성된 토큰을 `t=1`의 입력으로 사용
- 이를 반복하여 길이 32 시퀀스를 구성

이 방식은 리듬/진행 구조가 이전 토큰의 영향을 받도록 하여  
단순히 rest(0)으로만 붕괴되는 현상을 줄이는 데 도움이 되었다.

#### 2) Rest logit penalty 적용

Autoregressive만으로도 개선은 되었지만,  
여전히 rest가 과도하게 생성되는 경우가 있었다.

따라서 디코딩 시점에서 rest 토큰의 logit을 일정 값만큼 감소시키는  
**rest logit penalty**를 적용하였다.

- 목적: rest 토큰이 “너무 쉽게” 선택되는 것을 방지
- 효과: 음이 실제로 생성될 확률 증가

이때 penalty는 과도하면 불필요하게 음이 빽빽해질 수 있으므로,  
`1.0 ~ 4.0` 범위에서 실험적으로 조절하였다.

#### 3) Temperature sampling으로 다양성 확보

`argmax` 방식은 항상 가장 확률이 높은 토큰만 선택하므로  
출력이 단조롭거나 특정 패턴에 고정되는 문제가 생길 수 있다.

이를 완화하기 위해 softmax 확률 분포에서  
**temperature sampling**을 적용하였다.

- temperature를 높이면: 더 다양한 토큰이 선택됨 (다양성 증가)
- temperature를 낮추면: 더 안정적인 토큰이 선택됨 (품질 안정)

본 프로젝트에서는 `1.0 ~ 1.3` 범위에서  
“다양성 vs 안정성”의 균형을 맞추는 방식으로 사용하였다.

#### 결과

위의 세 가지 변경을 적용한 이후에는  
생성된 토큰 시퀀스에 실제 pitch 이벤트가 포함되기 시작했고,  
결과 MIDI 파일에서도 노트가 생성되어  
실제로 연주 가능한 기타 리프를 얻을 수 있었다.

또한 이 과정은 단순히 오류수정이 아  
소규모 데이터 환경에서 생성 모델이 쉽게 붕괴할 수 있다는 점을 확인할 수 있었고
이를 디코딩 단계에서 안정화하는 방법또한 알게 되었습니다.

---

## 프로젝트 구조

```text
my_riff_project_verseonly/
 ├─ midi_riffs/              # Verse-only MIDI 데이터
 ├─ config.py
 ├─ midi_utils.py
 ├─ dataset.py
 ├─ models.py
 ├─ train_vae.py
 ├─ train_diffusion.py
 ├─ generate.py
 ├─ visualize_tokens.py
 └─ check_midi.py



실행 방법
1. 라이브러리 설치
pip install torch pretty_midi numpy matplotlib

2. 데이터 준비
midi_riffs/ 폴더에
Verse 구간 2마디로 잘라낸 기타 MIDI 파일을 넣는다.

3. VAE 학습
python train_vae.py
결과: riff_vae.pth

4. Latent Diffusion 학습
python train_diffusion.py
결과: latent_diffusion.pth

5. 기타 리프 생성
python generate.py
출력: generated_riff_long.mid

6. (선택) 결과 확인
python check_midi.py
생성된 MIDI의 노트 개수와 분포를 확인할 수 있다








