# ML-Practice-Lab

《데싸노트의 실전에서 통하는 머신러닝》 책을 기반으로 
머신러닝 알고리즘을 순서대로 실습하며 정리한 레포지토리입니다.  
각 알고리즘을 코드로 직접 구현하고 특징과 활용법을 정리합니다.

---

## 학습 기간
- **2025.08.26 ~ 2025.09.16**

---

## 학습 교재
- 《데싸노트의 실전에서 통하는 머신러닝》

---

## 레포지토리 구조 (실습 순서)
```text
ML-Practice-Lab/
├── 01_LinearRegression/       # 선형 회귀 (Linear Regression)
├── 02_LogisticRegression/     # 로지스틱 회귀 (Logistic Regression)
├── 03_KNN/                    # K-최근접 이웃 (K-Nearest Neighbors)
├── 04_NaiveBayes/             # 나이브 베이즈 (Naive Bayes)
├── 05_DecisionTree/           # 결정 트리 (Decision Tree)
├── 06_RandomForest/           # 랜덤 포레스트 (Random Forest)
├── 07_XGBoost/                # XGBoost
├── 08_LightGBM/               # LightGBM
├── 09_KMeans/                 # K-평균 군집화 (K-Means Clustering)
├── 10_PCA/                    # 주성분 분석 (Principal Component Analysis)
└── README.md
```

---

## 사용 기술
- **Language**: Python 3.10+
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn, jupyter
- **Optional**: xgboost, lightgbm

---

## 학습 목표
1. 기본적인 지도학습/비지도학습 알고리즘을 직접 구현하며 개념 정리하기  
2. 각 알고리즘의 **적합한 데이터 유형과 장단점** 파악하기  
3. 교재 속 예제를 넘어서 다양한 데이터셋에 응용해보기  

---

## 알고리즘별 학습 포인트
- **선형 회귀**: 연속형 데이터 예측, 기초 회귀 모델 이해  
- **로지스틱 회귀**: 분류 문제 접근, 시그모이드 함수 이해  
- **KNN**: 거리 기반 분류, 단순하지만 강력한 모델  
- **나이브 베이즈**: 조건부 확률 기반 분류, 텍스트 분류에 강점  
- **결정 트리**: 트리 구조 기반 해석 가능성, 과적합 이슈  
- **랜덤 포레스트**: 앙상블 기법, 변수 중요도 해석  
- **XGBoost**: 강력한 부스팅 기법, 성능 최적화 경험  
- **LightGBM**: 대용량 데이터 최적화, 속도와 성능 비교  
- **K-Means**: 비지도 학습 기초, 클러스터링 응용  
- **PCA**: 차원 축소, 데이터 시각화 및 특징 추출  

---

## 향후 계획
- 교재 외부 데이터셋(Kaggle 등)을 활용한 알고리즘 확장 실습  
- 성능 비교 및 모델 선택 기준 심화 학습  
- ML → DL 확장 (딥러닝 기초, PyTorch/TensorFlow)  
- 모델 서빙 및 MLOps 학습으로 실전 적용  
