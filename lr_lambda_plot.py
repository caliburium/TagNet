import numpy as np
import matplotlib.pyplot as plt

def lr_lambda(progress):
    mu_0 = 0.01
    alpha = 10
    beta = 0.75
    # 계산된 학습률 반환
    return mu_0 * (1 + alpha * progress) ** (-beta)

# progress(p)의 값 범위 설정 (0.0에서 1.0까지)
progress = np.linspace(0, 1, 100)

# progress 값에 따른 학습률(LR) 계산
lr = lr_lambda(progress)

mu_0 = 0.01
alpha = 10
beta = 0.75

# 그래프 그리기
plt.plot(progress, lr)
# LaTex 문법을 사용한 제목
plt.title(r'$LR = \mu_0 (1 + \alpha \cdot p)^{-\beta}$' + f' ($\\mu_0={mu_0}, \\alpha={alpha}, \\beta={beta}$)')
plt.xlabel('Progress (p)')
plt.ylabel('Learning Rate (LR)')
plt.grid(True)
plt.show()