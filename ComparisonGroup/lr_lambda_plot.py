import numpy as np
import matplotlib.pyplot as plt


# progress(p)의 값 범위 설정 (0.0에서 1.0까지)
p = np.linspace(0, 1, 100)

# progress 값에 따른 학습률(LR) 계산
lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

# 그래프 그리기
plt.plot(p, lambda_p)
# LaTex 문법을 사용한 제목
plt.title(r'$umm$')
plt.xlabel('Progress (p)')
plt.ylabel('Learning Rate (LR)')
plt.grid(True)
plt.show()
# plt.savefig("my_plot.png")