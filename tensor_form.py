import torch
from torch import tensor
import matplotlib.pyplot as plt


def Gaussian_Wave_Packet(k, sigma, miu, x):
    """
    k = tensor([])
    sigma = tensor([])
    miu = tensor([])
    x = tensor([])

    输出四阶张量
    index_0 = k
    index_1 = sigma
    index_2 = miu
    index_3 = x
    """
    k = k.view(-1, 1, 1, 1)
    sigma = sigma.view(1, -1, 1, 1)
    miu = miu.view(1, 1, -1, 1)
    x = x.view(1, 1, 1, -1)
    envelope = torch.exp(-((x - miu) ** 2) / (2 * sigma**2))
    oscillate = torch.exp(
        1j * k * (x - miu)
    )  # planner wave traveling in the positive x direction
    wavelet = envelope * oscillate
    return wavelet


ks = torch.arange(20, 200, 25)
ss = torch.arange(0.1, 1.8, 0.3)
ms = torch.arange(2, 9, 1.5)
xs = torch.linspace(0, 10, 6000)
res = Gaussian_Wave_Packet(ks, ss, ms, xs)

fig = plt.figure(num="subplots-axes", figsize=(8, 6))  # 创建画板，num是画板编号int or str
fig.text(
    x=0.0, y=0.95, s="Designed by longwarriors", style="italic", fontsize=8, color="red"
)

ax1 = plt.subplot(2, 2, 1)
for i in range(len(ms)):
    ax1.plot(xs, res[3, 5, i, :].abs())  # "格式控制字符串"="颜色"+"点型"+"线型"
# ax1.set_ylim(bottom=-5, top=5)
ax1.set_title("Centers changing")

ax2 = plt.subplot(2, 2, 2)
for i in range(len(ss)):
    ax2.plot(xs, res[3, i, 2, :].abs())
# ax2.set_ylim(bottom=-5, top=5)
ax2.set_title("$\sigma$ changing")

ax3 = plt.subplot(2, 1, 2)
for i in range(len(ks)):
    ax3.plot(xs, res[i, 3, 2, :].real)
# ax3.set_xlim(left=0, right=3)
ax3.set_title("Wave-vector changing")

plt.tight_layout()
plt.show()
