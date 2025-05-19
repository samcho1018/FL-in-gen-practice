
import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal

# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gaussian 분포 파라미터
mu1, sigma1 = -2.0, 1.0
mu2, sigma2 = 2.0, 1.5

# x 값 (공통)
x = torch.linspace(-10, 10, 500, device=device)

# p1, p2 선언
p1 = Normal(loc=mu1, scale=sigma1)
p2 = Normal(loc=mu2, scale=sigma2)

# 1. LHS: ∇log p1(x) + ∇log p2(x)
x1 = x.clone().detach().requires_grad_(True)

logp1 = p1.log_prob(x1)
logp2 = p2.log_prob(x1)

grad_logp1 = torch.autograd.grad(logp1.sum(), x1, create_graph=False)[0]
grad_logp2 = torch.autograd.grad(logp2.sum(), x1, create_graph=False)[0]
lhs = grad_logp1 + grad_logp2

# 2. RHS: ∇log(p1(x) + p2(x))
x2 = x.clone().detach().requires_grad_(True)

pdf1 = torch.exp(p1.log_prob(x2))
pdf2 = torch.exp(p2.log_prob(x2))

p_total = pdf1 + pdf2
log_p_total = torch.log(p_total)

rhs = torch.autograd.grad(log_p_total.sum(), x2, create_graph=False)[0]

# 시각화1
x_np = x.detach().cpu().numpy()
lhs_np = lhs.detach().cpu().numpy()
rhs_np = rhs.detach().cpu().numpy()
grad_logp1_np = grad_logp1.detach().cpu().numpy()
grad_logp2_np = grad_logp2.detach().cpu().numpy()

plt.figure(figsize=(10, 5))
plt.plot(x_np, lhs_np, label="∇log p1(x) + ∇log p2(x)")
plt.plot(x_np, rhs_np, label="∇log(p1(x) + p2(x))", linestyle="dashed", color="orange")
plt.plot(x_np, grad_logp1_np, label="∇log p1(x)", color = "red")
plt.plot(x_np, grad_logp2_np, label="∇log p2(x)", color = "black")
plt.title("Score Function Comparison 1")
plt.xlabel("x")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()

# 시각화2
x_np = x.detach().cpu().numpy()
lhs_np = lhs.detach().cpu().numpy()
rhs_np = rhs.detach().cpu().numpy()
grad_logp1_np = grad_logp1.detach().cpu().numpy()
grad_logp2_np = grad_logp2.detach().cpu().numpy()

plt.figure(figsize=(10, 5))
plt.plot(x_np, lhs_np, label="∇log p1(x) + ∇log p2(x)", linestyle="dashed")
#plt.plot(x_np, rhs_np, label="∇log(p1(x) + p2(x))", linestyle="dashed", color = "orange")
plt.plot(x_np, grad_logp1_np, label="∇log p1(x)", color = "red")
plt.plot(x_np, grad_logp2_np, label="∇log p2(x)", color = "black")
plt.title("Score Function Comparison 2")
plt.xlabel("x")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()

# 시각화3
x_np = x.detach().cpu().numpy()
lhs_np = lhs.detach().cpu().numpy()
rhs_np = rhs.detach().cpu().numpy()
grad_logp1_np = grad_logp1.detach().cpu().numpy()
grad_logp2_np = grad_logp2.detach().cpu().numpy()

plt.figure(figsize=(10, 5))
#plt.plot(x_np, lhs_np, label="∇log p1(x) + ∇log p2(x)")
plt.plot(x_np, rhs_np, label="∇log(p1(x) + p2(x))", linestyle="dashed", color = "orange")
plt.plot(x_np, grad_logp1_np, label="∇log p1(x)", color = "red")
plt.plot(x_np, grad_logp2_np, label="∇log p2(x)", color = "black")
plt.title("Score Function Comparison 3")
plt.xlabel("x")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()

