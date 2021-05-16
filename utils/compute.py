
# 边缘检测
# alpha = lambda N: 23 * 32 * N
# beta = lambda N: 2 * 32 * N
# gamma = lambda N: 31 * N - 36

# 多边形化
alpha = lambda N: 21 * 32 * N
beta = lambda N: 0 * 32 * N
gamma = lambda N: 32 * N

div = lambda x1, x2: float('inf') if x2 == 0 else x1 / x2

b1 = lambda N: div(65536, alpha(N))
b2 = lambda N: div(98304, beta(N))
b3 = lambda N: div(2048, (32 * N))

b = lambda N: min(b1(N), b2(N), b3(N), 32)

tau = lambda N: gamma(N)*b(N)

max_value = 0
max_N = 0
for n in range(1,33):
    this_value = tau(n)
    print(n, this_value)

    if(this_value > max_value):
        max_value = this_value
        max_N = n

print("N: ", max_N)
print("regs: ", alpha(max_N))
print("shared mem: ", beta(max_N))
print("block: ", b(max_N))