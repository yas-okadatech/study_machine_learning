import scipy as sp
import matplotlib.pyplot as plt

def error(f, x, y):
  return sp.sum((f(x) - y) ** 2)

data = sp.genfromtxt('data/web_traffic.tsv', delimiter="\t")
print(data[:10])
print(data.shape)

x = data[:, 0]
y = data[:, 1]

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

# plot
plt.scatter(x, y)
plt.title('Web traffic over the last month')
plt.xlabel('Time')
plt.ylabel('Hits/hours')
plt.xticks([w*7*24 for w in range(10)], ['week %i '%w for w in range(10)])


# curve_fitting (d=1)
fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full = True)
print('Model paramters: %s' % fp1)
print(residuals)

f1 = sp.poly1d(fp1)
print(error(f1, x, y))

fx = sp.linspace(0, x[-1], 1000)
plt.plot(fx, f1(fx), linewidth = 4)

# curve_fitting (d=2)
fp2 = sp.polyfit(x, y, 2)
print(fp2)
f2 = sp.poly1d(fp2)
print(error(f2, x, y))

plt.plot(fx, f2(fx), linewidth = 4)

legends = [
    'd=%i' % f1.order,
    'd=%i' % f2.order,
]
plt.legend(legends, loc = 'upper left')

plt.autoscale(tight=True)
plt.grid()
plt.show()

