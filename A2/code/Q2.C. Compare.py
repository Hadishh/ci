#write your code here
#Q2C_graded
y = model_MLP.predict(x_train)
plt.plot(x_train, y)
plt.title("MLP Model Sine Function")
plt.show()

y = [compute_f(x, mu, w, sigma) for x in x_train]
plt.plot(x_train, y)
plt.title("RBF Model Sine Function")
plt.show()

