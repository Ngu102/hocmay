import numpy as np

class Perceptron:
    def __init__(self, lr=0.1, n_epochs=50, random_state=42,
                 shuffle=True, verbose=False):
        self.lr = lr                        
        self.n_epochs = n_epochs            
        self.random_state = random_state    
        self.shuffle = shuffle              
        self.verbose = verbose              

    def net_input(self, X):
        # z = w·x + b
        return np.asarray(X, dtype=float) @ self.w_ + self.b_

    def predict(self, X):
        # ŷ = 1 nếu z >= 0, ngược lại 0
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def score(self, X, y):
        # tỉ lệ dự báo đúng
        return np.mean(self.predict(X) == np.asarray(y))

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)

        # Khởi tạo tham số
        rng = np.random.default_rng(self.random_state)
        self.w_ = rng.normal(0.0, 0.01, size=X.shape[1])
        self.b_ = 0.0
        self.errors_ = []                   # số lần cập nhật mỗi epoch

        for _ in range(self.n_epochs):
            n_errors = 0
            idx = rng.permutation(len(X)) if self.shuffle else np.arange(len(X))
            for xi, yi in zip(X[idx], y[idx]):
                update = self.lr * (yi - self.predict(xi))   # η(y − ŷ)
                self.w_ += update * xi                       # w ← w + η(y − ŷ)x
                self.b_ += update                            # b ← b + η(y − ŷ)
                n_errors += int(update != 0.0)
                if self.verbose:
                    print(xi, yi, update, self.w_, self.b_)

            self.errors_.append(n_errors)
            if self.verbose:
                print(f"--- epoch {len(self.errors_)}: {n_errors} lỗi ---")
            if n_errors == 0:               # không còn mẫu sai: hội tụ
                break
        return self
    
X = [[0, 0], [0, 1], [1, 0], [1, 1]]

m_and = Perceptron(lr=0.1, n_epochs=50).fit(X, [0, 0, 0, 1])
print(m_and.predict(X), m_and.errors_)   

m_xor = Perceptron(lr=0.1, n_epochs=50).fit(X, [0, 1, 1, 0])
print(m_xor.predict(X), m_xor.errors_)   
