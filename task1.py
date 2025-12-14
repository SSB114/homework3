import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures, SplineTransformer

# 1. 加载数据（三分类+两个特征）
iris = load_iris()
X = iris.data[:, [2, 3]]  # 花瓣长度、花瓣宽度
y = iris.target
feature_names = ['Petal Length', 'Petal Width']
target_names = iris.target_names
class_colors = ['yellow', 'green', 'blue'] 
cmap = mcolors.ListedColormap(class_colors)

# 2. 数据集划分（分层抽样，避免类别不平衡）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. 配置指定的7个分类器
classifiers = {
    "Logistic Regression\n(C=0.01)": LogisticRegression(C=0.01, max_iter=500),
    "Logistic Regression\n(C=1)": LogisticRegression(C=1, max_iter=500),
    "Gaussian Process": GaussianProcessClassifier(kernel=1.0 * RBF([1.0, 1.0])),
    "Logistic Regression\n(RBF features)": make_pipeline(
        Nystroem(kernel="rbf", gamma=0.5, n_components=50, random_state=1),
        LogisticRegression(C=10, max_iter=500)
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(),
    "Logistic Regression\n(binned features)": make_pipeline(
        KBinsDiscretizer(n_bins=5, quantile_method="averaged_inverted_cdf"),
        PolynomialFeatures(interaction_only=True),
        LogisticRegression(C=10, max_iter=500)
    ),
    "Logistic Regression\n(spline features)": make_pipeline(
        SplineTransformer(n_knots=5),
        PolynomialFeatures(interaction_only=True),
        LogisticRegression(C=10, max_iter=500)
    )
}

# 4. 生成网格（用于绘制决策边界和概率图）
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

# 5. 可视化布局（7个分类器，每个分类器1行4列子图：整体决策边界+3个类别概率）
fig, axes = plt.subplots(7, 4, figsize=(20, 35), dpi=300)
fig.suptitle('Task 1: Multi-Classifier Results (3-Class, 2 Features)', fontsize=20, y=0.98)

for idx, (name, clf) in enumerate(classifiers.items()):
    # 训练模型
    clf.fit(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    
    # 预测网格类别和概率
    Z = clf.predict(grid).reshape(xx.shape)
    probs = clf.predict_proba(grid).reshape(xx.shape[0], xx.shape[1], 3)
    
    # 子图1：整体决策边界
    ax1 = axes[idx, 0]
    ax1.contourf(xx, yy, Z, alpha=0.6, cmap=cmap)
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='black', s=50)
    ax1.set_title(f"{name}\nTest Acc: {test_acc:.3f}", fontsize=12)
    ax1.set_xlabel(feature_names[0], fontsize=10)
    ax1.set_ylabel(feature_names[1], fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 子图2-4：每个类别的概率图
    for class_idx in range(3):
        ax = axes[idx, class_idx + 1]
        class_prob = probs[:, :, class_idx]
        # 生成对应类别的渐变色板
        custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            f'class_{class_idx}', ['white', class_colors[class_idx]], N=256
        )
        contour = ax.contourf(xx, yy, class_prob, alpha=0.7, cmap=custom_cmap)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='black', s=30, alpha=0.8)
        fig.colorbar(contour, ax=ax, shrink=0.6)
        ax.set_title(f'Class {class_idx} Probability', fontsize=11)
        ax.set_xlabel(feature_names[0], fontsize=10)
        ax.set_ylabel(feature_names[1], fontsize=10)
        ax.grid(alpha=0.3)

# 调整布局，避免重叠
plt.tight_layout()
plt.subplots_adjust(top=0.96)
# 保存图片
plt.show()
plt.savefig("task1_classifier_results.png", dpi=300, bbox_inches='tight')
plt.close()

print("任务一完成！已保存多分类器可视化结果：task1_classifier_results.png")
print(f"共测试7个分类器，涵盖逻辑回归（不同特征处理）、高斯过程、梯度提升")