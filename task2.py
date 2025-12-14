import matplotlib
# 强制使用Agg后端（适配无GUI环境，如服务器/CI）
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from typing import Tuple, Dict

# ====================== 全局参数配置区（一键调整） ======================
# 基础配置
SEED = 42  # 随机种子
TEST_SIZE = 0.3  # 测试集比例
LR_MAX_ITER = 500  # 逻辑回归最大迭代数

# 可视化配置
FIG_SIZE = (12, 10)  # 画布尺寸
FIG_DPI = 300  # 图片分辨率
VIEW_ELEV = 20  # 3D视角仰角（PPT默认20°）
VIEW_AZIM = 45  # 3D视角方位角（PPT默认45°）
CLASS_COLORS = {0: '#FF4444', 1: '#0066CC'}  # 类别颜色（红/蓝，精准匹配PPT）
BOUNDARY_COLOR = '#ADD8E6'  # 决策超平面颜色（lightblue十六进制）
SAVE_PATH = "task2_3d_boundary.png"  # 保存路径

# 特征配置
FEATURE_INDICES = [0, 1, 2]  # 选择的3个特征索引（萼片长、萼片宽、花瓣长）
FEATURE_NAMES = ['Sepal Length', 'Sepal Width', 'Petal Length']  # 特征名

def load_and_preprocess_iris() -> Tuple[np.ndarray, np.ndarray, list]:
    """
    加载鸢尾花数据集并预处理为二分类数据
    返回：3维特征矩阵（标准化）、二分类标签、特征名
    """
    # 加载原始数据
    iris = load_iris()
    X = iris.data[:, FEATURE_INDICES]  # 取指定3个特征
    y = iris.target
    
    # 转换为二分类（0类=Setosa，1类=Versicolor+Virginica）
    y_bin = np.where(y == 0, 0, 1)
    print(f"✅ 数据集加载完成：")
    print(f"  - 样本数：{X.shape[0]}, 特征数：{X.shape[1]}")
    print(f"  - 二分类样本分布：Class 0(Setosa)={np.sum(y_bin==0)}, Class 1(Others)={np.sum(y_bin==1)}")
    
    # 数据维度校验
    if X.shape[1] != 3:
        raise ValueError(f"❌ 特征维度错误：期望3维，实际{X.shape[1]}维")
    
    # 标准化（消除量纲影响，提升模型效果）
    scaler = StandardScaler()
    X_scaled_3d = scaler.fit_transform(X)
    
    # 标准化后范围校验
    print(f"  - 标准化后特征范围：")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"    {name}: [{X_scaled_3d[:, i].min():.3f}, {X_scaled_3d[:, i].max():.3f}]")
    
    return X_scaled_3d, y_bin, FEATURE_NAMES

def train_logistic_regression(X: np.ndarray, y: np.ndarray) -> Tuple[LogisticRegression, float]:
    """
    训练逻辑回归模型并评估测试集准确率
    返回：训练好的分类器、测试集准确率
    """
    # 划分训练/测试集（分层抽样，保证类别分布）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    
    try:
        # 训练模型（显式指定求解器，避免版本兼容警告）
        clf = LogisticRegression(
            max_iter=LR_MAX_ITER,
            random_state=SEED,
            solver='lbfgs'
        )
        clf.fit(X_train, y_train)
        
        # 评估准确率
        test_acc = clf.score(X_test, y_test)
        print(f"✅ 模型训练完成：")
        print(f"  - 迭代数：{clf.n_iter_[0]}/{LR_MAX_ITER}")
        print(f"  - 测试集准确率：{test_acc:.3f}")
        
        return clf, test_acc
    except Exception as e:
        raise RuntimeError(f"❌ 模型训练失败：{str(e)}")

def calculate_3d_decision_plane(clf: LogisticRegression, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算3D决策超平面（方程：w1x1 + w2x2 + w3x3 + b = 0）
    返回：x1, x2（网格）、x3（超平面上的z值）
    """
    # 获取模型权重和偏置
    coef_weights = clf.coef_[0]  # 权重向量 [w1, w2, w3]
    intercept_bias = clf.intercept_[0]  # 偏置项 b
    
    # 打印超平面方程（便于验证）
    print(f"✅ 决策超平面方程：")
    print(f"  {coef_weights[0]:.3f}*x1 + {coef_weights[1]:.3f}*x2 + {coef_weights[2]:.3f}*x3 + {intercept_bias:.3f} = 0")
    
    # 校验权重非零（避免除以零）
    if abs(coef_weights[2]) < 1e-8:
        raise ValueError(f"❌ 第3个特征权重接近0（{coef_weights[2]:.8f}），无法求解x3")
    
    # 生成x1-x2网格
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x1, x2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 20),
        np.linspace(x2_min, x2_max, 20)
    )
    
    # 求解超平面上的x3值（x3 = -(w1x1 + w2x2 + b)/w3）
    x3 = -(coef_weights[0] * x1 + coef_weights[1] * x2 + intercept_bias) / coef_weights[2]
    
    print(f"✅ 3D决策超平面计算完成：")
    print(f"  - 网格范围：x1[{x1_min:.3f}, {x1_max:.3f}], x2[{x2_min:.3f}, {x2_max:.3f}]")
    
    return x1, x2, x3

def plot_3d_decision_boundary(X: np.ndarray, y: np.ndarray, x1: np.ndarray, x2: np.ndarray, 
                              x3: np.ndarray, test_acc: float, feature_names: list):
    """
    绘制3D决策边界图（匹配PPT样式），并保存图片
    """
    # 创建画布和3D轴
    fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. 绘制决策超平面（半透明浅蓝色，PPT同款效果）
    ax.plot_surface(
        x1, x2, x3,
        color=BOUNDARY_COLOR,
        alpha=0.5,
        edgecolor='none',
        shade=False
    )
    
    # 2. 绘制原始数据点（区分两类）
    mask_0 = y == 0  # Class 0 (Setosa)
    mask_1 = y == 1  # Class 1 (Others)
    
    # Class 0 数据点（红色）
    ax.scatter(
        X[mask_0, 0], X[mask_0, 1], X[mask_0, 2],
        c=CLASS_COLORS[0], s=80, edgecolors='black', linewidth=1,
        label='Setosa (Class 0)', zorder=5, alpha=0.9
    )
    
    # Class 1 数据点（蓝色）
    ax.scatter(
        X[mask_1, 0], X[mask_1, 1], X[mask_1, 2],
        c=CLASS_COLORS[1], s=80, edgecolors='black', linewidth=1,
        label='Others (Class 1)', zorder=5, alpha=0.9
    )
    
    # 3. 样式调整（精准匹配PPT）
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)  # PPT默认视角
    ax.set_xlabel(feature_names[0], fontsize=12, labelpad=10)
    ax.set_ylabel(feature_names[1], fontsize=12, labelpad=10)
    ax.set_zlabel(feature_names[2], fontsize=12, labelpad=10)
    ax.set_title(
        f'Task 2: 3D Decision Boundary (Test Acc: {test_acc:.3f})',
        fontsize=16, pad=20
    )
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3)  # 网格半透明，不遮挡内容
    
    # 4. 保存图片（tight布局避免裁剪）
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=FIG_DPI, bbox_inches='tight')
    print(f"✅ 可视化图片已保存至：{SAVE_PATH}")
    
    # 释放画布资源
    plt.close(fig)

def main():
    """主函数：串联所有流程"""
    try:
        print("="*60)
        print("🚀 开始执行Task 2：3D决策边界绘制")
        print("="*60)
        
        # 1. 加载并预处理数据
        X_scaled_3d, y_bin, feature_names = load_and_preprocess_iris()
        
        # 2. 训练逻辑回归模型
        clf, test_acc = train_logistic_regression(X_scaled_3d, y_bin)
        
        # 3. 计算3D决策超平面
        x1, x2, x3 = calculate_3d_decision_plane(clf, X_scaled_3d)
        
        # 4. 绘制并保存3D决策边界图
        plot_3d_decision_boundary(X_scaled_3d, y_bin, x1, x2, x3, test_acc, feature_names)
        
        print("\n🎉 任务二完成！")
        print(f"📋 二分类准确率：{test_acc:.3f}（Setosa与其他两类线性可分）")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 程序执行失败：{str(e)}")
        raise  # 抛出异常便于调试

if __name__ == "__main__":
    main()