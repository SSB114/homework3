import matplotlib
# 强制使用无界面后端（适配服务器/无GUI环境）
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from typing import Tuple, Dict

# ====================== 全局参数配置区（一键调整） ======================
# 基础配置
SEED = 42  # 随机种子
TEST_SIZE = 0.3  # 测试集比例
LR_MAX_ITER = 200  # 逻辑回归最大迭代数
GRID_STEP = 0.1  # 网格步长（越小越精细，速度越慢）

# 可视化配置
FIG_SIZE = (20, 5)  # 画布尺寸（1行4列）
FIG_DPI = 300  # 图片分辨率
CLASS_COLORS = ['yellow', 'green', 'blue']  # 三类鸢尾花配色（黄/绿/蓝）
SAVE_PATH = "iris_classifier_result.png"  # 保存路径
TITLE_FONT_SIZE = 14  # 子图标题字体大小
LABEL_FONT_SIZE = 12  # 坐标轴标签字体大小

# 特征配置
FEATURE_IDX = [2, 3]  # 选择后两个特征（Petal Length, Petal Width）
FEATURE_NAMES = ['Petal Length', 'Petal Width']  # 特征名（修复原代码Sepal Width错误）

def load_iris_data() -> Tuple[np.ndarray, np.ndarray, list]:
    """
    加载鸢尾花数据集，选择指定特征
    返回：特征矩阵（后两个特征）、标签、类别名称
    """
    iris = load_iris()
    X = iris.data[:, FEATURE_IDX]  # 选择后两个特征
    y = iris.target
    target_names = iris.target_names
    
    # 数据校验
    if X.shape[1] != 2:
        raise ValueError(f"❌ 特征维度错误：期望2维，实际{X.shape[1]}维")
    
    print(f"✅ 数据集加载完成：")
    print(f"  - 样本数：{X.shape[0]}, 特征数：{X.shape[1]}")
    print(f"  - 类别数：{len(np.unique(y))}（{target_names.tolist()}）")
    print(f"  - 特征范围：")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"    {name}: [{X[:, i].min():.2f}, {X[:, i].max():.2f}]")
    
    return X, y, target_names

def train_logistic_regression(X: np.ndarray, y: np.ndarray) -> Tuple[LogisticRegression, float]:
    """
    训练逻辑回归模型，评估测试集准确率
    返回：训练好的模型、测试集准确率
    """
    # 划分训练/测试集（分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    
    try:
        # 训练模型
        model = LogisticRegression(max_iter=LR_MAX_ITER, random_state=SEED)
        model.fit(X_train, y_train)
        
        # 评估准确率
        test_acc = model.score(X_test, y_test)
        print(f"\n✅ 模型训练完成：")
        print(f"  - 迭代数：{model.n_iter_[0]}/{LR_MAX_ITER}")
        print(f"  - 测试集准确率：{test_acc:.3f}")
        
        return model, test_acc
    except Exception as e:
        raise RuntimeError(f"❌ 模型训练失败：{str(e)}")

def generate_grid(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成2D网格用于决策边界和概率预测
    返回：xx, yy（网格坐标）、grid_points（展平的网格点）
    """
    # 计算网格范围（扩展1单位，覆盖更多区域）
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # 生成网格
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, GRID_STEP),
        np.arange(y_min, y_max, GRID_STEP)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    print(f"\n✅ 网格生成完成：")
    print(f"  - 网格尺寸：{xx.shape[0]}×{xx.shape[1]}")
    print(f"  - 预测点总数：{grid_points.shape[0]}")
    
    return xx, yy, grid_points

def predict_grid_results(model: LogisticRegression, grid_points: np.ndarray, xx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    预测网格点的类别和概率
    返回：预测标签矩阵、类别概率矩阵
    """
    try:
        # 预测类别（决策边界）
        pred_labels = model.predict(grid_points).reshape(xx.shape)
        
        # 预测概率（每个类别）
        class_probs = model.predict_proba(grid_points)
        # 重塑为 (height, width, classes)
        class_probs = class_probs.reshape(xx.shape[0], xx.shape[1], -1)
        
        # 概率维度校验
        if class_probs.shape[-1] != len(CLASS_COLORS):
            raise ValueError(f"❌ 概率维度错误：期望{len(CLASS_COLORS)}类，实际{class_probs.shape[-1]}类")
        
        print(f"✅ 网格预测完成：")
        print(f"  - 预测类别范围：[{pred_labels.min()}, {pred_labels.max()}]")
        print(f"  - 概率范围：[{class_probs.min():.3f}, {class_probs.max():.3f}]")
        
        return pred_labels, class_probs
    except NotFittedError:
        raise RuntimeError("❌ 模型未训练，无法预测")
    except Exception as e:
        raise RuntimeError(f"❌ 网格预测失败：{str(e)}")

def plot_classifier_results(X: np.ndarray, y: np.ndarray, xx: np.ndarray, yy: np.ndarray,
                           pred_labels: np.ndarray, class_probs: np.ndarray):
    """
    绘制1×4子图：整体决策边界 + 3个类别的概率图
    """
    # 创建画布（1行4列）
    fig, axs = plt.subplots(1, 4, figsize=FIG_SIZE, dpi=FIG_DPI)
    
    # ====================== 子图1：整体决策边界 ======================
    ax0 = axs[0]
    # 绘制决策区域
    ax0.imshow(
        pred_labels,
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        origin='lower',
        cmap=mcolors.ListedColormap(CLASS_COLORS),
        alpha=0.6
    )
    # 绘制数据点
    ax0.scatter(
        X[:, 0], X[:, 1],
        c=y, edgecolors='k', marker='o', s=50,
        cmap=mcolors.ListedColormap(CLASS_COLORS),
        alpha=1
    )
    # 设置样式
    ax0.set_title('Overall Decision Boundaries', fontsize=TITLE_FONT_SIZE)
    ax0.set_xlabel(FEATURE_NAMES[0], fontsize=LABEL_FONT_SIZE)
    ax0.set_ylabel(FEATURE_NAMES[1], fontsize=LABEL_FONT_SIZE)
    ax0.grid(alpha=0.3)
    
    # ====================== 子图2-4：每个类别的概率图 ======================
    for i in range(len(CLASS_COLORS)):
        ax = axs[i+1]
        class_prob = class_probs[:, :, i]
        
        # 创建专属渐变色映射（白色→类别色）
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f'class_{i}_cmap', ['white', CLASS_COLORS[i]], N=256
        )
        
        # 绘制概率填充图
        contour = ax.contourf(xx, yy, class_prob, alpha=0.7, cmap=cmap, levels=20)
        
        # 绘制数据点
        ax.scatter(
            X[:, 0], X[:, 1],
            c=y, edgecolors='k', marker='o', s=50,
            cmap=mcolors.ListedColormap(CLASS_COLORS),
            alpha=1
        )
        
        # 添加颜色条（带标签）
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label(f'Probability (Class {i})', fontsize=LABEL_FONT_SIZE-1)
        
        # 设置样式
        ax.set_title(f'Class {i} Probability', fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel(FEATURE_NAMES[0], fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel(FEATURE_NAMES[1], fontsize=LABEL_FONT_SIZE)
        ax.grid(alpha=0.3)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=FIG_DPI, bbox_inches='tight')
    print(f"\n✅ 可视化图片已保存至：{SAVE_PATH}")
    
    # 释放画布资源
    plt.close(fig)

def main():
    """主函数：串联所有流程"""
    try:
        print("="*60)
        print("🚀 开始执行鸢尾花分类决策边界可视化任务")
        print("="*60)
        
        # 1. 加载数据（选择后两个特征）
        X, y, target_names = load_iris_data()
        
        # 2. 训练逻辑回归模型
        model, test_acc = train_logistic_regression(X, y)
        
        # 3. 生成预测网格
        xx, yy, grid_points = generate_grid(X)
        
        # 4. 预测网格点的类别和概率
        pred_labels, class_probs = predict_grid_results(model, grid_points, xx)
        
        # 5. 绘制并保存可视化结果
        plot_classifier_results(X, y, xx, yy, pred_labels, class_probs)
        
        print("\n🎉 任务完成！")
        print(f"📋 模型测试集准确率：{test_acc:.3f} | 可视化包含：决策边界 + 3个类别概率图")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 程序执行失败：{str(e)}")
        raise  # 抛出异常便于调试

if __name__ == "__main__":
    main()