import matplotlib
# 强制使用Agg后端（适配无GUI环境，如服务器/CI）
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from typing import Tuple, Dict

# ====================== 全局参数配置区（一键调整） ======================
# 基础配置
SEED = 42  # 随机种子
LR_MAX_ITER = 500  # 逻辑回归最大迭代数
TEST_CLASS_LABEL = 1  # 可视化的目标类别（Class 1）

# 可视化配置
FIG_SIZE = (14, 12)  # 画布尺寸
FIG_DPI = 300  # 图片分辨率
GRID_DENSITY = 15  # 3D网格密度（15×15×15）
VIEW_ELEV = 25  # 3D视角仰角
VIEW_AZIM = 50  # 3D视角方位角
CLASS_COLORS = {0: '#FF4444', 1: '#0066CC'}  # 类别颜色（红/蓝，匹配PPT）
CMAP = 'coolwarm'  # 概率颜色映射（PPT常用冷暖色）
SAVE_PATH = "task3_3d_probability_map.png"  # 保存路径

# 特征配置
FEATURE_INDICES = [0, 1, 2]  # 选择的3个特征索引
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
    
    # 转换为二分类（0类为Setosa，其余为1类）
    y_bin = np.where(y == 0, 0, 1)
    print(f"✅ 数据集加载完成：")
    print(f"  - 样本数：{X.shape[0]}, 特征数：{X.shape[1]}")
    print(f"  - 二分类样本分布：Class 0={np.sum(y_bin==0)}, Class 1={np.sum(y_bin==1)}")
    
    # 标准化（消除量纲影响）
    scaler = StandardScaler()
    X_scaled_3d = scaler.fit_transform(X)
    
    # 数据校验
    if X_scaled_3d.shape[1] != 3:
        raise ValueError(f"❌ 特征维度错误：期望3维，实际{X_scaled_3d.shape[1]}维")
    
    return X_scaled_3d, y_bin, FEATURE_NAMES

def train_logistic_regression(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """训练逻辑回归模型，返回训练好的分类器"""
    try:
        clf = LogisticRegression(
            max_iter=LR_MAX_ITER,
            random_state=SEED,
            solver='lbfgs'  # 显式指定求解器，避免默认值警告
        )
        clf.fit(X, y)
        print(f"✅ 模型训练完成：迭代数={clf.n_iter_[0]}/{LR_MAX_ITER}")
        return clf
    except Exception as e:
        raise RuntimeError(f"❌ 模型训练失败：{str(e)}")

def generate_3d_grid(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成3D网格用于概率预测
    返回：xx, yy, zz（网格坐标）、grid_points（展平的网格点）
    """
    # 计算每个特征的网格范围（扩展1单位，覆盖更多区域）
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x3_min, x3_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    
    # 生成等间距网格
    x1 = np.linspace(x1_min, x1_max, GRID_DENSITY)
    x2 = np.linspace(x2_min, x2_max, GRID_DENSITY)
    x3 = np.linspace(x3_min, x3_max, GRID_DENSITY)
    xx, yy, zz = np.meshgrid(x1, x2, x3)
    
    # 展平网格点用于模型预测
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    print(f"✅ 3D网格生成完成：")
    print(f"  - 网格维度：{GRID_DENSITY}×{GRID_DENSITY}×{GRID_DENSITY}")
    print(f"  - 预测点总数：{grid_points.shape[0]}")
    
    return xx, yy, zz, grid_points

def predict_grid_probabilities(clf: LogisticRegression, grid_points: np.ndarray, 
                               xx: np.ndarray) -> np.ndarray:
    """预测网格点的概率，返回Class 1的概率矩阵（形状与网格一致）"""
    try:
        # 预测每个网格点的概率
        probs = clf.predict_proba(grid_points)
        # 提取目标类（Class 1）的概率并重塑为网格形状
        class1_probs = probs[:, TEST_CLASS_LABEL].reshape(xx.shape)
        print(f"✅ 概率预测完成：Class {TEST_CLASS_LABEL}概率范围 [{class1_probs.min():.3f}, {class1_probs.max():.3f}]")
        return class1_probs
    except NotFittedError:
        raise RuntimeError("❌ 模型未训练，无法预测概率")
    except Exception as e:
        raise RuntimeError(f"❌ 概率预测失败：{str(e)}")

def plot_3d_probability_map(X: np.ndarray, y: np.ndarray, xx: np.ndarray, yy: np.ndarray, 
                           zz: np.ndarray, class1_probs: np.ndarray, feature_names: list):
    """绘制3D概率图（曲面+等高线+数据点），完全匹配PPT样式"""
    # 创建画布和3D轴
    fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. 绘制3D概率曲面（固定x3中间层，概率值缩放为高度）
    mid_layer = GRID_DENSITY // 2  # 取x3的中间层
    prob_surface = class1_probs[:, :, mid_layer]
    # 归一化概率值到[0,1]（避免颜色映射警告）
    prob_normalized = (prob_surface - prob_surface.min()) / (prob_surface.max() - prob_surface.min())
    
    surf = ax.plot_surface(
        xx[:, :, mid_layer], yy[:, :, mid_layer], prob_surface * 5,
        facecolors=plt.cm.get_cmap(CMAP)(prob_normalized),
        alpha=0.8, edgecolor='none', shade=False
    )
    
    # 2. 绘制底面等高线（补充2D视角）
    ax.contourf(
        xx[:, :, 0], yy[:, :, 0], class1_probs[:, :, 0],
        zdir='z', offset=X[:, 2].min() - 2,  # 等高线置于Z轴下方
        cmap=CMAP, alpha=0.5, levels=10
    )
    
    # 3. 叠加原始数据点（区分两类）
    mask_0 = y == 0
    mask_1 = y == 1
    # Class 0（Setosa）
    ax.scatter(
        X[mask_0, 0], X[mask_0, 1], X[mask_0, 2],
        c=CLASS_COLORS[0], s=80, edgecolors='black', linewidth=1,
        label='Setosa (Class 0)', zorder=10, alpha=0.9
    )
    # Class 1（Others）
    ax.scatter(
        X[mask_1, 0], X[mask_1, 1], X[mask_1, 2],
        c=CLASS_COLORS[1], s=80, edgecolors='black', linewidth=1,
        label='Others (Class 1)', zorder=10, alpha=0.9
    )
    
    # 4. 样式调整（匹配PPT风格）
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)  # 优化视角
    ax.set_xlabel(feature_names[0], fontsize=12, labelpad=10)
    ax.set_ylabel(feature_names[1], fontsize=12, labelpad=10)
    ax.set_zlabel('Probability (Scaled)', fontsize=12, labelpad=10)
    ax.set_title('Task 3: 3D Probability Map (Class 1 Probability)', fontsize=16, pad=20)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # 5. 添加归一化的概率颜色条（修复原代码警告）
    norm = plt.Normalize(class1_probs.min(), class1_probs.max())
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])  # 必须设置空数组，避免警告
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label(f'Probability of Class {TEST_CLASS_LABEL} (Others)', fontsize=11)
    
    # 6. 保存图片（tight布局避免裁剪）
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=FIG_DPI, bbox_inches='tight')
    print(f"✅ 可视化图片已保存至：{SAVE_PATH}")
    
    # 释放画布资源
    plt.close(fig)

def main():
    """主函数：串联所有流程"""
    try:
        print("="*60)
        print("🚀 开始执行Task 3：3D概率图绘制")
        print("="*60)
        
        # 1. 加载并预处理数据
        X_scaled_3d, y_bin, feature_names = load_and_preprocess_iris()
        
        # 2. 训练逻辑回归模型
        clf = train_logistic_regression(X_scaled_3d, y_bin)
        
        # 3. 生成3D网格
        xx, yy, zz, grid_points = generate_3d_grid(X_scaled_3d)
        
        # 4. 预测网格点概率
        class1_probs = predict_grid_probabilities(clf, grid_points, xx)
        
        # 5. 绘制并保存3D概率图
        plot_3d_probability_map(X_scaled_3d, y_bin, xx, yy, zz, class1_probs, feature_names)
        
        print("\n🎉 任务三完成！")
        print("📋 可视化包含：3D概率曲面 + 底面等高线 + 原始数据点，完全匹配PPT样式")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 程序执行失败：{str(e)}")
        raise  # 抛出异常便于调试

if __name__ == "__main__":
    main()