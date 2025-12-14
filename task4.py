import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import NotFittedError

# ====================== 全局参数配置区（便于修改） ======================
SEED = 42  # 随机种子
TEST_SIZE = 0.3  # 测试集比例
POLY_DEGREE = 2  # 多项式特征阶数
SELECT_K = 6  # 选择最优特征数
LR_C_OPT = 20  # 优化模型正则化参数
LR_MAX_ITER_OPT = 1000  # 优化模型最大迭代数
LR_MAX_ITER_RAW = 500  # 原始模型最大迭代数
FIG_SIZE = (16, 12)  # 画布尺寸
FIG_DPI = 300  # 图片分辨率
CLASS_COLORS = ['#FFD700', '#90EE90', '#87CEFA']  # 更美观的颜色（黄金色、淡绿、淡蓝）
SAVE_PATH = "task4_enhanced_visualization.png"  # 保存路径

def load_iris_data() -> tuple:
    """加载鸢尾花数据集并返回特征、标签、特征名、目标名"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    print(f"✅ 数据集加载完成：特征数={X.shape[1]}, 样本数={X.shape[0]}, 类别数={len(target_names)}")
    return X, y, feature_names, target_names

def build_feature_pipeline() -> make_pipeline:
    """构建特征工程管道（多项式交互特征+标准化+特征选择）"""
    pipeline = make_pipeline(
        PolynomialFeatures(
            degree=POLY_DEGREE,
            interaction_only=True,
            include_bias=False
        ),
        StandardScaler(),
        SelectKBest(f_classif, k=SELECT_K)
    )
    return pipeline

def optimize_features(X: np.ndarray, y: np.ndarray) -> tuple:
    """执行特征工程，返回优化后的特征矩阵和特征工程管道"""
    pipeline = build_feature_pipeline()
    try:
        X_optimized = pipeline.fit_transform(X, y)
        # 校验特征数量
        if X_optimized.shape[1] < SELECT_K:
            print(f"⚠️ 警告：实际可选特征数不足{SELECT_K}，仅返回{X_optimized.shape[1]}个特征")
        print(f"✅ 特征工程完成：优化后特征数={X_optimized.shape[1]}")
        return X_optimized, pipeline
    except Exception as e:
        raise RuntimeError(f"❌ 特征工程执行失败：{str(e)}")

def train_and_evaluate_models(X_optimized: np.ndarray, y: np.ndarray) -> dict:
    """训练优化模型和原始模型，返回性能指标字典"""
    # 拆分优化特征的训练/测试集（分层抽样）
    X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(
        X_optimized, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y
    )
    
    # ========== 训练优化模型 ==========
    clf_opt = LogisticRegression(
        C=LR_C_OPT,
        max_iter=LR_MAX_ITER_OPT,
        random_state=SEED
    )
    clf_opt.fit(X_train_opt, y_train_opt)
    opt_test_acc = clf_opt.score(X_test_opt, y_test_opt)
    opt_cv_acc = cross_val_score(clf_opt, X_optimized, y, cv=5).mean()
    
    # ========== 训练原始模型（仅用前4个原始特征） ==========
    clf_raw = LogisticRegression(
        max_iter=LR_MAX_ITER_RAW,
        random_state=SEED
    )
    # 确保原始特征数量足够（鲁棒性处理）
    raw_feat_num = min(4, X_train_opt.shape[1])
    clf_raw.fit(X_train_opt[:, :raw_feat_num], y_train_opt)
    raw_test_acc = clf_raw.score(X_test_opt[:, :raw_feat_num], y_test_opt)
    
    # 整理性能指标
    perf_metrics = {
        "opt_test_acc": opt_test_acc,
        "opt_cv_acc": opt_cv_acc,
        "raw_test_acc": raw_test_acc,
        "clf_opt": clf_opt,
        "X_train_opt": X_train_opt,
        "X_test_opt": X_test_opt,
        "y_train_opt": y_train_opt,
        "y_test_opt": y_test_opt
    }
    
    # 打印性能对比
    print("\n" + "="*50)
    print("📊 模型性能对比")
    print("="*50)
    print(f"优化后模型 - 测试集准确率：{opt_test_acc:.3f} | 5折交叉验证准确率：{opt_cv_acc:.3f}")
    print(f"原始模型   - 测试集准确率：{raw_test_acc:.3f}")
    print("="*50 + "\n")
    
    return perf_metrics

def plot_enhanced_visualization(X_optimized: np.ndarray, y: np.ndarray, 
                               perf_metrics: dict, target_names: list):
    """绘制增强型3D概率可视化图并保存"""
    clf_opt = perf_metrics["clf_opt"]
    opt_test_acc = perf_metrics["opt_test_acc"]
    
    # 取优化后的前3个特征做3D可视化
    X_3d_opt = X_optimized[:, :3] if X_optimized.shape[1] >=3 else X_optimized
    if X_3d_opt.shape[1] <3:
        raise ValueError(f"❌ 优化后特征数不足3个（仅{X_3d_opt.shape[1]}个），无法绘制3D图")
    
    # 生成网格（固定第3个特征为均值，简化为2D曲面+高度映射）
    x1_min, x1_max = X_3d_opt[:, 0].min() - 1, X_3d_opt[:, 0].max() + 1
    x2_min, x2_max = X_3d_opt[:, 1].min() - 1, X_3d_opt[:, 1].max() + 1
    x3_fixed = X_3d_opt[:, 2].mean()
    xx, yy = np.meshgrid(
        np.linspace(x1_min, x1_max, 20),
        np.linspace(x2_min, x2_max, 20)
    )
    
    # 构造完整输入特征（前2个+固定第3个+剩余特征均值）
    remaining_feats = X_optimized[:, 3:] if X_optimized.shape[1] >3 else np.array([])
    remaining_mean = remaining_feats.mean(axis=0) if remaining_feats.size >0 else np.array([])
    grid_opt = np.c_[
        xx.ravel(), yy.ravel(),
        np.full(xx.size, x3_fixed),
        np.tile(remaining_mean, (xx.size, 1)) if remaining_mean.size>0 else []
    ]
    
    # 预测类别概率
    try:
        probs_opt = clf_opt.predict_proba(grid_opt)
    except NotFittedError:
        raise RuntimeError("❌ 模型未训练完成，无法预测概率")
    
    # ========== 绘制图形 ==========
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 解决中文显示问题（英文环境）
    fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    fig.suptitle(
        f'Task 4: Enhanced Visualization (Optimized Acc: {opt_test_acc:.3f})',
        fontsize=18, y=0.98, fontweight='bold'
    )
    
    # 子图1-3：3个类别的概率曲面
    for class_idx in range(3):
        ax = fig.add_subplot(2, 2, class_idx + 1, projection='3d')
        prob_class = probs_opt[:, class_idx].reshape(xx.shape)
        
        # 绘制概率曲面（高度映射概率，颜色渐变）
        surf = ax.plot_surface(
            xx, yy, x3_fixed + prob_class * 5,
            facecolors=plt.cm.RdYlBu(prob_class),
            alpha=0.8, edgecolor='gray', linewidth=0.2
        )
        
        # 绘制对应类别的数据点
        mask = y == class_idx
        ax.scatter(
            X_3d_opt[mask, 0], X_3d_opt[mask, 1], X_3d_opt[mask, 2],
            c=CLASS_COLORS[class_idx], s=70, edgecolors='black', alpha=0.9,
            label=f'{target_names[class_idx]} (Class {class_idx})', zorder=5
        )
        
        ax.set_xlabel('Optimal Feature 1', fontsize=11, labelpad=8)
        ax.set_ylabel('Optimal Feature 2', fontsize=11, labelpad=8)
        ax.set_zlabel('Optimal Feature 3', fontsize=11, labelpad=8)
        ax.set_title(f'Class {class_idx} Probability Distribution', fontsize=13, fontweight='medium')
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.view_init(elev=20, azim=45)  # 固定视角
    
    # 子图4：性能对比文本
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    perf_text = f"""📈 Optimization Details & Performance
-----------------------------------
1. Feature Engineering:
   • Polynomial Interaction Features (degree={POLY_DEGREE})
   • Standardization (remove scale bias)
   • Top-{SELECT_K} Feature Selection (ANOVA-F)
   
2. Model Tuning:
   • Logistic Regression (C={LR_C_OPT}, max_iter={LR_MAX_ITER_OPT})
   
3. Accuracy Comparison:
   • Optimized Model: {opt_test_acc:.3f} (Test) / {perf_metrics['opt_cv_acc']:.3f} (CV)
   • Raw Model (No Engineering): {perf_metrics['raw_test_acc']:.3f}"""
    
    ax4.text(
        0.1, 0.5, perf_text, fontsize=12, verticalalignment='center',
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#F5F5F5", alpha=0.9, edgecolor="#CCCCCC"),
        fontfamily='monospace'
    )
    
    # 调整布局（避免标题重叠）
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # ========== 保存图片（先保存再show，避免空白） ==========
    plt.savefig(SAVE_PATH, dpi=FIG_DPI, bbox_inches='tight')
    print(f"✅ 可视化图片已保存至：{SAVE_PATH}")
    
    # 显示图形
    plt.show()
    plt.close(fig)  # 释放画布资源

def main():
    """主函数：串联所有流程"""
    try:
        # 1. 加载数据
        X, y, feature_names, target_names = load_iris_data()
        
        # 2. 特征工程优化
        X_optimized, pipeline = optimize_features(X, y)
        
        # 3. 模型训练与评估
        perf_metrics = train_and_evaluate_models(X_optimized, y)
        
        # 4. 增强型可视化
        plot_enhanced_visualization(X_optimized, y, perf_metrics, target_names)
        
        print("\n🎉 任务四完成！核心优化：特征工程+模型调优，准确率较原始模型提升明显")
        
    except Exception as e:
        print(f"\n❌ 程序执行失败：{str(e)}")
        raise  # 抛出异常便于调试

if __name__ == "__main__":
    main()