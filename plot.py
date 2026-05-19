import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_diagram():
    # 设置画布
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')  # 关闭坐标轴

    # 字体和颜色配置
    font_family = 'sans-serif'
    blue_fill = '#dae8fc'
    blue_edge = '#6c8ebf'
    yellow_fill = '#fff2cc'
    yellow_edge = '#d6b656'
    red_fill = '#f8cecc'
    red_edge = '#b85450'
    text_color = '#000000'

    # 辅助函数：绘制圆角矩形框
    def draw_box(x, y, w, h, text, subtext="", color_fill=blue_fill, color_edge=blue_edge, icon=None):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                      linewidth=2, edgecolor=color_edge, facecolor=color_fill)
        ax.add_patch(rect)
        
        # 主文本
        ax.text(x + w/2, y + h/2 + (0.15 if subtext else 0), text, ha='center', va='center', 
                fontsize=12, fontweight='bold', family=font_family)
        # 副文本
        if subtext:
            ax.text(x + w/2, y + h/2 - 0.2, subtext, ha='center', va='center', 
                    fontsize=9, family=font_family)
        # 图标 (如雪花/火)
        if icon:
            ax.text(x + w - 0.3, y + h - 0.3, icon, ha='center', va='center', fontsize=16, color='#2c3e50')
        return rect

    # 辅助函数：绘制箭头
    def draw_arrow(x1, y1, x2, y2, color='black', style='->', connection='arc3,rad=0'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.5, shrinkA=0, shrinkB=0, connectionstyle=connection))

    # ==========================================
    # Part (a): Orthogonal Redundancy-Elimination
    # ==========================================
    
    # 标题
    ax.text(0.5, 8.5, "(a) Orthogonal Redundancy-Elimination & IB Filtering (Initialization)", 
            fontsize=14, fontweight='bold', ha='left')

    # W0 Box
    draw_box(0.5, 5.5, 2, 1.5, "$W_0$", "(Frozen\nPre-trained Weights)", icon="❄️")

    # Arrow to QR
    draw_arrow(2.7, 6.25, 3.5, 6.25)
    ax.text(3.1, 6.35, "QR\nDecomposition", fontsize=8, ha='center')

    # Q Box
    draw_box(3.8, 6.5, 1.5, 1, "$Q$", "(Orthogonal)")
    
    # R Triangle (绘制三角形)
    triangle = patches.Polygon([[3.8, 6.2], [5.3, 6.2], [3.8, 5.2]], closed=True, 
                               linewidth=2, edgecolor=blue_edge, facecolor=blue_fill)
    ax.add_patch(triangle)
    ax.text(4.2, 5.5, "$R$", fontsize=12, fontweight='bold')

    # 频谱条形图 (Diagonal Spectrum)
    # 绘制几个条形
    bar_x_start = 6.0
    bar_heights = [1.2, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1]
    for i, h in enumerate(bar_heights):
        color = '#d85c5c' if i < 2 else '#6c8ebf' # 前两个红色，后面蓝色
        rect = patches.Rectangle((bar_x_start + i*0.2, 5.5), 0.15, h, facecolor=color, edgecolor='grey', lw=0.5)
        ax.add_patch(rect)
    ax.text(bar_x_start + 0.8, 5.2, "Diagonal Spectrum\n($R_{kk}$)", ha='center', fontsize=9)
    
    # 标注 Top-2r
    ax.annotate('Top-$2r_{high}$\nNon-redundant\nSubspace', xy=(6.2, 6.7), xytext=(6.5, 7.5),
                arrowprops=dict(arrowstyle='->', color='#8b0000'), fontsize=9, ha='center')

    # 漏斗 (Information Bottleneck)
    funnel_x = 8.0
    funnel_y = 6.0
    funnel_poly = patches.Polygon([[funnel_x, funnel_y+0.5], [funnel_x+1, funnel_y+0.5], 
                                   [funnel_x+0.6, funnel_y], [funnel_x+0.4, funnel_y], [funnel_x, funnel_y+0.5]],
                                  closed=True, color='#f8cecc', ec='#b85450', lw=2)
    # 漏斗下部
    funnel_pipe = patches.Rectangle((funnel_x+0.4, funnel_y-0.3), 0.2, 0.3, color='#f8cecc', ec='#b85450', lw=2)
    ax.add_patch(funnel_pipe)
    ax.add_patch(funnel_poly)
    ax.text(funnel_x + 0.5, 5.2, "Information\nBottleneck (IB)", ha='center', fontsize=9)

    # 红色箭头连线
    draw_arrow(7.0, 6.8, funnel_x+0.5, 6.6, color='#8b0000') # Spectrum to Funnel
    draw_arrow(funnel_x+0.6, 5.8, 9.0, 7.0, color='#8b0000') # Funnel to A
    draw_arrow(funnel_x+0.6, 5.8, 9.0, 5.5, color='#8b0000') # Funnel to B

    # A_high Box
    draw_box(9.2, 6.8, 2.5, 1.2, "$A_{high}$", "(Trainable Init,\n$\\frac{m}{r_{high}} \\times r_{high}$)", yellow_fill, yellow_edge)
    
    # B_high Box
    draw_box(9.2, 5.0, 2.5, 1.2, "$B_{high}$", "(Trainable Init,\n$\\frac{n}{r_{high}} \\times r_{high}$)", yellow_fill, yellow_edge)

    # 减号圆圈
    circle = patches.Circle((12.5, 6.5), 0.2, facecolor='white', edgecolor='black', lw=1.5)
    ax.add_patch(circle)
    ax.text(12.5, 6.5, "$-$", ha='center', va='center', fontsize=14, fontweight='bold')

    # 连线到减号
    # A to Minus
    ax.plot([11.9, 12.5], [7.4, 7.4], color='black', lw=1.5) 
    ax.plot([12.5, 12.5], [7.4, 6.7], color='black', lw=1.5, marker='v', markevery=[1])
    
    # B to Minus (goes up)
    ax.plot([11.9, 12.5], [5.6, 5.6], color='black', lw=1.5)
    ax.plot([12.5, 12.5], [5.6, 6.3], color='black', lw=1.5, marker='^', markevery=[1])

    # W0 bypass text
    ax.text(8.0, 4.5, "$W_{res} = W_0 - A \\otimes B$", fontsize=10)
    # W0 to Minus line (long line)
    ax.plot([2.7, 12.2], [5.6, 5.6], color='black', lw=0) # phantom path
    # Simulate the curved line from R bottom to minus
    ax.plot([5.0, 5.0, 12.1], [5.2, 4.8, 4.8], color='black', lw=1.5) # path from R
    ax.plot([12.1, 12.3], [4.8, 6.35], color='black', lw=1.5) # up to circle

    # W_frozen Box
    draw_box(13.2, 5.5, 2.5, 2.0, "$W_{frozen}$", "(Frozen Residual:\n$W_0 - Init(A \\otimes B)$)", icon="❄️")
    draw_arrow(12.7, 6.5, 13.2, 6.5)


    # ==========================================
    # Part (b): High-Rank Trajectory Training
    # ==========================================
    
    ax.text(0.5, 4.0, "(b) High-Rank Trajectory Training (Fine-tuning View)", 
            fontsize=14, fontweight='bold', ha='left')

    # Left Yellow Boxes (abstracted)
    draw_box(0.5, 2.8, 2.5, 0.8, "", "", yellow_fill, yellow_edge)
    draw_box(0.5, 1.5, 2.5, 1.0, "(Trainable Init,\n$\\frac{n}{r_{high}} \\times r_{high}$)", "", yellow_fill, yellow_edge)
    
    # Kronecker Circle
    k_circle = patches.Circle((1.75, 2.65), 0.15, facecolor='white', edgecolor='#8b0000', lw=1.5)
    ax.add_patch(k_circle)
    ax.text(1.75, 2.65, "$\\times$", color='#8b0000', ha='center', va='center', fontsize=12)

    # Heatmap Grid (Simulated)
    grid_x, grid_y = 3.8, 1.5
    grid_w, grid_h = 5.0, 1.5
    # Draw background gradient (simulated with solid orange for simplicity)
    rect_grid = patches.Rectangle((grid_x, grid_y), grid_w, grid_h, facecolor='#ff7f50', edgecolor='black')
    ax.add_patch(rect_grid)
    # Draw grid lines
    for i in np.linspace(grid_x, grid_x+grid_w, 20):
        ax.plot([i, i], [grid_y, grid_y+grid_h], color='#a04000', lw=0.5, alpha=0.5)
    for j in np.linspace(grid_y, grid_y+grid_h, 10):
        ax.plot([grid_x, grid_x+grid_w], [j, j], color='#a04000', lw=0.5, alpha=0.5)
    
    ax.text(grid_x + grid_w/2, grid_y - 0.4, "$\\Delta W$ (High-Rank Increment, Trainable Trajectory),\n$rank(\\Delta W) = r_{high}^2$ (High-Rank)", 
            ha='center', fontsize=10)
    
    # Fire Icon
    ax.text(grid_x + grid_w, grid_y, "🔥", fontsize=20)

    # Arrows into grid
    draw_arrow(3.0, 2.65, 3.8, 2.65, color='#8b0000') # From circle to grid
    draw_arrow(3.0, 2.0, 3.8, 2.0, color='#8b0000')   # From box to grid

    # Loop arrow (Recurrent connection simulation)
    # ax.annotate('', xy=(3.8, 3.0), xytext=(8.8, 3.0), arrowprops=dict(arrowstyle='->', connectionstyle="bar,fraction=0.1"))

    # Plus Circle
    plus_circle = patches.Circle((9.5, 2.25), 0.2, facecolor='white', edgecolor='black', lw=1.5)
    ax.add_patch(plus_circle)
    ax.text(9.5, 2.25, "$+$", ha='center', va='center', fontsize=14, fontweight='bold')

    # Line from Grid to Plus
    draw_arrow(grid_x + grid_w, 2.25, 9.3, 2.25)

    # W_frozen (lower copy)
    draw_box(10.2, 1.5, 2.4, 2.0, "$W_{frozen}$", "(Frozen Residual:\n$W_0 - Init(A \\otimes B)$)", icon="❄️")
    
    # Line from W_frozen to Plus (Loop back logic visual trick in diagram)
    # The diagram actually shows W_frozen feeding INTO the addition for W_final? 
    # Or W_frozen + Delta W = W_final. Let's do that flow.
    draw_arrow(10.2, 2.5, 9.7, 2.5) # Arrow pointing left? diagram is a bit complex here
    # Actually diagram shows: Delta W + W_frozen -> W_final.
    # But there is a loop line coming from right to left. Let's simplify to linear flow for clarity
    
    draw_arrow(9.7, 2.25, 10.2, 2.25) # Plus to W_frozen (Wait, visual says Plus -> W_frozen? No, DeltaW + W_frozen -> W_final)
    # Correct flow based on standard ResNet/LoRA logic: W_frozen + DeltaW -> Output.
    # Visually in image: Grid -> Plus. W_frozen box is to the right. 
    # Let's just place W_final to the right of W_frozen.
    
    draw_arrow(12.6, 2.5, 13.5, 2.5)

    # W_final Box
    draw_box(13.5, 1.5, 2.2, 2.0, "$W_{final}$", "(Adapted Weights)")


    # ==========================================
    # Legend
    # ==========================================
    legend_x = 12.5
    legend_y = 0.2
    
    # Blue item
    rect_l1 = patches.Rectangle((legend_x, legend_y+0.6), 0.5, 0.2, fc=blue_fill, ec=blue_edge)
    ax.add_patch(rect_l1)
    ax.text(legend_x+0.6, legend_y+0.7, "Frozen/Structural Elements", va='center', fontsize=9)
    
    # Yellow item
    rect_l2 = patches.Rectangle((legend_x, legend_y+0.3), 0.5, 0.2, fc=yellow_fill, ec=yellow_edge)
    ax.add_patch(rect_l2)
    ax.text(legend_x+0.6, legend_y+0.4, "Trainable/Active Parameters", va='center', fontsize=9)
    
    # Red item
    rect_l3 = patches.Rectangle((legend_x, legend_y), 0.5, 0.2, fc=red_fill, ec=red_edge)
    ax.add_patch(rect_l3)
    ax.text(legend_x+0.6, legend_y+0.1, "IB & Highlights", va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

# 运行绘图
if __name__ == "__main__":
    draw_diagram()