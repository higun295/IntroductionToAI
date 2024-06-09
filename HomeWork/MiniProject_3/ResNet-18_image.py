import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 그림 크기 설정
fig, ax = plt.subplots(figsize=(16, 6))

# 각 블록의 위치와 크기 정의
blocks = [
    {"name": "Input Image", "xy": (0, 3), "width": 3, "height": 1, "color": "lightgray"},
    {"name": "Conv1\n7x7, 64, stride 2", "xy": (4, 3), "width": 3, "height": 1, "color": "skyblue"},
    {"name": "MaxPool\n3x3, stride 2", "xy": (8, 3), "width": 3, "height": 1, "color": "lightgreen"},
    {"name": "ResBlock\n64, 3x3\nx2", "xy": (12, 3), "width": 3, "height": 1, "color": "lightcoral"},
    {"name": "ResBlock\n128, 3x3\nx2, stride 2", "xy": (16, 3), "width": 3, "height": 1, "color": "lightcoral"},
    {"name": "ResBlock\n256, 3x3\nx2, stride 2", "xy": (20, 3), "width": 3, "height": 1, "color": "lightcoral"},
    {"name": "ResBlock\n512, 3x3\nx2, stride 2", "xy": (24, 3), "width": 3, "height": 1, "color": "lightcoral"},
    {"name": "AvgPool", "xy": (28, 3), "width": 3, "height": 1, "color": "lightgreen"},
    {"name": "Fully Connected\n512 -> num_classes", "xy": (32, 3), "width": 3, "height": 1, "color": "skyblue"},
]

# 블록을 그림에 추가
for block in blocks:
    rect = patches.FancyBboxPatch(block["xy"], block["width"], block["height"], boxstyle="round,pad=0.3", ec="black", fc=block["color"])
    ax.add_patch(rect)
    rx, ry = block["xy"]
    cx = rx + block["width"] / 2.0
    cy = ry + block["height"] / 2.0
    ax.annotate(block["name"], (cx, cy), color="black", weight="bold", fontsize=10, ha="center", va="center")

# 화살표 추가
for i in range(len(blocks) - 1):
    ax.annotate('', xy=(blocks[i+1]["xy"][0], blocks[i+1]["xy"][1] + blocks[i+1]["height"] / 2),
                xytext=(blocks[i]["xy"][0] + blocks[i]["width"], blocks[i]["xy"][1] + blocks[i]["height"] / 2),
                arrowprops=dict(arrowstyle="->", color="black"))

# 축 설정
ax.set_xlim(-1, 37)
ax.set_ylim(2, 5)
ax.axis('off')

plt.title("ResNet-18 Architecture")
plt.show()
