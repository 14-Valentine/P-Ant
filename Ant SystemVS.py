# aco_visual.py
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.cm as cm

# -------------------------
# PARAMETERS (แก้ได้ตรงนี้)
# -------------------------
random.seed(42)
np.random.seed(42)

n = 20     # จำนวนเมือง (nodes)
m = 10     # จำนวนมด (ants)
iters = 10      # จำนวนรอบใหญ่ (iterations)
alpha = 0.5 # pheromone importance (ความสำคัญของฟีโรโมน)
beta = 0.5  # visibility importance (ความสำคัญของระยะทาง)
rho = 0.5      # evaporation rate (อัตราการระเหยของฟีโรโมน)
Q = 1.0         # deposit constant (ค่าคงที่สำหรับการฝากฟีโรโมน)
tau0 = 1.0      # initial pheromone on every edge (ฟีโรโมนเริ่มต้นบนแต่ละเส้นทาง)

# visualization params
node_size = 80
edge_alpha = 0.25
visited_edge_alpha = 0.9
pause_time = 0.01

# -------------------------
# Build random graph (positions + distance matrix)
# -------------------------
pos = np.random.rand(n, 2)  # สุ่มตำแหน่งเมืองในกรอบ [0,1]^2

# สร้างเมทริกซ์ระยะทาง D (ระยะทางระหว่างเมืองทุกคู่)
D = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            D[i, j] = np.linalg.norm(pos[i] - pos[j])

# สร้างเมทริกซ์ฟีโรโมนเริ่มต้น tau และเมทริกซ์ visibility eta
tau = np.full((n, n), tau0)
np.fill_diagonal(tau, 0.0)  # ไม่ให้มีฟีโรโมนบนเส้นทางไปตัวเอง
eta = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if D[i, j] > 0:
            eta[i, j] = 1.0 / D[i, j]  # visibility = 1/distance

# -------------------------
# Helpers
# -------------------------
def tour_length_from_order(order):
    # คำนวณความยาวทัวร์จากลำดับเมืองที่เยี่ยมชม
    L = 0.0
    for k in range(n):
        i = order[k]
        j = order[(k+1) % n]  # กลับไปเมืองเริ่มต้น
        L += D[i, j]
    return L

def transition_probs_matrix(current, allowed, tau_snapshot):
    # คำนวณความน่าจะเป็นในการเลือกเมืองถัดไปสำหรับมดแต่ละตัว
    nums = []
    for j in allowed:
        # ความน่าจะเป็นขึ้นกับฟีโรโมนและ visibility
        nums.append((tau_snapshot[current, j] ** alpha) * (eta[current, j] ** beta))
    s = sum(nums)
    if s == 0.0:
        return [1.0/len(allowed)] * len(allowed)
    return [x/s for x in nums]

# -------------------------
# MAIN ACO SIMULATION (store all steps)
# -------------------------
# all_rounds[iter] = ข้อมูลแต่ละรอบ (iteration)
# แต่ละรอบเก็บข้อมูลมดทุกตัว (ants_steps) และผลลัพธ์สุดท้าย

all_rounds = []

global_best_len = float('inf')   # ความยาวทัวร์ที่ดีที่สุดที่พบ
global_best_tour = None          # ลำดับเมืองที่ดีที่สุดที่พบ

for it in range(iters):
    # สร้าง snapshot ของฟีโรโมนสำหรับการตัดสินใจในรอบนี้
    tau_snapshot = tau.copy()

    # เตรียม storage สำหรับข้อมูลแต่ละมดในรอบนี้
    iter_data = []          # เก็บขั้นตอนการเดินของมดแต่ละตัว
    completed_tours = []    # เก็บทัวร์สุดท้ายของแต่ละมด
    completed_lengths = []  # เก็บความยาวทัวร์สุดท้ายของแต่ละมด

    # สร้างทัวร์สำหรับมดแต่ละตัว
    for a in range(m):
        start = 0 # กำหนดให้ทุกมดเริ่มที่เมือง 0
        current = start
        visited = [start]
        edges = []
        partial_len = 0.0

        steps = []
        # บันทึกสถานะเริ่มต้น
        steps.append({
            "current": current,
            "visited": visited.copy(),
            "edges": edges.copy(),
            "partial_len": partial_len
        })

        allowed = [j for j in range(n) if j != current]
        # เดินไปเรื่อยๆ จนเยี่ยมชมครบทุกเมือง
        while len(visited) < n:
            allowed = [j for j in range(n) if j not in visited]
            probs = transition_probs_matrix(current, allowed, tau_snapshot)
            next_city = random.choices(allowed, weights=probs, k=1)[0]
            edges.append((current, next_city))
            partial_len += D[current, next_city]
            visited.append(next_city)
            current = next_city
            # บันทึกสถานะหลังเดินแต่ละก้าว
            steps.append({
                "current": current,
                "visited": visited.copy(),
                "edges": edges.copy(),
                "partial_len": partial_len
            })
        # ปิดทัวร์โดยกลับไปเมืองเริ่มต้น
        edges_with_return = edges.copy()
        edges_with_return.append((visited[-1], visited[0]))
        partial_len_closed = partial_len + D[visited[-1], visited[0]]
        steps.append({
            "current": visited[0],  # หลังกลับถึงเมืองเริ่มต้น
            "visited": visited.copy(),
            "edges": edges_with_return.copy(),
            "partial_len": partial_len_closed
        })

        iter_data.append(steps)
        completed_tours.append(visited.copy())
        completed_lengths.append(partial_len_closed)

        # อัปเดต global best ถ้าพบทัวร์ที่สั้นกว่าเดิม
        if partial_len_closed < global_best_len:
            global_best_len = partial_len_closed
            global_best_tour = visited.copy()

    # หลังจากมดทุกตัวเดินครบแล้ว อัปเดตฟีโรโมน
    # 1. ระเหยฟีโรโมน
    tau = (1.0 - rho) * tau
    np.fill_diagonal(tau, 0.0)

    # 2. ฝากฟีโรโมนบนเส้นทางที่มดแต่ละตัวเดิน
    for tour_nodes, L in zip(completed_tours, completed_lengths):
        deposit = Q / L
        for k in range(n):
            i = tour_nodes[k]
            j = tour_nodes[(k+1) % n]
            tau[i, j] += deposit
            tau[j, i] += deposit

    # บันทึกข้อมูลรอบนี้สำหรับการแสดงผล
    iter_record = {
        "ants_steps": iter_data,        # ข้อมูลขั้นตอนการเดินของมดแต่ละตัว
        "final_tours": completed_tours, # ทัวร์สุดท้ายของแต่ละมด
        "final_lengths": completed_lengths, # ความยาวทัวร์สุดท้าย
        "tau_after": tau.copy()         # ฟีโรโมนหลังอัปเดต
    }
    all_rounds.append(iter_record)

    # สรุปผลรอบนี้ทาง console
    best_idx = int(np.argmin(completed_lengths))
    print(f"[ITER {it}] Best this iter: Ant {best_idx}  L={completed_lengths[best_idx]:.4f}")

print("\nGLOBAL BEST:")
print(" Best tour (0-indexed):", global_best_tour)
print(" Best length:", global_best_len)

# -------------------------
# VISUALIZATION (matplotlib with 2 sliders)
# -------------------------
# เตรียมข้อมูลสำหรับ slider (จำนวน step สูงสุดแต่ละรอบ)
max_steps_per_iter = [ max(len(all_rounds[it]['ants_steps'][a]) for a in range(m)) for it in range(iters) ]
global_max_steps = max(max_steps_per_iter)

fig, ax = plt.subplots(figsize=(16, 16))
plt.subplots_adjust(left=0.05, right=0.98, bottom=0.22)  # ขยายพื้นที่สำหรับ slider

# วาดกราฟพื้นฐาน (เส้นสีเทา)
for i in range(n):
    for j in range(i+1, n):
        ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], color='gray', alpha=edge_alpha, zorder=1)

# วาด node เมือง
sc = ax.scatter(pos[:,0], pos[:,1], s=node_size, c='white', edgecolors='black', zorder=3)
for i, (x,y) in enumerate(pos):
    ax.text(x+0.01, y+0.01, str(i), fontsize=10, zorder=4)

ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.2, 1.2)
ax.set_aspect('equal', 'box')

# สร้าง slider สำหรับเลือก iteration และ step
ax_iter = plt.axes([0.15, 0.10, 0.7, 0.03])
ax_step = plt.axes([0.15, 0.05, 0.7, 0.03])

slider_iter = Slider(ax_iter, 'Iteration', 0, iters-1, valinit=0, valstep=1)
slider_step = Slider(ax_step, 'Step', 0, global_max_steps-1, valinit=0, valstep=1)

# สร้าง colormap สำหรับมดแต่ละตัว
cmap = cm.get_cmap('tab10', m)

# ตัวแปรสำหรับเก็บ artist ที่ต้องลบเมื่อ update plot
edge_lines = []   # รายการ Line2D สำหรับเส้นทางที่มดเดิน
current_scatter = None
text_handles = []

def update_plot(val):
    global edge_lines, current_scatter, text_handles

    iter_idx = int(slider_iter.val)
    step_idx = int(slider_step.val)

    # จำกัด step_idx ไม่ให้เกินจำนวน step ที่มีจริง
    max_steps = max_steps_per_iter[iter_idx]
    if step_idx >= max_steps:
        step_idx = max_steps - 1
        slider_step.set_val(step_idx)

    # ลบข้อความเดิมทุกครั้งก่อนวาดใหม่
    for t in text_handles:
        try:
            t.remove()
        except Exception:
            pass
    text_handles = []

    iter_data = all_rounds[iter_idx]['ants_steps']
    final_lengths = all_rounds[iter_idx]['final_lengths']
    best_local = int(np.argmin(final_lengths))

    ax.clear()
    # วาดกราฟพื้นฐาน (เส้นสีเทา)
    for i in range(n):
        for j in range(i+1, n):
            ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], color='gray', alpha=edge_alpha, zorder=1)
    # วาด node เมือง (สีขาว)
    ax.scatter(pos[:,0], pos[:,1], s=node_size, c='white', edgecolors='black', zorder=3)
    for i, (x,y) in enumerate(pos):
        ax.text(x+0.01, y+0.01, str(i), fontsize=10, zorder=4)

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal', 'box')
    ax.set_title(f"Iteration {iter_idx} / Step {step_idx}")

    ax.set_xticks([])
    ax.set_yticks([])

    # --- ข้อมูลมดอยู่นอกกราฟ ---
    for ant_idx in range(m):
        steps_list = iter_data[ant_idx]
        s_idx = min(step_idx, len(steps_list)-1)
        rec = steps_list[s_idx]
        col = cmap(ant_idx)
        t = fig.text(0.75, 0.95 - 0.04*ant_idx,
                     f"Ant {ant_idx}: partial L={rec['partial_len']:.3f}",
                     color=col, fontsize=10)
        text_handles.append(t)

    # --- Final lengths อยู่นอกกราฟ ---
    t2 = fig.text(0.02, 0.95, "Final Ls (this iter):", fontsize=10)
    text_handles.append(t2)
    for k, L in enumerate(final_lengths):
        txt = fig.text(0.02, 0.95 - 0.04*(k+1),
                       f"Ant {k}: L={L:.4f}" + ("  <-- best" if k==best_local else ""),
                       color=cmap(k), fontsize=9)
        text_handles.append(txt)

    # --- Global best อยู่นอกกราฟ ---
    gbest = global_best_len
    t3 = fig.text(0.02, 0.02, f"Global best L so far: {gbest:.4f}", fontsize=10)
    text_handles.append(t3)

    # --- วาดเส้นทางที่มดเดินใน step นี้ ---
    for ant_idx in range(m):
        steps_list = iter_data[ant_idx]
        s_idx = min(step_idx, len(steps_list)-1)
        rec = steps_list[s_idx]
        col = cmap(ant_idx)
        # วาดเส้นทางที่เดินแล้ว (แต่ละมดคนละสี)
        for (i, j) in rec['edges']:
            ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
                    color=col, alpha=visited_edge_alpha, linewidth=2, zorder=2)

    # --- วาด node ที่ถูกเยี่ยมชมแล้ว (เปลี่ยนสีเป็นฟ้าอ่อน) ---
    visited_nodes = set()
    for ant_idx in range(m):
        steps_list = iter_data[ant_idx]
        s_idx = min(step_idx, len(steps_list)-1)
        rec = steps_list[s_idx]
        visited_nodes.update(rec['visited'])
    node_colors = ['lightblue' if i in visited_nodes else 'white' for i in range(n)]
    ax.scatter(pos[:,0], pos[:,1], s=node_size, c=node_colors, edgecolors='black', zorder=3)

    plt.draw()

# ฟังก์ชันสำหรับเปลี่ยน iteration (slider)
def on_iter_change(v):
    # เมื่อ iteration เปลี่ยน ให้ปรับ step slider max และ reset step เป็น 0
    slider_step.valmax = max_steps_per_iter[int(v)]
    slider_step.ax.set_xlim(slider_step.valmin, slider_step.valmax - 1 if slider_step.valmax > 0 else 0)
    slider_step.set_val(0)
    update_plot(v)

slider_iter.on_changed(on_iter_change)
slider_step.on_changed(update_plot)

# initialize
update_plot(0)
plt.show()

# -------------------------
# SUMMARY PLOTS (หลังจบ ACO simulation)
# -------------------------

# 1. เก็บข้อมูลแต่ละรอบ
best_lengths = []    # ความยาวทัวร์ที่ดีที่สุดแต่ละรอบ
rms_lengths = []     # การกระจาย (RMS) ของความยาวทัวร์แต่ละรอบ
branch_counts = []   # จำนวนกิ่งฟีโรโมน (edge ที่มีฟีโรโมนสูงกว่าค่าเฉลี่ย)

for it in range(iters):
    final_lengths = all_rounds[it]['final_lengths']
    tau_after = all_rounds[it]['tau_after']

    # (a) Best length in this iteration
    best_lengths.append(np.min(final_lengths))

    # (b) RMS (standard deviation) of tour lengths
    rms_lengths.append(np.std(final_lengths))

    # (c) Average number of pheromone trail branches
    # นับ edge ที่มีฟีโรโมนสูงกว่าค่าเฉลี่ย (ไม่นับเส้นทแยงมุม)
    tau_mean = np.mean(tau_after)
    branches = np.sum((tau_after > tau_mean) & (tau_after > 0)) // 2
    branch_counts.append(branches / n)  # normalize ด้วยจำนวนเมือง

# 2. วาดกราฟสรุปผล
fig2, axs = plt.subplots(3, 1, figsize=(10, 12))
iters_range = np.arange(1, iters+1)

# (a) Solution dynamics
axs[0].plot(iters_range, best_lengths, '-k')
axs[0].set_title('(a) Solution dynamics')
axs[0].set_ylabel('Length of route')
axs[0].grid(True)

# (b) Solution scattering
axs[1].plot(iters_range, rms_lengths, '-k')
axs[1].set_title('(b) Solution scattering')
axs[1].set_ylabel('RMS')
axs[1].grid(True)

# (c) Average number of pheromone trail branches
axs[2].plot(iters_range, branch_counts, '-k')
axs[2].set_title('(c) Average number of pheromone trail branches')
axs[2].set_xlabel('Iteration number')
axs[2].set_ylabel('Number of branches')
axs[2].grid(True)

plt.tight_layout()
plt.show()
