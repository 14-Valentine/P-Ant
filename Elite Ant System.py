# aco_visual.py
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.cm as cm

# -------------------------
# PARAMETERS (แก้ได้ตรงนี้) (Tunable parameters)
# -------------------------
random.seed(42)        # ตั้งค่า seed ของ random ให้ผลซ้ำได้ (set random seed for reproducibility)
np.random.seed(42)     # ตั้งค่า seed ของ numpy ให้ผลซ้ำได้ (set numpy seed for reproducibility)

n = 20                 # จำนวนเมือง (nodes) (number of cities/nodes)
m = n                  # จำนวนมด (ants) (number of ants)
elite_e = 5            # จำนวนมดเอลิท (elitist weight e) — 0 = ปิด, >0 = เปิด Elite Ants
iters = 10             # จำนวนรอบใหญ่ (iterations) (number of global iterations)
alpha = 0.5            # ความสำคัญของฟีโรโมน (pheromone importance, α)
beta = 0.5             # ความสำคัญของระยะทาง/การมองเห็น (visibility importance, β = 1/dist)
rho = 0.5              # อัตราการระเหยฟีโรโมน (evaporation rate, ρ)
Q = 1.0                # ค่าคงที่สำหรับการฝากฟีโรโมน (deposit constant, Q)
tau0 = 1.0             # ฟีโรโมนเริ่มต้นบนแต่ละเส้นทาง (initial pheromone on every edge, τ0)

# visualization params (พารามิเตอร์สำหรับการแสดงผล)
node_size = 80               # ขนาดจุดเมือง (node size)
edge_alpha = 0.25            # ความโปร่งของเส้นฐาน (alpha of background edges)
visited_edge_alpha = 0.9     # ความโปร่งของเส้นที่ถูกเดินแล้ว (alpha of visited edges)
pause_time = 0.01            # เวลาหน่วงเล็กน้อยตอนวาด (render pause time)

# -------------------------
# Build random graph (positions + distance matrix)
# -------------------------
pos = np.random.rand(n, 2)   # สุ่มพิกัดเมืองในกรอบ [0,1]^2 (random 2D positions in unit square)

# สร้างเมทริกซ์ระยะทาง D (ระยะทางระหว่างเมืองทุกคู่) (distance matrix between all city pairs)
D = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            # ใช้ระยะทางยูคลิด (Euclidean distance)
            D[i, j] = np.linalg.norm(pos[i] - pos[j])

# สร้างเมทริกซ์ฟีโรโมนเริ่มต้น tau และเมทริกซ์ visibility eta (init pheromone & visibility matrices)
tau = np.full((n, n), tau0)          # เริ่มทุกขอบด้วยค่า tau0 (initialize all edges with τ0)
np.fill_diagonal(tau, 0.0)           # ไม่มีฟีโรโมนบนเส้นทางไปตัวเอง (no self-loop pheromone)
eta = np.zeros((n, n))               # visibility = 1 / distance (inverse distance heuristic)
for i in range(n):
    for j in range(n):
        if D[i, j] > 0:
            eta[i, j] = 1.0 / D[i, j]  # ถ้าระยะไกล → visibility ต่ำ (longer distance → lower visibility)

# -------------------------
# Helpers
# -------------------------
def tour_length_from_order(order):
    """
    คำนวณความยาวทัวร์จากลำดับเมืองที่เยี่ยมชม (compute tour length from visiting order).
    ปิดทัวร์โดยกลับไปเมืองแรกเสมอ (closed tour back to start).
    (Computes total length and closes the loop back to the start)
    """
    L = 0.0
    for k in range(n):
        i = order[k]
        j = order[(k+1) % n]  # กลับไปเมืองเริ่มต้นเมื่อ k ถึงตัวสุดท้าย (wrap-around to start)
        L += D[i, j]
    return L

def transition_probs_matrix(current, allowed, tau_snapshot):
    """
    คำนวณโพรบสำหรับการเลือกเมืองถัดไปตามสูตร ACO:
    p(i->j) ∝ (tau_ij^alpha) * (eta_ij^beta)
    จากนั้นทำ normalization ให้รวมกันได้ 1
    (Compute next-city probabilities via ACO rule and normalize to sum to 1)
    """
    nums = []
    for j in allowed:
        # ความน่าจะเป็นขึ้นกับฟีโรโมนและ visibility (pheromone & visibility contribution)
        nums.append((tau_snapshot[current, j] ** alpha) * (eta[current, j] ** beta))
    s = sum(nums)
    if s == 0.0:
        # ถ้าไม่มีข้อมูลนำทาง ให้สุ่มเท่ากันทุกตัวเลือก (fallback to uniform when zero-sum)
        return [1.0/len(allowed)] * len(allowed)
    return [x/s for x in nums]  # ทำให้ผลรวมเป็น 1 (normalize)

# -------------------------
# MAIN ACO SIMULATION (store all steps)
# -------------------------
# all_rounds[iter] = เก็บข้อมูลรายละเอียดของแต่ละรอบ (store per-iteration data)
# ในแต่ละรอบ เก็บเส้นทางทุกมดแบบ step-by-step เพื่อใช้วาดภาพภายหลัง
# (record step-by-step paths for visualization)
all_rounds = []

global_best_len = float('inf')   # ความยาวทัวร์ที่ดีที่สุดเท่าที่เคยพบ (global best tour length)
global_best_tour = None          # ลำดับเมืองที่ดีที่สุดเท่าที่เคยพบ (global best tour order)

for it in range(iters):
    # ใช้ snapshot ของฟีโรโมนเพื่อการตัดสินใจในรอบนี้ (pheromone snapshot for decision this iter)
    tau_snapshot = tau.copy()

    # เตรียมที่เก็บข้อมูลต่อมดในรอบนี้ (containers for this iteration)
    iter_data = []          # ขั้นตอนการเดินของมดแต่ละตัว (per-ant step records)
    completed_tours = []    # ทัวร์สุดท้ายของมดแต่ละตัว (final tours per ant)
    completed_lengths = []  # ความยาวทัวร์สุดท้าย (final tour lengths per ant)

    # สร้างทัวร์สำหรับมดแต่ละตัว (construct tour for each ant)
    for a in range(m):
        start = np.random.randint(0,n)                 # บังคับเริ่มที่เมือง 0 เพื่อเทียบกันง่าย (force start at city 0 for comparability)
        current = start           # ตำแหน่งปัจจุบันของมด (current city)
        visited = [start]         # รายการเมืองที่เยี่ยมชมแล้ว (visited list)
        edges = []                # ขอบที่เดินผ่าน (visited edges)
        partial_len = 0.0         # ระยะทางสะสมระหว่างเดิน (accumulated partial length)

        steps = []                # เก็บสถานะเป็นช่วงๆ (snapshot per step)
        # บันทึกสถานะเริ่มต้น (record initial state)
        steps.append({
            "current": current,
            "visited": visited.copy(),
            "edges": edges.copy(),
            "partial_len": partial_len
        })

        # allowed เมืองที่ยังไม่เคยไป (candidate next cities)
        allowed = [j for j in range(n) if j != current]

        # เดินจนกว่าจะครบทุกเมือง (loop until all cities visited)
        while len(visited) < n:
            allowed = [j for j in range(n) if j not in visited]  # ตัวเลือกเมืองที่ยังไม่เคยไป (unvisited candidates)
            probs = transition_probs_matrix(current, allowed, tau_snapshot)  # โพรบเลือกเมืองถัดไป (probabilities)
            # random.choices รองรับ weights (เลือกหนึ่งเมืองตามโพรบ) (sample next city by weights)
            next_city = random.choices(allowed, weights=probs, k=1)[0]

            # อัปเดตสถานะการเดิน (update path state)
            edges.append((current, next_city))                 # เพิ่มขอบที่เดิน (append traversed edge)
            partial_len += D[current, next_city]               # บวกระยะที่เดินเพิ่ม (accumulate length)
            visited.append(next_city)                          # ทำเครื่องหมายว่าเยี่ยมชมแล้ว (mark visited)
            current = next_city                                # ย้ายไปเมืองถัดไป (move to next)

            # บันทึกหลังแต่ละก้าว (record after each step)
            steps.append({
                "current": current,
                "visited": visited.copy(),
                "edges": edges.copy(),
                "partial_len": partial_len
            })

        # ปิดทัวร์โดยกลับสู่เมืองเริ่มต้น (close the tour by returning to start)
        edges_with_return = edges.copy()
        edges_with_return.append((visited[-1], visited[0]))     # เพิ่มขอบกลับเมืองแรก (add closing edge)
        partial_len_closed = partial_len + D[visited[-1], visited[0]]  # ระยะรวมหลังปิดทัวร์ (closed tour length)

        # บันทึกสเต็ปสุดท้ายหลังปิดทัวร์ (record final closed state)
        steps.append({
            "current": visited[0],   # กลับถึงเมืองเริ่มต้น (back to start)
            "visited": visited.copy(),
            "edges": edges_with_return.copy(),
            "partial_len": partial_len_closed
        })

        # เก็บข้อมูลของมดตัวนี้ (store this ant result)
        iter_data.append(steps)
        completed_tours.append(visited.copy())
        completed_lengths.append(partial_len_closed)

        # อัปเดต global best ถ้าพบทัวร์ที่สั้นกว่า (update global best if improved)
        if partial_len_closed < global_best_len:
            global_best_len = partial_len_closed
            global_best_tour = visited.copy()

    # หลังจากมดทุกตัวเดินครบแล้ว อัปเดตฟีโรโมน (update pheromones after all ants finished)
    # 1) ระเหยฟีโรโมน (evaporation): tau = (1 - rho) * tau
    tau = (1.0 - rho) * tau
    np.fill_diagonal(tau, 0.0)  # ไม่ให้เกิด self-loop (ensure no self-pheromone)

    # 2) ฝากฟีโรโมนตามเส้นทางของแต่ละมด (deposit pheromone along each ant's route)
    for tour_nodes, L in zip(completed_tours, completed_lengths):
        deposit = Q / L  # ฟีโรโมนต่อขอบตามความยาวทัวร์ (pheromone proportional to 1/L)
        for k in range(n):
            i = tour_nodes[k]
            j = tour_nodes[(k+1) % n]
            tau[i, j] += deposit
            tau[j, i] += deposit   # ทำให้เป็นกราฟไร้ทิศทาง (symmetric update for undirected graph)

    # 3) Elitist update: บูสต์ฟีโรโมนบนทัวร์ที่ดีที่สุดทั่วโลก (Eq: Δτ_ij,e = e * Q / L+)
    if elite_e > 0 and global_best_tour is not None:
        L_plus = global_best_len
        elite_deposit = elite_e * Q / L_plus
        for k in range(n):
            i = global_best_tour[k]
            j = global_best_tour[(k+1) % n]
            tau[i, j] += elite_deposit
            tau[j, i] += elite_deposit


    # บันทึกข้อมูลรอบนี้สำหรับการแสดงผล (record this iteration for visualization)
    iter_record = {
        "ants_steps": iter_data,             # ขั้นตอนของมดทุกตัว (per-ant steps)
        "final_tours": completed_tours,      # ทัวร์สุดท้าย (final tours)
        "final_lengths": completed_lengths,  # ความยาวทัวร์สุดท้าย (final lengths)
        "tau_after": tau.copy()              # ฟีโรโมนหลังอัปเดต (pheromone after update)
    }
    all_rounds.append(iter_record)

    # แสดงสรุปผลรอบนี้ใน console (print per-iteration summary)
    best_idx = int(np.argmin(completed_lengths))
    print(f"[ITER {it}] Best this iter: Ant {best_idx}  L={completed_lengths[best_idx]:.4f}")

print("\nGLOBAL BEST:")
print(" Best tour (0-indexed):", global_best_tour)  # แสดงลำดับเมือง (show best tour order)
print(" Best length:", global_best_len)             # แสดงความยาวที่ดีที่สุด (show best length)

# -------------------------
# VISUALIZATION (matplotlib with 2 sliders)
# -------------------------
# เตรียมข้อมูลสำหรับ slider (จำนวน step สูงสุดแต่ละรอบ) (max steps per iteration)
max_steps_per_iter = [ max(len(all_rounds[it]['ants_steps'][a]) for a in range(m)) for it in range(iters) ]
global_max_steps = max(max_steps_per_iter)  # ใช้กำหนด max ของ slider step (used for step slider max)

fig, ax = plt.subplots(figsize=(16, 16))
# ขยายพื้นที่ด้านล่างให้พอวาง slider (increase bottom margin for sliders)
plt.subplots_adjust(left=0.05, right=0.98, bottom=0.22)

# วาดกราฟพื้นฐาน (เส้นสีเทา) เพื่อให้เห็นโครงข่ายทั้งหมด (draw background complete graph)
for i in range(n):
    for j in range(i+1, n):
        ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], color='gray', alpha=edge_alpha, zorder=1)

# วาด node เมือง (draw city nodes)
sc = ax.scatter(pos[:,0], pos[:,1], s=node_size, c='white', edgecolors='black', zorder=3)
for i, (x,y) in enumerate(pos):
    ax.text(x+0.01, y+0.01, str(i), fontsize=10, zorder=4)  # ใส่หมายเลขเมือง (label city index)

# ตั้งขอบเขตและอัตราส่วนแกน (set axes limits and aspect)
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.2, 1.2)
ax.set_aspect('equal', 'box')

# สร้าง slider สำหรับเลือก iteration และ step (sliders for iteration & step)
ax_iter = plt.axes([0.15, 0.10, 0.7, 0.03])  # พื้นที่วาด slider iteration (axes for iteration slider)
ax_step = plt.axes([0.15, 0.05, 0.7, 0.03])  # พื้นที่วาด slider step (axes for step slider)

slider_iter = Slider(ax_iter, 'Iteration', 0, iters-1, valinit=0, valstep=1)             # เลือกรอบ (iteration slider)
slider_step = Slider(ax_step, 'Step', 0, global_max_steps-1, valinit=0, valstep=1)       # เลือกก้าว (step slider)

# สร้าง colormap สำหรับมดแต่ละตัว (colormap for ants)
cmap = cm.get_cmap('tab10', m)

# ตัวแปรเก็บ artist/handles ที่ต้องลบเมื่อ update plot (keep handles to remove on update)
edge_lines = []   # รายการเส้นทางที่วาด (list of Line2D for edges)
current_scatter = None
text_handles = [] # เก็บข้อความนอกกราฟ (off-axes text handles)

def update_plot(val):
    """
    callback เมื่อเลื่อน slider: ล้างภาพเดิม, วาดเส้นทางตาม step ของแต่ละมด,
    แสดงสรุประยะทางบางส่วน และไฮไลท์โหนดที่ถูกเยี่ยมชม
    (Slider callback: redraw paths at a given step and show partial stats)
    """
    global edge_lines, current_scatter, text_handles

    iter_idx = int(slider_iter.val)   # รอบที่เลือก (selected iteration)
    step_idx = int(slider_step.val)   # สเต็ปที่เลือก (selected step)

    # จำกัด step_idx ไม่ให้เกินจำนวน step จริงของรอบนี้ (cap step within available steps)
    max_steps = max_steps_per_iter[iter_idx]
    if step_idx >= max_steps:
        step_idx = max_steps - 1
        slider_step.set_val(step_idx)  # sync slider value (safeguard)

    # ลบข้อความนอกกราฟเก่าก่อนวาดใหม่ (remove previous off-axes texts)
    for t in text_handles:
        try:
            t.remove()
        except Exception:
            pass
    text_handles = []

    iter_data = all_rounds[iter_idx]['ants_steps']        # ข้อมูลการเดินของมดในรอบนี้ (per-ant steps this iter)
    final_lengths = all_rounds[iter_idx]['final_lengths'] # ความยาวทัวร์สุดท้ายในรอบนี้ (final lengths this iter)
    best_local = int(np.argmin(final_lengths))            # ดัชนีมดที่ดีที่สุดในรอบนี้ (best ant index this iter)

    # เคลียร์แกนก่อนวาดใหม่ (clear axes before re-draw)
    ax.clear()

    # วาดกราฟพื้นฐาน (เส้นสีเทา) (draw background full graph)
    for i in range(n):
        for j in range(i+1, n):
            ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], color='gray', alpha=edge_alpha, zorder=1)

    # วาด node เมือง (สีขาว) (draw nodes as white circles)
    ax.scatter(pos[:,0], pos[:,1], s=node_size, c='white', edgecolors='black', zorder=3)
    for i, (x,y) in enumerate(pos):
        ax.text(x+0.01, y+0.01, str(i), fontsize=10, zorder=4)

    # ตั้งค่ากรอบแสดงผลอีกครั้ง (reset axes settings)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal', 'box')
    ax.set_title(f"Iteration {iter_idx} / Step {step_idx}")  # ชื่อกราฟบอกสถานะ (title showing state)

    ax.set_xticks([])  # เอาแกน x ออกเพื่อความสะอาด (hide x ticks)
    ax.set_yticks([])  # เอาแกน y ออกเพื่อความสะอาด (hide y ticks)

    # --- ข้อมูลมดอยู่นอกกราฟ (off-axes info per ant) ---
    for ant_idx in range(m):
        steps_list = iter_data[ant_idx]
        s_idx = min(step_idx, len(steps_list)-1)  # ป้องกันเกินช่วง (guard index)
        rec = steps_list[s_idx]                   # สถานะของมดตัวนี้ ณ step ที่เลือก (state at selected step)
        col = cmap(ant_idx)
        t = fig.text(
            0.75, 0.95 - 0.04*ant_idx,
            f"Ant {ant_idx}: partial L={rec['partial_len']:.3f}",
            color=col, fontsize=10
        )
        text_handles.append(t)

    # --- Final lengths อยู่นอกกราฟ (show final lengths for this iter) ---
    t2 = fig.text(0.02, 0.95, "Final Ls (this iter):", fontsize=10)
    text_handles.append(t2)
    for k, L in enumerate(final_lengths):
        txt = fig.text(
            0.02, 0.95 - 0.04*(k+1),
            f"Ant {k}: L={L:.4f}" + ("  <-- best" if k == best_local else ""),
            color=cmap(k), fontsize=9
        )
        text_handles.append(txt)

    # --- Global best อยู่นอกกราฟ (show global best so far) ---
    gbest = global_best_len
    t3 = fig.text(0.02, 0.02, f"Global best L so far: {gbest:.4f}", fontsize=10)
    text_handles.append(t3)

    # --- วาดเส้นทางที่มดเดินใน step นี้ (draw traversed edges up to this step) ---
    for ant_idx in range(m):
        steps_list = iter_data[ant_idx]
        s_idx = min(step_idx, len(steps_list)-1)
        rec = steps_list[s_idx]
        col = cmap(ant_idx)
        # วาดเส้นทางที่เดินแล้ว (แต่ละมดคนละสี) (draw edges; different color per ant)
        for (i, j) in rec['edges']:
            ax.plot(
                [pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
                color=col, alpha=visited_edge_alpha, linewidth=2, zorder=2
            )

    # --- วาด node ที่ถูกเยี่ยมชมแล้ว (เปลี่ยนสีเป็นฟ้าอ่อน) (mark visited nodes) ---
    visited_nodes = set()
    for ant_idx in range(m):
        steps_list = iter_data[ant_idx]
        s_idx = min(step_idx, len(steps_list)-1)
        rec = steps_list[s_idx]
        visited_nodes.update(rec['visited'])

    node_colors = ['lightblue' if i in visited_nodes else 'white' for i in range(n)]
    ax.scatter(pos[:,0], pos[:,1], s=node_size, c=node_colors, edgecolors='black', zorder=3)

    plt.draw()  # อัปเดตภาพ (trigger redraw)

# ฟังก์ชันสำหรับเปลี่ยน iteration (slider) (when iteration slider changes)
def on_iter_change(v):
    # เมื่อ iteration เปลี่ยน ให้ปรับ step slider max และ reset step เป็น 0
    # (when iteration changes, update step slider's max and reset to 0)
    slider_step.valmax = max_steps_per_iter[int(v)]
    slider_step.ax.set_xlim(slider_step.valmin, slider_step.valmax - 1 if slider_step.valmax > 0 else 0)
    slider_step.set_val(0)
    update_plot(v)  # วาดใหม่สำหรับ iter ใหม่นี้ (redraw for new iteration)

# ผูก callback เข้ากับ sliders (bind callbacks)
slider_iter.on_changed(on_iter_change)
slider_step.on_changed(update_plot)

# initialize การแสดงผลครั้งแรก (initial draw)
update_plot(0)
plt.show()

# -------------------------
# SUMMARY PLOTS (หลังจบ ACO simulation) (post-simulation summary)
# -------------------------

# 1) รวบรวมข้อมูลต่อรอบเพื่อ plot สรุป (collect per-iteration stats)
best_lengths = []    # ความยาวทัวร์ที่ดีที่สุดแต่ละรอบ (best length per iteration)
rms_lengths = []     # ค่ากระจายมาตรฐานของความยาวทัวร์ (std/RMS of lengths per iter)
branch_counts = []   # จำนวนกิ่งฟีโรโมน (edge ที่มี tau สูงกว่าค่าเฉลี่ย) / n (avg branches)

for it in range(iters):
    final_lengths = all_rounds[it]['final_lengths']
    tau_after = all_rounds[it]['tau_after']

    # (a) best length ในรอบนี้ (best length this iteration)
    best_lengths.append(np.min(final_lengths))

    # (b) ความกระจายของความยาวทัวร์ (scattering; std as RMS proxy)
    rms_lengths.append(np.std(final_lengths))

    # (c) นับจำนวนขอบที่ tau > ค่าเฉลี่ย (average number of strong-pheromone branches)
    # หาร 2 เพราะกราฟไร้ทิศทาง (divide by 2 for undirected symmetry)
    tau_mean = np.mean(tau_after)
    branches = np.sum((tau_after > tau_mean) & (tau_after > 0)) // 2
    branch_counts.append(branches / n)  # แปลงให้เทียบตามจำนวนเมือง (normalize by n)

# 2) วาดกราฟสรุปผล (draw summary charts)
fig2, axs = plt.subplots(3, 1, figsize=(10, 12))
iters_range = np.arange(1, iters+1)

# (a) Solution dynamics — แนวโน้มความยาวที่ดีที่สุด (trend of best length)
axs[0].plot(iters_range, best_lengths, '-k')  # เส้นสีดำ (black line)
axs[0].set_title('(a) Solution dynamics')
axs[0].set_ylabel('Length of route')
axs[0].grid(True)

# (b) Solution scattering — การกระจายของผลลัพธ์ในแต่ละรอบ (spread/variance per iter)
axs[1].plot(iters_range, rms_lengths, '-k')
axs[1].set_title('(b) Solution scattering')
axs[1].set_ylabel('RMS')
axs[1].grid(True)

# (c) Average number of pheromone trail branches — จำนวนกิ่งฟีโรโมนเฉลี่ย (avg pheromone branches)
axs[2].plot(iters_range, branch_counts, '-k')
axs[2].set_title('(c) Average number of pheromone trail branches')
axs[2].set_xlabel('Iteration number')
axs[2].set_ylabel('Number of branches')
axs[2].grid(True)

plt.tight_layout()
plt.show()
