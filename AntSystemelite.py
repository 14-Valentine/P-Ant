import random # Library for generating random numbers / ไลบรารีสำหรับสร้างตัวเลขสุ่ม
import numpy as np # Library for numerical operations on arrays / ไลบรารีสำหรับจัดการข้อมูลที่เป็นตัวเลขในรูปแบบอาเรย์
import matplotlib.pyplot as plt # Library for creating 2D plots / ไลบรารีสำหรับสร้างกราฟ 2D
import matplotlib.gridspec as gridspec # Used to manage subplot layouts / ใช้สำหรับจัดการเลย์เอาต์ของกราฟใน subplot
from matplotlib.widgets import Slider # Used to create interactive sliders / ใช้สำหรับสร้างแถบเลื่อน (Slider) แบบโต้ตอบ
import matplotlib.cm as cm # Used for handling colormaps / ใช้สำหรับจัดการแผนที่สี (Colormap)

# -------------------------
# PARAMETERS / พารามิเตอร์
# -------------------------
random.seed(42) # Set seed for random number generator / กำหนด seed เพื่อให้ผลลัพธ์การสุ่มเหมือนเดิมทุกครั้ง
np.random.seed(42) # Set seed for numpy's random generator / กำหนด seed สำหรับ numpy
n = 20 # Number of cities / จำนวนเมือง
m = n # Number of ants / จำนวนมด
iters = 100 # Number of iterations / จำนวนรอบการทำงาน
alpha = 0.5 # Pheromone importance factor / น้ำหนักของฟีโรโมน
beta = 0.5 # Heuristic importance factor / น้ำหนักของ Heuristic (ระยะทาง)
rho = 0.5 # Pheromone evaporation rate / อัตราการระเหยของฟีโรโมน
Q = 1.0 # Pheromone constant / ค่าคงที่สำหรับคำนวณฟีโรโมน
tau0 = 1.0 # Initial pheromone value / ค่าฟีโรโมนเริ่มต้น
elite_weight = 2.0 # Extra weight for the elite ant / น้ำหนักพิเศษสำหรับมด Elite

# Visualization params / พารามิเตอร์การแสดงผล
node_size = 80 # Size of city nodes / ขนาดของเมือง
edge_alpha = 0.25 # Transparency of unvisited edges / ความทึบของเส้นทางที่ยังไม่ถูกเลือก
visited_edge_alpha = 0.9 # Transparency of visited edges / ความทึบของเส้นทางที่ถูกเลือก

# -------------------------
# GRAPH SETUP / การตั้งค่ากราฟ
# -------------------------
pos = np.random.rand(n, 2) # Randomly generate city positions / สร้างตำแหน่งสุ่มของเมือง n เมืองใน 2 มิติ
# Calculate the distance matrix between all cities / คำนวณเมทริกซ์ระยะทางระหว่างเมือง (D)
D = np.array(
    [[np.linalg.norm(pos[i] - pos[j]) if i != j else 0 for j in range(n)]
     for i in range(n)])
# Calculate the heuristic matrix (1/distance) / คำนวณเมทริกซ์ Heuristic (eta) จากระยะทาง
eta = np.array(
    [[1.0 / D[i, j] if D[i, j] > 0 else 0 for j in range(n)]
     for i in range(n)])

# -------------------------
# HELPERS / ฟังก์ชันเสริม
# -------------------------
def tour_length_from_order(order):
    """Calculates the total length of a tour."""
    """คำนวณความยาวทั้งหมดของเส้นทาง"""
    if not order or len(order) != n:
        return float('inf') # Return infinity for invalid tours / คืนค่าอนันต์สำหรับเส้นทางที่ไม่ถูกต้อง
    return sum(D[order[k], order[(k + 1) % n]] for k in range(n))

def transition_probs_matrix(current, allowed, tau_snapshot):
    """Calculates transition probabilities for an ant."""
    """คำนวณความน่าจะเป็นในการเคลื่อนที่สำหรับมด"""
    # Numerator of the transition probability formula / ตัวเศษของสมการความน่าจะเป็น
    nums = [(tau_snapshot[current, j] ** alpha) * (eta[current, j] ** beta)
            for j in allowed]
    s = sum(nums) # Denominator of the formula / ตัวส่วนของสมการ
    if s == 0.0:
        return ([1.0 / len(allowed)] * len(allowed) if allowed else []) # Avoid division by zero / ป้องกันการหารด้วยศูนย์
    return [x / s for x in nums] # Return calculated probabilities / คืนค่าความน่าจะเป็น

# =================================================================
# FUNCTION FOR RUNNING A SINGLE ACO SIMULATION
# ฟังก์ชันสำหรับรันการจำลอง ACO เพียงครั้งเดียว
# =================================================================
def run_aco_simulation(elite_weight_val):
    """
    Runs an Ant Colony Optimization simulation with a specified elite weight.
    รันการจำลองอัลกอริทึม ACO ด้วยการกำหนดน้ำหนัก Elite ที่ระบุ
    """
    tau = np.full((n, n), tau0) # Initialize pheromone matrix / สร้างเมทริกซ์ฟีโรโมนเริ่มต้น
    np.fill_diagonal(tau, 0)
    all_rounds = []
    gbest_len = float('inf') # Global best tour length / ความยาวเส้นทางที่ดีที่สุดทั่วโลก
    gbest_tour = None # Global best tour order / ลำดับเส้นทางที่ดีที่สุดทั่วโลก

    for it in range(iters): # Loop through iterations / วนลูปตามจำนวนรอบการทำงาน
        tau_snapshot = tau.copy() # Snapshot of pheromone matrix / สร้างสำเนาของเมทริกซ์ฟีโรโมน
        iter_data = {"ants_steps": [], "final_lengths": [],
                     "gbest_tour_this_iter": gbest_tour} # Data for the current iteration / Dictionary สำหรับเก็บข้อมูลของรอบนี้

        for a in range(m): # Loop through ants / วนลูปตามจำนวนมด
            start = np.random.randint(0, n) # Random start city / เมืองเริ่มต้นแบบสุ่ม
            visited, edges, partial_len, steps = [start], [], 0.0, []
            steps.append({"visited": visited.copy(), "edges": edges.copy(),
                          "partial_len": partial_len}) # Store initial state / เก็บสถานะเริ่มต้น
            while len(visited) < n: # Until all cities are visited / จนกว่ามดจะเดินทางครบทุกเมือง
                current = visited[-1]
                allowed = [j for j in range(n) if j not in visited]
                probs = transition_probs_matrix(current, allowed, tau_snapshot)
                if not allowed:
                    break
                next_city = random.choices(allowed, weights=probs, k=1)[0] # Choose next city based on probability / สุ่มเลือกเมืองถัดไปตามความน่าจะเป็น
                edges.append((current, next_city))
                partial_len += D[current, next_city]
                visited.append(next_city)
                steps.append({"visited": visited.copy(), "edges": edges.copy(),
                              "partial_len": partial_len})

            if len(visited) == n: # If a full tour is completed / ถ้าเดินทางครบทุกเมืองแล้ว
                edges.append((visited[-1], visited[0])) # Add return path / เพิ่มเส้นทางกลับสู่เมืองเริ่มต้น
                partial_len += D[visited[-1], visited[0]]
                steps.append({"visited": visited.copy(),
                              "edges": edges.copy(),
                              "partial_len": partial_len})
                iter_data["ants_steps"].append(steps)
                iter_data["final_lengths"].append(partial_len)
                if partial_len < gbest_len:
                    gbest_len, gbest_tour = partial_len, visited.copy() # Update Global Best Tour / อัปเดต Global Best Tour

        iter_data["gbest_tour_this_iter"] = gbest_tour
        tau *= (1.0 - rho) # Pheromone evaporation / การระเหยของฟีโรโมน
        for i, tour in enumerate(iter_data["ants_steps"]): # Pheromone deposit / การฝากฟีโรโมน
            L = iter_data["final_lengths"][i]
            deposit = Q / L
            for (c1, c2) in tour[-1]["edges"]:
                tau[c1, c2] += deposit

        if elite_weight_val > 0 and gbest_tour: # Elitist ACO / หากเป็น Elitist ACO
            elite_deposit = elite_weight_val * (Q / gbest_len) # Calculate elite pheromone / คำนวณปริมาณฟีโรโมนพิเศษ
            for k in range(n):
                tau[gbest_tour[k], gbest_tour[(k + 1) % n]] += elite_deposit # Deposit elite pheromone on best path / ฝากฟีโรโมนพิเศษลงบนเส้นทาง Global Best

        iter_data["tau_after"] = tau.copy()
        all_rounds.append(iter_data)
    return all_rounds

# =================================================================
# MAIN EXECUTION AND PLOTTING / การรันและแสดงผลหลัก
# =================================================================
# Define elite weights to compare. elite_weight = 0 is equivalent to Standard ACO.
# กำหนดน้ำหนัก Elite ที่ต้องการเปรียบเทียบ โดย Elite_weight = 0 คือ Standard ACO
elite_weights_to_compare = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
all_results = {} # Dictionary to store all simulation results / Dictionary สำหรับเก็บผลลัพธ์การจำลองทั้งหมด
for ew in elite_weights_to_compare: # Loop to run simulation for each elite weight / วนลูปเพื่อรันการจำลองแต่ละค่า
    key = f"Elite_Weight_{ew}"
    if ew == 0.0:
        key = "Standard_ACO"
    all_results[key] = run_aco_simulation(ew)

# Retrieve results for interactive plots (Standard ACO vs. a specific Elitist ACO)
# ดึงผลลัพธ์สำหรับกราฟแบบโต้ตอบ (Standard ACO vs. Elitist ACO ที่ elite_weight=2.0)
all_rounds_std = all_results.get("Standard_ACO")
all_rounds_elite = all_results.get("Elite_Weight_2.0")

# --- Interactive Plot / กราฟแบบโต้ตอบ ---
fig = plt.figure(figsize=(22, 11))
gs = gridspec.GridSpec(1, 4, width_ratios=[4, 2, 4, 2], wspace=0.3)
ax_std = fig.add_subplot(gs[0, 0])
ax_std_text = fig.add_subplot(gs[0, 1])
ax_elite = fig.add_subplot(gs[0, 2])
ax_elite_text = fig.add_subplot(gs[0, 3])
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.85)
overall_best_len_std, overall_best_iter_std = min(
    (tour_length_from_order(r['gbest_tour_this_iter']), i)
    for i, r in enumerate(all_rounds_std))
overall_best_len_elite, overall_best_iter_elite = min(
    (tour_length_from_order(r['gbest_tour_this_iter']), i)
    for i, r in enumerate(all_rounds_elite))
fig.suptitle(
    "ACO Comparison (Scrollable Info Panels) / การเปรียบเทียบ ACO (แผงข้อมูลแบบเลื่อนได้)\n"
    f"Overall Best Standard: {overall_best_len_std:.4f} (found in iter {overall_best_iter_std}) | "
    f"Overall Best Elitist: {overall_best_len_elite:.4f} (found in iter {overall_best_iter_elite})",
    fontsize=14)
max_steps = n + 1
ax_iter = plt.axes([0.15, 0.1, 0.7, 0.03])
ax_step = plt.axes([0.15, 0.05, 0.7, 0.03])
slider_iter = Slider(ax_iter, 'Iteration / รอบ', 0, iters - 1, valinit=0,
                     valstep=1)
slider_step = Slider(ax_step, 'Step / ขั้น', 0, max_steps - 1, valinit=0,
                     valstep=1)
cmap = cm.get_cmap('tab10', m)
ANTS_TO_DISPLAY = 18
scroll_offset_std = 0
scroll_offset_elite = 0

def on_scroll(event):
    """Handles mouse scroll events to pan through text panels."""
    """จัดการเหตุการณ์การเลื่อนเมาส์เพื่อเลื่อนดูข้อมูลในแผงข้อความ"""
    global scroll_offset_std, scroll_offset_elite
    if event.inaxes == ax_std_text:
        max_offset = max(0, m - ANTS_TO_DISPLAY)
        scroll_offset_std = int(np.clip(scroll_offset_std - event.step, 0,
                                        max_offset))
        update_plot(slider_iter.val)
    elif event.inaxes == ax_elite_text:
        max_offset = max(0, m - ANTS_TO_DISPLAY)
        scroll_offset_elite = int(np.clip(scroll_offset_elite - event.step, 0,
                                         max_offset))
        update_plot(slider_iter.val)

fig.canvas.mpl_connect('scroll_event', on_scroll)

def draw_graph_on_ax(ax, title, iter_rec, step_idx):
    """Draws the main graph on a specified axes."""
    """วาดกราฟหลักลงบนแกนที่ระบุ"""
    ax.clear()
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', 'box')
    for i in range(n):
        for j in range(i + 1, n):
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    color='gray', alpha=edge_alpha, zorder=1) # Draw all possible edges / วาดเส้นทางที่เป็นไปได้ทั้งหมด
    ants_steps = iter_rec.get("ants_steps", [])
    for ant_idx, steps in enumerate(ants_steps):
        s_idx = min(step_idx, len(steps) - 1)
        rec = steps[s_idx]
        for (c1, c2) in rec['edges']:
            ax.plot([pos[c1, 0], pos[c2, 0]], [pos[c1, 1], pos[c2, 1]],
                    color=cmap(ant_idx), alpha=visited_edge_alpha,
                    linewidth=2, zorder=2) # Draw ant paths / วาดเส้นทางที่มดเดิน
    gbest_tour = iter_rec.get("gbest_tour_this_iter")
    if gbest_tour:
        for k in range(n):
            i, j = gbest_tour[k], gbest_tour[(k + 1) % n]
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    color='black', linewidth=3, linestyle='--',
                    zorder=3) # Draw the Global Best tour / วาดเส้นทาง Global Best
    visited_nodes = set().union(
        *(steps[min(step_idx, len(steps) - 1)]['visited']
          for steps in ants_steps))
    node_colors = (
        ['lightblue' if i in visited_nodes else 'white' for i in range(n)])
    ax.scatter(pos[:, 0], pos[:, 1], s=node_size, c=node_colors,
               edgecolors='black', zorder=4) # Draw city nodes / วาดจุดเมือง

def update_text_panel(ax, iter_rec, step_idx, scroll_offset):
    """Updates the text panel with ant information."""
    """อัปเดตแผงข้อความด้วยข้อมูลของมด"""
    ax.clear()
    ax.axis('off')
    final_lengths = iter_rec.get("final_lengths", [])
    ants_steps = iter_rec.get("ants_steps", [])
    header = f"{'Ant':<5}{'Final L':<12}{'Partial L':<12}"
    ax.text(0, 1.0, header, fontsize=9, weight='bold', va='top',
            fontfamily='monospace')
    ax.text(0, 0.97, "-" * 35, fontsize=9, va='top', fontfamily='monospace')
    if not final_lengths:
        return
    best_local_idx = np.argmin(final_lengths)
    for i in range(ANTS_TO_DISPLAY):
        ant_idx = i + scroll_offset
        if ant_idx >= len(final_lengths):
            break
        final_L = final_lengths[ant_idx]
        is_best = " B" if ant_idx == best_local_idx else ""
        final_L_str = f"{final_L:<9.4f}{is_best}"
        s_idx = min(step_idx, len(ants_steps[ant_idx]) - 1)
        partial_L = ants_steps[ant_idx][s_idx]['partial_len']
        partial_L_str = f"{partial_L:<9.3f}"
        line_str = f"{ant_idx:<5}{final_L_str:<12}{partial_L_str:<12}"
        ax.text(0, 0.90 - 0.05 * i, line_str, fontsize=9, va='top',
                color=cmap(ant_idx), fontfamily='monospace')
    if m > ANTS_TO_DISPLAY:
        ax.text(0.5, 0.0, "(Scroll mouse wheel to see more)",
                ha='center', fontsize=8, style='italic')

def update_plot(val):
    """Main update function for interactive plots."""
    """ฟังก์ชันหลักสำหรับอัปเดตกราฟแบบโต้ตอบ"""
    iter_idx = int(slider_iter.val)
    step_idx = int(slider_step.val)
    draw_graph_on_ax(ax_std, f"Standard ACO (Iter {iter_idx})",
                     all_rounds_std[iter_idx], step_idx)
    draw_graph_on_ax(ax_elite, f"Elitist ACO (Iter {iter_idx})",
                     all_rounds_elite[iter_idx], step_idx)
    update_text_panel(ax_std_text, all_rounds_std[iter_idx],
                      step_idx, scroll_offset_std)
    update_text_panel(ax_elite_text, all_rounds_elite[iter_idx],
                      step_idx, scroll_offset_elite)
    plt.draw()

slider_iter.on_changed(update_plot)
slider_step.on_changed(update_plot)
update_plot(0)
plt.show() # Display the interactive plots / แสดงผลกราฟแบบโต้ตอบ

# =================================================================
# COMPARATIVE SUMMARY PLOTS / กราฟสรุปผลการเปรียบเทียบ
# =================================================================
fig_comp, axs_comp = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
fig_comp.suptitle("Performance Comparison by Elite Weight", fontsize=16)
iters_range = np.arange(1, iters + 1)
colors = plt.cm.viridis(np.linspace(0, 1, len(elite_weights_to_compare)))

for i, ew in enumerate(elite_weights_to_compare):
    key = f"Elite_Weight_{ew}"
    if ew == 0.0:
        key = "Standard_ACO"
    
    # Calculate performance metrics for each elite weight
    # คำนวณเมตริกประสิทธิภาพสำหรับแต่ละน้ำหนัก Elite
    gbest_lens = [
        tour_length_from_order(r['gbest_tour_this_iter'])
        for r in all_results[key]]
    rms_lens = [
        np.std(r['final_lengths']) if r['final_lengths'] else 0
        for r in all_results[key]]

    branch_counts = []
    for r in all_results[key]:
        tau_after = r['tau_after']
        tau_mean = np.mean(tau_after[np.nonzero(tau_after)])
        branches = np.sum(tau_after > tau_mean) / 2
        branch_counts.append(branches / n)

    # Plot on each subplot
    # พล็อตลงบนแต่ละกราฟย่อย
    axs_comp[0].plot(iters_range, gbest_lens, color=colors[i],
                     label=f'Elite Weight = {ew}')
    axs_comp[1].plot(iters_range, rms_lens, color=colors[i],
                     label=f'Elite Weight = {ew}')
    axs_comp[2].plot(iters_range, branch_counts, color=colors[i],
                     label=f'Elite Weight = {ew}')

axs_comp[0].set_title('Solution dynamics')
axs_comp[0].set_ylabel('Length')
axs_comp[0].grid(True)
axs_comp[1].set_title('Solution scattering')
axs_comp[1].set_ylabel('RMS')
axs_comp[1].grid(True)
axs_comp[2].set_title('Average number of pheromone trail branches')
axs_comp[2].set_ylabel('Number of branches')
axs_comp[2].set_xlabel('Iteration Number')
axs_comp[2].grid(True)

lines, labels = axs_comp[0].get_legend_handles_labels()
fig_comp.legend(lines, labels, loc='upper center',
                bbox_to_anchor=(0.5, 0.95), ncol=len(elite_weights_to_compare))

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show() # Display the summary plots / แสดงผลกราฟสรุป