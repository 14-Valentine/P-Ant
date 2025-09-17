import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# =================================================================================
# PART 1: ALGORITHM CONFIGURATION (ส่วนที่ 1: การตั้งค่าพารามิเตอร์)
# =================================================================================

# --- Core ACO Parameters | พารามิเตอร์หลักของ ACO ---
N_CITIES = 20                   # Number of cities (nodes) | จำนวนเมือง (โหนด) ในปัญหา
N_ANTS = N_CITIES               # Number of ants in the colony | จำนวนมดในฝูง
N_ITERATIONS = 100              # Number of algorithm iterations | จำนวนรอบการทำงานของอัลกอริทึม

ALPHA = 0.5                     # The influence of the pheromone trail (τ^α) | อิทธิพลของฟีโรโมน (τ^α)
BETA = 0.5                      # The influence of heuristic information (η^β) | อิทธิพลของข้อมูลฮิวริสติก (η^β)
RHO = 0.5                       # Pheromone evaporation rate (ρ) | อัตราการระเหยของฟีโรโมน (ρ)
Q = float(N_CITIES)             # Pheromone deposit constant | ค่าคงที่ในการวางฟีโรโมน
TAU0 = 1.0                      # Initial pheromone level | ระดับฟีโรโมนเริ่มต้นบนทุกเส้นทาง

# --- Experiment Parameters | พารามิเตอร์สำหรับการทดลอง ---
# We will compare standard ACO (elite_weight=0) with EAS using different weights.
# เราจะเปรียบเทียบ ACO แบบมาตรฐาน (elite_weight=0) กับ EAS ที่มีค่าน้ำหนักแตกต่างกัน
ELITE_WEIGHTS_TO_COMPARE = [0.0, 3.0, 5.0]

# --- Reproducibility | การกำหนดค่าเริ่มต้นเพื่อผลลัพธ์ที่ตรงกัน ---
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =================================================================================
# PART 2: DATA SETUP & HELPER FUNCTIONS (ส่วนที่ 2: การเตรียมข้อมูลและฟังก์ชันเสริม)
# =================================================================================

# --- Problem Definition: TSP Instance | การสร้างข้อมูลสำหรับปัญหา TSP ---
# Generate random coordinates for N cities in a 2D space. This is a Euclidean TSP.
# สร้างพิกัดแบบสุ่มสำหรับเมืองต่างๆ ในพื้นที่ 2 มิติ ซึ่งเป็นปัญหา TSP แบบยุคลิด
POSITIONS = np.random.rand(N_CITIES, 2)

# --- Distance Matrix (D_ij) Calculation | การคำนวณเมทริกซ์ระยะทาง (D_ij) ---
# This matrix stores the Euclidean distance between every pair of cities (i, j).
# เมทริกซ์นี้เก็บค่าระยะทางแบบยุคลิดระหว่างเมืองทุกคู่ (i, j) 
DISTANCE_MATRIX = np.zeros((N_CITIES, N_CITIES))
for i in range(N_CITIES):
    for j in range(N_CITIES):
        DISTANCE_MATRIX[i, j] = np.linalg.norm(POSITIONS[i] - POSITIONS[j]) 

# --- Heuristic Matrix (η_ij) Calculation | การคำนวณเมทริกซ์ฮิวริสติก (η_ij) ---
# This matrix stores the "visibility" or heuristic desirability of moving from city i to j.
# For TSP, this is commonly defined as the inverse of the distance (1/D_ij).
# เมทริกซ์นี้เก็บค่า "การมองเห็น" หรือความน่าสนใจเชิงฮิวริสติกในการเดินทางจากเมือง i ไป j
# สำหรับปัญหา TSP ค่านี้มักจะกำหนดเป็นส่วนกลับของระยะทาง (1/D_ij)
HEURISTIC_MATRIX = np.zeros((N_CITIES, N_CITIES))
for i in range(N_CITIES):
    for j in range(N_CITIES):
        if DISTANCE_MATRIX[i, j] > 0:
            HEURISTIC_MATRIX[i, j] = 1.0 / DISTANCE_MATRIX[i, j]

# --- Objective Function | ฟังก์ชันเป้าหมาย ---
# Calculates the total length of a given tour. This is the value we want to minimize.
# คำนวณความยาวรวมของเส้นทางที่กำหนด ซึ่งเป็นค่าที่เราต้องการทำให้ต่ำที่สุด
def get_tour_length(tour_order):
    if not tour_order or len(tour_order) != N_CITIES: return float('inf')
    return sum(DISTANCE_MATRIX[tour_order[k], tour_order[(k + 1) % N_CITIES]] for k in range(N_CITIES))

# --- Ant's Decision-Making: Transition Probability | การตัดสินใจของมด: ความน่าจะเป็นในการเลือกเส้นทาง ---
# This function implements the core probabilistic transition rule of ACO.
# P_ij(t) = [τ_ij(t)]^α * [η_ij]^β / Σ([τ_ik(t)]^α * [η_ik]^β) for all allowed cities k.
# ฟังก์ชันนี้คือหัวใจของกฎการเลือกเส้นทางของ ACO ตามสมการข้างต้น
def calculate_transition_probs(current_city, allowed_cities, pheromone_snapshot):
    # Numerator part: (τ^α * η^β) | ส่วนของตัวเศษ
    numerators = []
    for j in allowed_cities:
        pheromone_level = pheromone_snapshot[current_city, j]
        heuristic_value = HEURISTIC_MATRIX[current_city, j]
        numerators.append((pheromone_level ** ALPHA) * (heuristic_value ** BETA))

    # Denominator part: Σ(τ^α * η^β) | ส่วนของตัวส่วน
    denominator = sum(numerators)
    
    # Avoid division by zero | ป้องกันการหารด้วยศูนย์
    if denominator == 0.0:
        return [1.0 / len(allowed_cities)] * len(allowed_cities) if allowed_cities else []
    
    # Return the final probabilities | คืนค่าความน่าจะเป็นของแต่ละเส้นทาง
    return [num / denominator for num in numerators]

# =================================================================================
# PART 3: MAIN SIMULATION FUNCTION (ส่วนที่ 3: ฟังก์ชันหลักสำหรับรันการจำลอง)
# =================================================================================

def run_aco_simulation(elite_weight_val):
    # --- Initialization | การตั้งค่าเริ่มต้น ---
    pheromone_matrix = np.full((N_CITIES, N_CITIES), TAU0)
    np.fill_diagonal(pheromone_matrix, 0)
    
    all_iterations_data = []
    global_best_length, global_best_tour = float('inf'), None

    # --- Main Algorithm Loop | ลูปหลักของอัลกอริทึม ---
    for it in range(N_ITERATIONS):
        pheromone_snapshot = pheromone_matrix.copy()
        iteration_data = {"tours": [], "tour_lengths": [], "ants_steps": []}

        # --- Phase 1: Tour Construction | เฟสที่ 1: การสร้างเส้นทางของมดแต่ละตัว ---
        for _ in range(N_ANTS):
            # Each ant starts at a random city | มดแต่ละตัวเริ่มจากเมืองสุ่ม
            start_city = np.random.randint(0, N_CITIES)
            visited_cities = [start_city]
            edges, partial_len, steps = [], 0.0, [{"visited": [start_city], "edges": [], "partial_len": 0.0}]

            # Sequentially build a tour by choosing the next city | สร้างเส้นทางไปทีละขั้น
            while len(visited_cities) < N_CITIES:
                current = visited_cities[-1]
                allowed = [j for j in range(N_CITIES) if j not in visited_cities]
                
                # Use the transition rule to decide the next move | ใช้กฎความน่าจะเป็นเพื่อตัดสินใจเลือกเมืองถัดไป
                probs = calculate_transition_probs(current, allowed, pheromone_snapshot)
                next_city = random.choices(allowed, weights=probs, k=1)[0]
                
                edges.append((current, next_city))
                partial_len += DISTANCE_MATRIX[current, next_city]
                visited_cities.append(next_city)
                steps.append({"visited": visited_cities.copy(), "edges": edges.copy(), "partial_len": partial_len})

            # Complete the tour by returning to the start city | กลับสู่เมืองเริ่มต้นเพื่อปิดเส้นทาง
            edges.append((visited_cities[-1], visited_cities[0]))
            partial_len += DISTANCE_MATRIX[visited_cities[-1], visited_cities[0]]
            steps.append({"visited": visited_cities.copy(), "edges": edges.copy(), "partial_len": partial_len})
            
            # Store the completed tour and its length | บันทึกเส้นทางและความยาว
            iteration_data["tours"].append(visited_cities)
            iteration_data["tour_lengths"].append(partial_len)
            
            # Update the global best solution if this ant found a better one
            # อัปเดตคำตอบที่ดีที่สุด (global best) หากมดตัวนี้ค้นพบเส้นทางที่ดีกว่า
            if partial_len < global_best_length:
                global_best_length, global_best_tour = partial_len, visited_cities.copy()
        
        # --- Phase 2: Pheromone Update | เฟสที่ 2: การอัปเดตฟีโรโมน ---
        
        # 2a. Pheromone Evaporation: (1 - ρ) * τ | การระเหยของฟีโรโมน
        # All trails are weakened to encourage exploration and avoid stagnation.
        # ฟีโรโมนบนทุกเส้นทางจะลดลง เพื่อกระตุ้นให้เกิดการสำรวจและป้องกันการติดគាំង
        pheromone_matrix *= (1.0 - RHO)
        
        # 2b. Standard Pheromone Deposit: Δτ = Q / L_k | การวางฟีโรโมนของมดทุกตัว
        # Each ant deposits pheromones on its tour, with shorter tours depositing more.
        # มดทุกตัวจะวางฟีโรโมนบนเส้นทางของตนเอง โดยเส้นทางที่สั้นกว่าจะมีการวางฟีโรโมนมากกว่า
        for tour, length in zip(iteration_data["tours"], iteration_data["tour_lengths"]):
            deposit_amount = Q / length
            for k in range(N_CITIES):
                c1, c2 = tour[k], tour[(k+1) % N_CITIES]
                pheromone_matrix[c1, c2] += deposit_amount
                pheromone_matrix[c2, c1] += deposit_amount
        
        # 2c. Elitist Pheromone Deposit (The core of EAS) | การวางฟีโรโมนพิเศษ (หัวใจของ EAS)
        # The best-so-far tour gets an extra pheromone boost to intensify the search around it.
        # This implements the formula: Δτ_elite = e * (Q / L_best)
        # เส้นทางที่ดีที่สุดที่เคยพบจะได้รับฟีโรโมนเพิ่มเติมเป็นพิเศษ เพื่อเน้นการค้นหาในบริเวณนั้นให้เข้มข้นขึ้น
        if elite_weight_val > 0 and global_best_tour:
            elite_deposit = elite_weight_val * (Q / global_best_length)
            for k in range(N_CITIES):
                c1, c2 = global_best_tour[k], global_best_tour[(k+1) % N_CITIES]
                pheromone_matrix[c1, c2] += elite_deposit
                pheromone_matrix[c2, c1] += elite_deposit

        # --- Data Logging for Visualization | การบันทึกข้อมูลเพื่อนำไปแสดงผล ---
        all_iterations_data.append({
            "gbest_tour_this_iter": global_best_tour,
            "final_lengths": iteration_data["tour_lengths"],
            "ants_steps": iteration_data["ants_steps"],
            "tau_after": pheromone_matrix.copy()
        })
        
        if (it + 1) % 10 == 0: 
            print(f"  [Iteration {it+1}/{N_ITERATIONS}] Current Best Length: {global_best_length:.4f}")
            
    return all_iterations_data

# =================================================================================
# PART 4: EXECUTION & PLOTTING (ส่วนที่ 4: การรันการทดลองและแสดงผล)
# =================================================================================
# This section remains unchanged as its purpose is well-defined.
# It runs the simulation for each elite weight and then generates comparative plots.
# ส่วนนี้ไม่มีการเปลี่ยนแปลง เนื่องจากมีจุดประสงค์ที่ชัดเจนอยู่แล้ว
# คือการรันการจำลองสำหรับค่าน้ำหนัก elite แต่ละค่า แล้วจึงสร้างกราฟเพื่อเปรียบเทียบผลลัพธ์
# =================================================================================

# --- Run the experiment for each configured elite weight | รันการทดลองสำหรับ Elite Weight แต่ละค่า ---
all_results = {}
for ew in ELITE_WEIGHTS_TO_COMPARE:
    key = "Standard_ACO" if ew == 0.0 else f"Elite_Weight_{ew}"
    print(f"\n--- Running ACO with {key.replace('_', ' ')} ---")
    all_results[key] = run_aco_simulation(ew)
    final_best_len = get_tour_length(all_results[key][-1]["gbest_tour_this_iter"])
    print(f"\nGLOBAL BEST FOUND: Length = {final_best_len:.4f}")

# --- Create comparative summary plots | สร้างกราฟสรุปผลการเปรียบเทียบ ---
fig_comp, axs_comp = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
fig_comp.suptitle("Performance Comparison by Elite Weight", fontsize=16)
iters_range = np.arange(1, N_ITERATIONS + 1)
colors = cm.plasma(np.linspace(0, 1, len(ELITE_WEIGHTS_TO_COMPARE)))

for i, ew in enumerate(ELITE_WEIGHTS_TO_COMPARE):
    key = "Standard_ACO" if ew == 0.0 else f"Elite_Weight_{ew}"
    
    # Plot 1: Global Best Solution Convergence | กราฟที่ 1: การลู่เข้าของคำตอบที่ดีที่สุด
    gbest_lens = [get_tour_length(r['gbest_tour_this_iter']) for r in all_results[key]]
    
    # Plot 2: Solution Diversity (Standard Deviation) | กราฟที่ 2: ความหลากหลายของคำตอบ
    std_dev_lens = [np.std(r['final_lengths']) if r['final_lengths'] else 0 for r in all_results[key]]
    
    # Plot 3: Pheromone Trail Exploration | กราฟที่ 3: การสำรวจผ่านร่องรอยฟีโรโมน
    branch_counts = []
    for r in all_results[key]:
        tau_after = r['tau_after']
        tau_mean = np.mean(tau_after[np.nonzero(tau_after)])
        branches = np.sum(tau_after > tau_mean) / 2
        branch_counts.append(branches / N_CITIES)
        
    label_text = f'Elite Weight = {ew}'
    axs_comp[0].plot(iters_range, gbest_lens, color=colors[i], label=label_text)
    axs_comp[1].plot(iters_range, std_dev_lens, color=colors[i], label=label_text)
    axs_comp[2].plot(iters_range, branch_counts, color=colors[i], label=label_text)

axs_comp[0].set_title('Global Best Solution Convergence'); axs_comp[0].set_ylabel('Tour Length'); axs_comp[0].grid(True, linestyle='--', alpha=0.6)
axs_comp[1].set_title('Solution Diversity (Std. Dev.)'); axs_comp[1].set_ylabel('Std. Dev.'); axs_comp[1].grid(True, linestyle='--', alpha=0.6)
axs_comp[2].set_title('Pheromone Trail Exploration'); axs_comp[2].set_ylabel('Avg. Branches per City'); axs_comp[2].set_xlabel('Iteration'); axs_comp[2].grid(True, linestyle='--', alpha=0.6)
lines, labels = axs_comp[0].get_legend_handles_labels()
fig_comp.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(ELITE_WEIGHTS_TO_COMPARE))
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()