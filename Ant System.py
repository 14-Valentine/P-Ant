# -----------------------------------------------
# Ant System (ACO) for TSP — แสดง "สรุปต่อรอบ" เท่านั้น (Round-only summary)
# ครอบคลุมสมการ: (Covers these equations)
# (1) P_ij,k(t) ∝ (tau_ij)^alpha * (eta_ij)^beta  (transition probability)
# (Δτ) Δtau_ij,k(t) = Q / L_k(t)                  (pheromone deposit on tour edges)
# (2) tau_ij(t+1) = (1 - rho) * tau_ij(t) + Σ_k Δtau_ij,k(t)  (evaporation + update)
# -----------------------------------------------

import random  # ใช้สุ่ม (random choices)
# ---------- ปัญหา: เมทริกซ์ระยะทาง (0 บนแนวทแยง) (Problem: distance matrix; 0 on diagonal) ----------
D = [
    [0, 12, 10, 19,  8],
    [12, 0,  3,  7,  2],
    [10, 3,  0,  6,  4],
    [19, 7,  6,  0, 11],
    [8,  2,  4, 11,  0],
]
n = len(D)        # จำนวนเมือง (number of cities)
m = n             # จำนวนมด (number of ants) — ตั้งเท่าจำนวนเมือง (set = n)

# ---------- พารามิเตอร์ ACO (ACO parameters) ----------
alpha = 0.5       # กำลังของฟีโรโมน (pheromone exponent)
beta  = 0.5       # กำลังของความมองเห็น 1/d (visibility exponent)
rho   = 0.5       # อัตราการระเหย (evaporation rate)
Q     = 1.0       # ค่าคงที่ฝากฟีโรโมน (deposit constant)
tau0  = 1e-3      # ฟีโรโมนตั้งต้น (initial pheromone)
iters = 10        # จำนวนรอบ (number of iterations to show)
PRINT_ANTS = m    # พิมพ์ทัวร์กี่ตัวต่อรอบ (how many ants to print per round)

random.seed(42)   # เพื่อผลสุ่มซ้ำได้ (reproducibility)

# ---------- ความมองเห็น eta = 1 / distance (visibility) ----------
eta = [[0.0]*n for _ in range(n)]
for i in range(n):
    for j in range(n):
        eta[i][j] = 1.0 / D[i][j] if D[i][j] > 0 else 0.0  # เมืองเดียวกัน = 0 (same city = 0)

# ---------- ฟีโรโมนตั้งต้น (initial pheromone) ----------
tau = [[tau0]*n for _ in range(n)]
for i in range(n):
    tau[i][i] = 0.0  # แนวทแยงไม่ใช้งาน (no self-edge)

# ---------- เครื่องมือช่วย (helpers) ----------
def tour_length(tour):
    """ความยาวทัวร์ (รวมปิดวงกลับเมืองแรก) (tour length incl. return to start)"""
    s = 0.0
    for k in range(n):
        i, j = tour[k], tour[(k+1) % n]
        s += D[i][j]
    return s

def transition_probs(current, unvisited):
    """สมการ (1): ความน่าจะเป็นเลือกเมืองถัดไป (Eq.1: transition probabilities)"""
    nums = []
    for j in unvisited:
        nums.append((tau[current][j] ** alpha) * (eta[current][j] ** beta))
    denom = sum(nums)
    if denom == 0.0:
        return [1.0/len(unvisited)] * len(unvisited)  # กรณีพิเศษ แบ่งเท่าๆ กัน (uniform fallback)
    return [x/denom for x in nums]

def construct_tour():
    """สร้างทัวร์ของมด 1 ตัวด้วยกฎรูเล็ตจากสมการ (1) (build one ant's tour via roulette from Eq.1)"""
    start = random.randrange(n)  # สุ่มเมืองเริ่ม (random start city)
    tour = [start]
    unvisited = [j for j in range(n) if j != start]
    current = start
    while unvisited:
        probs = transition_probs(current, unvisited)               # คำนวณ P จากสมการ (1) (compute Eq.1 probs)
        next_city = random.choices(unvisited, weights=probs, k=1)[0]  # เลือกตามน้ำหนัก (roulette)
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    return tour, tour_length(tour)

def delta_tau_from_tour(tour, L):
    """สมการ Δτ: ฝาก Q/L บนขอบทัวร์ (Eq. Δτ: deposit Q/L on tour edges)"""
    delta = [[0.0]*n for _ in range(n)]
    dep = Q / L
    for k in range(n):
        i, j = tour[k], tour[(k+1) % n]
        delta[i][j] += dep
        delta[j][i] += dep   # สมมาตร (symmetric)
    return delta

def add_inplace(A, B):
    """A += B สำหรับเมทริกซ์ n×n (in-place matrix add)"""
    for i in range(n):
        for j in range(n):
            A[i][j] += B[i][j]

def update_pheromone(sum_delta):
    """สมการ (2): ระเหย + บวก Δτ รวม (Eq.2: evaporation + add total Δτ)"""
    for i in range(n):
        for j in range(n):
            if i == j:
                tau[i][j] = 0.0
            else:
                tau[i][j] = (1.0 - rho) * tau[i][j] + sum_delta[i][j]

# ---------- ลูปหลัก: แสดง "สรุปต่อรอบ" (main loop: round-only summary) ----------
best_tour, best_len = None, float("inf")

for t in range(iters):
    sum_delta = [[0.0]*n for _ in range(n)]
    round_tours = []  # เก็บ (tour, L) ของทุกมดในรอบนี้ (store all ants' tours, L)

    for k in range(m):
        tour, L = construct_tour()
        round_tours.append((tour, L))
        add_inplace(sum_delta, delta_tau_from_tour(tour, L))  # รวม Δτ (accumulate Δτ)
        if L < best_len:
            best_tour, best_len = tour, L

    update_pheromone(sum_delta)  # อัปเดต τ ตามสมการ (2) (update τ via Eq.2)

    # ---------- สรุป “เฉพาะรอบนี้ไปไหนบ้าง” (round summary: where ants went) ----------
    print(f"\n[ITER {t}] Round summary")
    for k, (tour, L) in enumerate(round_tours[:PRINT_ANTS]):
        print(f"  Ant {k}: tour={tour}  length={L:.2f}")
    # ใครดีที่สุดในรอบนี้ (best within this round)
    bt_idx = min(range(m), key=lambda i: round_tours[i][1])
    print(f"  ➜ Best this round: Ant {bt_idx}  length={round_tours[bt_idx][1]:.2f}")
# ---------- ผลลัพธ์ท้ายสุด (final results) ----------
print("\n Global best overall:")
print("  Best tour, 0-indexed:", best_tour)
print("  Best length:", best_len)
