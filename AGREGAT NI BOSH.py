
# =============================================================================
# BAGIAN 0 — IMPORT LIBRARY
# =============================================================================

from pulp import (
    LpProblem,       # membuat objek model LP
    LpMinimize,      # menentukan arah optimasi: minimisasi
    LpVariable,      # mendefinisikan variabel keputusan
    lpSum,           # menjumlahkan ekspresi LP secara efisien
    value,           # mengambil nilai numerik dari variabel LP setelah solve
    PULP_CBC_CMD,    # solver CBC (bawaan PuLP, tidak perlu instalasi terpisah)
    LpStatus,        # konversi kode status solver ke string ("Optimal", dst.)
    LpStatusNotSolved,
)
import pandas as pd           # untuk membuat tabel output (DataFrame)
import numpy as np            # untuk operasi numerik dan sampling
from scipy.stats import norm  # untuk menghitung z-score safety stock
from dataclasses import dataclass, field  # untuk struktur data parameter
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")  # sembunyikan warning minor dari solver


# =============================================================================
# BAGIAN 1 — DATACLASS PARAMETER
# Menggunakan dataclass agar parameter mudah dikirim dari Streamlit sidebar
# ke fungsi solver tanpa harus mengoper banyak argumen terpisah.
# =============================================================================

@dataclass
class CostParams:
    """
    Semua parameter biaya yang diinput user di Streamlit sidebar.
    Satuan: Rupiah (Rp)
    """
    rt_labor_cost:    float  # biaya TK regular per orang per periode
    ot_labor_cost:    float  # biaya lembur per JAM per orang
    hiring_cost:      float  # biaya rekrut + pelatihan per orang baru
    firing_cost:      float  # biaya PHK + pesangon per orang
    inventory_cost:   float  # holding cost produk jadi per unit per periode
    stockout_cost:    float  # penalti lost sales per unit yang tidak terpenuhi
    material_cost:    float  # biaya bahan baku per unit produksi output
    subcon_cost:      float  # biaya subcontracting per unit ke mitra eksternal
    rm_holding_cost:  float  # holding cost bahan baku per unit per periode
    rm_ordering_cost: float  # biaya per kali memesan bahan baku (fixed order cost)
    rm_shortage_cost: float  # penalti per unit bahan baku yang kurang saat produksi


@dataclass
class CapacityParams:
    """
    Parameter kapasitas produksi dan tenaga kerja.
    """
    prod_time:          float  # jam kerja per unit produksi (jam/unit)
    rt_hours:           float  # jam regular per orang per periode (jam/orang/periode)
    ot_max_per_worker:  float  # maks jam lembur per orang per periode (jam)
    capacity_max:       float  # kapasitas mesin/fasilitas maks per periode (unit)
    W_min:              int    # jumlah TK minimum yang harus dipertahankan
    subcon_max:         float  # kapasitas subcontracting maks per periode (unit)
    rm_per_unit:        float  # kebutuhan bahan baku per unit output (unit RM/unit FG)


@dataclass
class InitialConditions:
    """
    Kondisi awal periode 0 (sebelum horizon perencanaan dimulai).
    """
    W0:            int    # jumlah TK awal (orang)
    I0_fg:         float  # inventory produk jadi awal (unit)
    I0_rm:         float  # inventory bahan baku awal (unit RM)
    RM_on_order0:  float  # RM yang sudah dipesan tapi belum tiba (unit RM)


@dataclass
class SupplyParams:
    """
    Parameter untuk perhitungan safety stock dan lead time bahan baku.
    """
    lt_mean:          float  # rata-rata lead time (periode, bukan minggu)
    lt_std:           float  # standar deviasi lead time (periode)
    service_level_rm: float  # target service level bahan baku (desimal, 0–1)
    rm_order_qty:     float  # kuantitas order bahan baku per pemesanan (unit RM)


@dataclass
class DemandScenario:
    """
    Tiga skenario demand untuk satu horizon perencanaan.
    Setiap skenario adalah list dengan panjang T (jumlah periode).
    """
    moderate:    List[float]  # demand skenario moderat (baseline forecast)
    optimistic:  List[float]  # demand skenario optimis
    pessimistic: List[float]  # demand skenario pesimis
    prob_opt:    float = 0.25  # probabilitas skenario optimis
    prob_mod:    float = 0.50  # probabilitas skenario moderat
    prob_pes:    float = 0.25  # probabilitas skenario pesimis

    def __post_init__(self):
        """Validasi: jumlah probabilitas harus = 1.0"""
        total = self.prob_opt + self.prob_mod + self.prob_pes
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Probabilitas ketiga skenario harus berjumlah 1.0, "
                f"saat ini: {total:.4f}"
            )

    @property
    def T(self) -> int:
        """Jumlah periode = panjang list demand moderat."""
        return len(self.moderate)

    def get(self, scenario: str) -> List[float]:
        """Ambil list demand berdasarkan nama skenario."""
        return {
            "optimistic": self.optimistic,
            "moderate":   self.moderate,
            "pessimistic": self.pessimistic,
        }[scenario]

    def prob(self, scenario: str) -> float:
        """Ambil probabilitas skenario."""
        return {
            "optimistic":  self.prob_opt,
            "moderate":    self.prob_mod,
            "pessimistic": self.prob_pes,
        }[scenario]


# =============================================================================
# BAGIAN 2 — PERHITUNGAN SAFETY STOCK DAN REORDER POINT
# Fungsi ini berjalan di luar LP solver — hasilnya dipakai sebagai
# constraint minimum inventory bahan baku di dalam model LP.
# =============================================================================

def compute_safety_stock(
    supply: SupplyParams,
    demand_avg: float,
    demand_std: float,
) -> Dict[str, float]:
    """
    Hitung safety stock bahan baku menggunakan formula statistik standar.

    Formula:
        SS = z × √(LT_mean × σ_D² + D_avg² × σ_LT²)

    di mana:
        z       = z-score dari target service level (dari distribusi normal)
        σ_D     = standar deviasi demand per periode
        D_avg   = rata-rata demand per periode
        σ_LT    = standar deviasi lead time

    Formula ini menggabungkan dua sumber variabilitas:
      - variabilitas demand (LT_mean × σ_D²): demand berfluktuasi selama lead time
      - variabilitas lead time (D_avg² × σ_LT²): lead time sendiri tidak pasti

    Referensi: Silver, Pyke & Peterson (1998), "Inventory Management and
    Production Planning and Scheduling", Chapter 7.

    Parameters
    ----------
    supply      : SupplyParams — parameter lead time dan service level
    demand_avg  : float — rata-rata demand per periode
    demand_std  : float — standar deviasi demand per periode

    Returns
    -------
    dict dengan kunci 'safety_stock', 'reorder_point', 'z_score'
    """
    # z-score berdasarkan target service level
    # contoh: service_level=0.95 → z=1.645; service_level=0.99 → z=2.326
    z = norm.ppf(supply.service_level_rm)

    # komponen variabilitas demand selama lead time
    var_demand_during_lt = supply.lt_mean * (demand_std ** 2)

    # komponen variabilitas lead time itu sendiri
    var_lt = (demand_avg ** 2) * (supply.lt_std ** 2)

    # safety stock total
    safety_stock = z * np.sqrt(var_demand_during_lt + var_lt)
    safety_stock = max(0.0, safety_stock)  # tidak boleh negatif

    # reorder point = demand selama lead time + safety stock
    reorder_point = supply.lt_mean * demand_avg + safety_stock

    return {
        "safety_stock":  round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "z_score":       round(z, 4),
    }


def build_rm_arrival_schedule(
    init: InitialConditions,
    supply: SupplyParams,
    demand_plan: List[float],
    cap: CapacityParams,
    T: int,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Buat jadwal pemesanan dan kedatangan bahan baku per periode.

    Logika:
      - Setiap periode, cek apakah RM_inventory < reorder_point
      - Jika ya, lakukan order sebesar rm_order_qty
      - RM tiba setelah lt_mean periode (dibulatkan ke integer)
      - Output: jadwal RM_arrive, RM_inventory, RM_order per periode

    Parameters
    ----------
    init        : InitialConditions
    supply      : SupplyParams
    demand_plan : List[float] — rencana produksi (dalam unit RM = prod × rm_per_unit)
    cap         : CapacityParams
    T           : int — jumlah periode

    Returns
    -------
    (rm_arrive, rm_inventory, rm_order) — masing-masing list panjang T
    """
    LT = max(1, round(supply.lt_mean))  # lead time dalam periode (integer)

    # demand rata-rata dan std untuk safety stock
    demand_avg = np.mean(demand_plan)
    demand_std = np.std(demand_plan) if len(demand_plan) > 1 else demand_avg * 0.15

    ss_result = compute_safety_stock(supply, demand_avg, demand_std)
    ss = ss_result["safety_stock"]
    rop = ss_result["reorder_point"]

    # inisialisasi array untuk tracking
    rm_inventory = [0.0] * (T + LT + 1)  # padding untuk lead time
    rm_inventory[0] = init.I0_rm
    rm_order   = [0.0] * (T + LT + 1)
    rm_arrive  = [0.0] * (T + LT + 1)

    # bahan baku yang sudah on-order tiba di periode LT
    rm_arrive[min(LT, T)] += init.RM_on_order0

    for t in range(1, T + 1):
        # bahan baku yang dibutuhkan periode ini
        rm_needed = demand_plan[t - 1] * cap.rm_per_unit

        # cek apakah perlu order baru
        # order dilakukan jika inventory saat ini < reorder point
        if rm_inventory[t - 1] + rm_arrive[t] < rop + rm_needed:
            order_qty = supply.rm_order_qty
            rm_order[t] = order_qty
            # order tiba setelah LT periode
            arrive_t = t + LT
            if arrive_t <= T + LT:
                rm_arrive[arrive_t] += order_qty

        # update inventory bahan baku
        available = rm_inventory[t - 1] + rm_arrive[t]
        used      = min(available, rm_needed)
        rm_inventory[t] = available - used

    # potong ke panjang T (buang padding)
    return (
        rm_arrive[1:T + 1],
        rm_inventory[1:T + 1],
        rm_order[1:T + 1],
    )


# =============================================================================
# BAGIAN 3 — MODEL LP UTAMA (SATU SKENARIO)
# Fungsi inti yang membangun dan menyelesaikan model LP untuk satu skenario
# demand. Dipanggil tiga kali (optimis, moderat, pesimis) dari solve_all_scenarios.
# =============================================================================

def solve_single_scenario(
    scenario_name: str,
    demand: List[float],
    cost: CostParams,
    cap: CapacityParams,
    init: InitialConditions,
    supply: SupplyParams,
    strategy: str = "mixed",
) -> Dict:
    """
    Bangun dan selesaikan model LP Aggregate Planning untuk SATU skenario demand.

    Strategy options:
      "chase"  — produksi mengikuti demand, workforce berfluktuasi bebas
      "level"  — produksi konstan (rata-rata demand), inventory menyerap fluktuasi
      "mixed"  — LP bebas memilih kombinasi optimal (tidak ada constraint strategi)

    Parameters
    ----------
    scenario_name : str — nama skenario ("optimistic"/"moderate"/"pessimistic")
    demand        : List[float] — demand per periode untuk skenario ini
    cost          : CostParams
    cap           : CapacityParams
    init          : InitialConditions
    supply        : SupplyParams
    strategy      : str — pilihan strategi AP

    Returns
    -------
    dict berisi status solver, total cost, dan DataFrame hasil per periode
    """
    T = len(demand)  # jumlah periode dalam horizon perencanaan

    # -------------------------------------------------------------------------
    # STEP 3.1 — hitung ketersediaan bahan baku per periode
    # Ini dilakukan sebelum LP karena RM_available menjadi constraint di dalam LP.
    # Gunakan demand sebagai proxy rencana produksi untuk pertama kali.
    # -------------------------------------------------------------------------
    rm_arrive, rm_inventory_pre, rm_order = build_rm_arrival_schedule(
        init, supply, demand, cap, T
    )

    # batas produksi dari ketersediaan RM per periode
    # jika RM tidak cukup, produksi internal dibatasi oleh RM yang ada
    rm_cap = [
        (init.I0_rm + rm_arrive[0]) / cap.rm_per_unit if cap.rm_per_unit > 0 else 1e9
    ]
    rm_running = init.I0_rm + rm_arrive[0] - min(demand[0], (init.I0_rm + rm_arrive[0]) / cap.rm_per_unit) * cap.rm_per_unit if cap.rm_per_unit > 0 else init.I0_rm

    for t in range(1, T):
        rm_avail_t = rm_running + rm_arrive[t]
        rm_cap.append(rm_avail_t / cap.rm_per_unit if cap.rm_per_unit > 0 else 1e9)
        rm_running = max(0, rm_avail_t - demand[t] * cap.rm_per_unit)

    # -------------------------------------------------------------------------
    # STEP 3.2 — INISIALISASI MODEL LP
    # LpProblem adalah objek model yang menampung semua variabel dan constraint.
    # LpMinimize berarti kita mencari nilai variabel yang MEMINIMALKAN objective.
    # -------------------------------------------------------------------------
    model_name = f"AP_{scenario_name}_{strategy}"
    prob = LpProblem(name=model_name, sense=LpMinimize)

    # -------------------------------------------------------------------------
    # STEP 3.3 — DEFINISI VARIABEL KEPUTUSAN
    # LpVariable(name, lowBound, upBound) mendefinisikan satu variabel.
    # lowBound=0 berarti variabel tidak boleh negatif (non-negativity constraint).
    # Indeks t=0 berarti periode 1, t=1 berarti periode 2, dst.
    # -------------------------------------------------------------------------

    # P[t] = jumlah unit yang diproduksi secara internal pada periode t
    # Batas atas: minimum dari (kapasitas mesin) dan (kapasitas dari RM tersedia)
    P = [
        LpVariable(
            name=f"P_{t+1}",
            lowBound=0,
            upBound=min(cap.capacity_max, rm_cap[t]),  # dibatasi RM dan mesin
        )
        for t in range(T)
    ]

    # W[t] = jumlah tenaga kerja aktif pada periode t (orang)
    # lowBound=W_min memastikan tidak boleh di bawah minimum workforce
    W = [
        LpVariable(name=f"W_{t+1}", lowBound=cap.W_min)
        for t in range(T)
    ]

    # H[t] = tenaga kerja yang DIREKRUT pada periode t (hiring)
    H = [LpVariable(name=f"H_{t+1}", lowBound=0) for t in range(T)]

    # F[t] = tenaga kerja yang di-PHK pada periode t (firing)
    F = [LpVariable(name=f"F_{t+1}", lowBound=0) for t in range(T)]

    # OT[t] = total jam lembur yang digunakan pada periode t (jam total)
    # Batas atas: semua pekerja lembur maksimal (W[t] × ot_max_per_worker)
    # Karena W[t] adalah variabel LP, batas ini di-set sebagai constraint, bukan upBound
    OT = [LpVariable(name=f"OT_{t+1}", lowBound=0) for t in range(T)]

    # SC[t] = jumlah unit yang disubcontract pada periode t
    SC = [
        LpVariable(name=f"SC_{t+1}", lowBound=0, upBound=cap.subcon_max)
        for t in range(T)
    ]

    # I[t] = inventory produk jadi pada akhir periode t (unit)
    I = [LpVariable(name=f"I_{t+1}", lowBound=0) for t in range(T)]

    # SO[t] = stockout pada periode t — unit demand yang tidak terpenuhi
    SO = [LpVariable(name=f"SO_{t+1}", lowBound=0) for t in range(T)]

    # -------------------------------------------------------------------------
    # STEP 3.4 — OBJECTIVE FUNCTION
    # Total cost = jumlah semua komponen biaya di semua periode.
    # lpSum([...]) bekerja seperti sum() tetapi untuk ekspresi LP.
    # prob += ekspresi berarti "tambahkan ini sebagai objective atau constraint".
    # -------------------------------------------------------------------------
    prob += lpSum([
        # biaya TK regular: Rp/orang × jumlah orang aktif
        cost.rt_labor_cost  * W[t]  +

        # biaya lembur: Rp/jam × total jam lembur
        cost.ot_labor_cost  * OT[t] +

        # biaya rekrut: Rp/orang × jumlah yang direkrut
        cost.hiring_cost    * H[t]  +

        # biaya PHK: Rp/orang × jumlah yang dipecat
        cost.firing_cost    * F[t]  +

        # biaya bahan baku: Rp/unit × total produksi internal
        cost.material_cost  * P[t]  +

        # biaya subcontracting: Rp/unit × unit yang disubcon
        cost.subcon_cost    * SC[t] +

        # holding cost produk jadi: Rp/unit × stok akhir periode
        cost.inventory_cost * I[t]  +

        # penalti stockout: Rp/unit × unit yang tidak terpenuhi
        cost.stockout_cost  * SO[t]

        for t in range(T)
    ]), "Total_Cost_Objective"

    # -------------------------------------------------------------------------
    # STEP 3.5 — CONSTRAINTS (PEMBATAS MODEL)
    # Setiap constraint ditambahkan ke prob dengan operator +=
    # Format: prob += (ekspresi_lp, "nama_constraint")
    # Nama constraint berguna untuk debugging jika solver tidak menemukan solusi.
    # -------------------------------------------------------------------------

    # --- CONSTRAINT 1: Inventory Balance Produk Jadi ---
    # Stok akhir periode t = stok awal + total pasokan - demand
    # Jika pasokan < demand, maka terjadi stockout (SO[t] > 0)
    # Formula: I[t] - SO[t] = I[t-1] + P[t] + SC[t] - D[t]
    for t in range(T):
        prev_I = init.I0_fg if t == 0 else I[t - 1]  # stok periode sebelumnya
        prob += (
            I[t] - SO[t] == prev_I + P[t] + SC[t] - demand[t],
            f"Inventory_Balance_{t+1}"
        )

    # --- CONSTRAINT 2: Workforce Balance ---
    # Jumlah TK periode t = TK periode lalu + yang direkrut - yang dipecat
    # Formula: W[t] = W[t-1] + H[t] - F[t]
    for t in range(T):
        prev_W = init.W0 if t == 0 else W[t - 1]
        prob += (
            W[t] == prev_W + H[t] - F[t],
            f"Workforce_Balance_{t+1}"
        )

    # --- CONSTRAINT 3: Kapasitas Produksi Regular Time ---
    # Produksi tidak boleh melebihi kapasitas dari jam kerja regular + lembur
    # Unit RT = (W[t] × rt_hours) / prod_time
    # Unit OT = OT[t] / prod_time
    # Total kapasitas: P[t] ≤ (W[t] × rt_hours + OT[t]) / prod_time
    for t in range(T):
        prob += (
            P[t] <= (W[t] * cap.rt_hours + OT[t]) / cap.prod_time,
            f"Production_Capacity_{t+1}"
        )

    # --- CONSTRAINT 4: Batas Lembur per Pekerja ---
    # Total jam lembur tidak boleh melebihi (jumlah pekerja × maks OT per orang)
    # Sesuai UU Ketenagakerjaan No.13/2003: maks 4 jam/hari atau 18 jam/minggu
    for t in range(T):
        prob += (
            OT[t] <= W[t] * cap.ot_max_per_worker,
            f"OT_Limit_{t+1}"
        )

    # --- CONSTRAINT 5: Strategy-specific Constraints ---
    # Chase: produksi harus sama dengan demand (tidak boleh ada inventory build-up)
    # Level: produksi harus konstan di semua periode
    # Mixed: tidak ada constraint tambahan — LP bebas optimasi
    if strategy == "chase":
        avg_demand = np.mean(demand)
        for t in range(T):
            # Chase strategy: total pasokan (internal + subcon) = demand
            # Inventory dan stockout diminimalkan oleh solver secara natural
            prob += (
                P[t] + SC[t] >= demand[t] * 0.95,  # toleransi 5% untuk fleksibilitas
                f"Chase_Strategy_{t+1}"
            )

    elif strategy == "level":
        # Level strategy: produksi internal konstan di semua periode
        # Variasi demand diserap oleh inventory atau stockout
        avg_demand = sum(demand) / T
        for t in range(T - 1):
            prob += (
                P[t] == P[t + 1],
                f"Level_Strategy_{t+1}"
            )

    # "mixed" tidak perlu constraint tambahan — LP bebas menentukan kombinasi optimal

    # -------------------------------------------------------------------------
    # STEP 3.6 — SELESAIKAN MODEL
    # solver=PULP_CBC_CMD() menggunakan solver CBC bawaan PuLP.
    # msg=0 mematikan output log solver agar tidak mengotori console Streamlit.
    # timeLimit=60 membatasi waktu solver maksimal 60 detik.
    # -------------------------------------------------------------------------
    solver = PULP_CBC_CMD(msg=0, timeLimit=60)
    prob.solve(solver)

    # -------------------------------------------------------------------------
    # STEP 3.7 — EKSTRAK HASIL
    # Setelah solve(), nilai variabel bisa diambil dengan value(var).
    # Jika solver tidak menemukan solusi optimal, kembalikan error message.
    # -------------------------------------------------------------------------
    status = LpStatus[prob.status]

    if prob.status != 1:
        # Status bukan "Optimal" — model infeasible atau waktu habis
        return {
            "scenario":   scenario_name,
            "strategy":   strategy,
            "status":     status,
            "total_cost": None,
            "df_plan":    pd.DataFrame(),
            "df_cost":    pd.DataFrame(),
            "error":      f"Solver tidak menemukan solusi optimal. Status: {status}. "
                          f"Periksa apakah constraint terlalu ketat (misal W_min terlalu "
                          f"tinggi atau kapasitas terlalu rendah).",
        }

    # Ambil nilai numerik setiap variabel setelah optimasi
    # value() mengembalikan float; max(0, ...) mencegah nilai negatif kecil akibat
    # floating point error (misal -1e-10 seharusnya 0)
    P_val  = [max(0.0, value(P[t]))  for t in range(T)]
    W_val  = [max(0.0, value(W[t]))  for t in range(T)]
    H_val  = [max(0.0, value(H[t]))  for t in range(T)]
    F_val  = [max(0.0, value(F[t]))  for t in range(T)]
    OT_val = [max(0.0, value(OT[t])) for t in range(T)]
    SC_val = [max(0.0, value(SC[t])) for t in range(T)]
    I_val  = [max(0.0, value(I[t]))  for t in range(T)]
    SO_val = [max(0.0, value(SO[t])) for t in range(T)]

    # kapasitas yang tersedia dari regular time per periode
    cap_rt_val = [
        W_val[t] * cap.rt_hours / cap.prod_time for t in range(T)
    ]
    # unit tambahan dari overtime
    cap_ot_val = [
        OT_val[t] / cap.prod_time for t in range(T)
    ]
    # utilisasi kapasitas = produksi aktual / kapasitas mesin maks (%)
    util_val = [
        min(100.0, P_val[t] / cap.capacity_max * 100) if cap.capacity_max > 0 else 0
        for t in range(T)
    ]
    # label keputusan make-or-buy per periode
    mob_label = []
    for t in range(T):
        has_ot  = OT_val[t] > 0.5
        has_sc  = SC_val[t] > 0.5
        if has_ot and has_sc:
            mob_label.append("OT + Subcon")
        elif has_ot:
            mob_label.append("Overtime")
        elif has_sc:
            mob_label.append("Subcon")
        else:
            mob_label.append("Internal RT")

    # -------------------------------------------------------------------------
    # STEP 3.8 — BUAT DATAFRAME OUTPUT
    # DataFrame memudahkan Streamlit untuk menampilkan tabel dan grafik.
    # Semua nilai dibulatkan agar tampilan bersih di UI.
    # -------------------------------------------------------------------------

    # Tabel Rencana Produksi
    df_plan = pd.DataFrame({
        "Periode":           range(1, T + 1),
        "Demand (unit)":     [round(d, 0) for d in demand],
        "Kapasitas RT":      [round(v, 1) for v in cap_rt_val],
        "Kapasitas OT":      [round(v, 1) for v in cap_ot_val],
        "Produksi Internal": [round(v, 1) for v in P_val],
        "Subcontracting":    [round(v, 1) for v in SC_val],
        "Total Pasokan":     [round(P_val[t] + SC_val[t], 1) for t in range(T)],
        "Inventory FG":      [round(v, 1) for v in I_val],
        "Stockout":          [round(v, 1) for v in SO_val],
        "TK Aktif":          [round(v, 0) for v in W_val],
        "Hiring":            [round(v, 0) for v in H_val],
        "Firing":            [round(v, 0) for v in F_val],
        "OT (jam)":          [round(v, 1) for v in OT_val],
        "Utilisasi (%)":     [round(v, 1) for v in util_val],
        "Make-or-Buy":       mob_label,
        "RM Tiba":           [round(v, 1) for v in rm_arrive],
        "RM Inventory":      [round(v, 1) for v in rm_inventory_pre],
    })

    # Tabel Breakdown Biaya
    rt_cost   = [cost.rt_labor_cost  * W_val[t]  for t in range(T)]
    ot_cost   = [cost.ot_labor_cost  * OT_val[t] for t in range(T)]
    h_cost    = [cost.hiring_cost    * H_val[t]  for t in range(T)]
    f_cost    = [cost.firing_cost    * F_val[t]  for t in range(T)]
    mat_cost  = [cost.material_cost  * P_val[t]  for t in range(T)]
    sc_cost   = [cost.subcon_cost    * SC_val[t] for t in range(T)]
    inv_cost  = [cost.inventory_cost * I_val[t]  for t in range(T)]
    so_cost   = [cost.stockout_cost  * SO_val[t] for t in range(T)]
    total_t   = [
        rt_cost[t] + ot_cost[t] + h_cost[t] + f_cost[t] +
        mat_cost[t] + sc_cost[t] + inv_cost[t] + so_cost[t]
        for t in range(T)
    ]

    df_cost = pd.DataFrame({
        "Periode":           range(1, T + 1),
        "RT Labor Cost":     [round(v, 0) for v in rt_cost],
        "OT Labor Cost":     [round(v, 0) for v in ot_cost],
        "Hiring Cost":       [round(v, 0) for v in h_cost],
        "Firing Cost":       [round(v, 0) for v in f_cost],
        "Material Cost":     [round(v, 0) for v in mat_cost],
        "Subcon Cost":       [round(v, 0) for v in sc_cost],
        "Inventory Cost":    [round(v, 0) for v in inv_cost],
        "Stockout Cost":     [round(v, 0) for v in so_cost],
        "Total Cost":        [round(v, 0) for v in total_t],
    })

    # Hitung total cost dari objective function
    total_cost = value(prob.objective)

    return {
        "scenario":       scenario_name,
        "strategy":       strategy,
        "status":         status,
        "total_cost":     round(total_cost, 0),
        "df_plan":        df_plan,
        "df_cost":        df_cost,
        "rm_ss_result":   compute_safety_stock(
                            supply,
                            np.mean(demand),
                            np.std(demand) if len(demand) > 1 else np.mean(demand) * 0.15,
                          ),
        "error":          None,
    }


# =============================================================================
# BAGIAN 4 — SOLVER TIGA SKENARIO SEKALIGUS
# Fungsi utama yang dipanggil oleh Streamlit.
# Menjalankan solve_single_scenario untuk ketiga skenario dan
# menghitung analisis robust (expected value + minimax regret).
# =============================================================================

def solve_all_scenarios(
    demand_scenario: DemandScenario,
    cost: CostParams,
    cap: CapacityParams,
    init: InitialConditions,
    supply: SupplyParams,
    strategy: str = "mixed",
) -> Dict:
    """
    Jalankan model LP untuk ketiga skenario demand secara berurutan.
    Hitung expected cost dan analisis robustness.

    Parameters
    ----------
    demand_scenario : DemandScenario — berisi demand ketiga skenario + probabilitas
    cost, cap, init, supply : parameter model (lihat dataclass di atas)
    strategy        : str — "chase", "level", atau "mixed"

    Returns
    -------
    dict lengkap berisi hasil ketiga skenario + analisis komparatif
    """
    scenarios = ["optimistic", "moderate", "pessimistic"]
    results   = {}

    # jalankan solver untuk masing-masing skenario
    for s in scenarios:
        results[s] = solve_single_scenario(
            scenario_name = s,
            demand        = demand_scenario.get(s),
            cost          = cost,
            cap           = cap,
            init          = init,
            supply        = supply,
            strategy      = strategy,
        )

    # -------------------------------------------------------------------------
    # STEP 4.1 — ANALISIS EXPECTED COST
    # Expected cost = Σ(probabilitas × total cost per skenario)
    # Ini adalah nilai yang paling relevan untuk pengambilan keputusan
    # di bawah ketidakpastian (Expected Value Criterion).
    # -------------------------------------------------------------------------
    costs = {s: results[s]["total_cost"] for s in scenarios}
    probs = {s: demand_scenario.prob(s)  for s in scenarios}

    # hanya hitung jika semua skenario berhasil diselesaikan
    all_ok = all(results[s]["status"] == "Optimal" for s in scenarios)

    if all_ok:
        expected_cost = sum(probs[s] * costs[s] for s in scenarios)

        # -------------------------------------------------------------------------
        # STEP 4.2 — MINIMAX REGRET
        # Regret[s] = TC[s] - TC_minimum_jika_tahu_s_pasti_terjadi
        # Karena kita hanya punya satu rencana (bukan adaptive),
        # regret dihitung sebagai: berapa kerugian tambahan jika demand ternyata
        # bukan skenario yang kita optimalkan?
        #
        # Cara sederhana: bandingkan TC di setiap skenario vs TC skenario moderat
        # sebagai benchmark "rencana yang dipilih".
        # -------------------------------------------------------------------------
        tc_values = [costs[s] for s in scenarios]
        min_possible = min(tc_values)  # biaya minimum yang mungkin dicapai
        regrets = {s: costs[s] - min_possible for s in scenarios}
        max_regret = max(regrets.values())

        # -------------------------------------------------------------------------
        # STEP 4.3 — BUAT TABEL PERBANDINGAN ANTAR SKENARIO
        # Tabel ini ditampilkan di Streamlit sebagai summary comparison.
        # -------------------------------------------------------------------------
        df_comparison = pd.DataFrame({
            "Skenario":        ["Optimistis", "Moderat", "Pesimistis"],
            "Probabilitas":    [f"{probs[s]*100:.0f}%" for s in scenarios],
            "Total Cost (Rp)": [f"{costs[s]:,.0f}" for s in scenarios],
            "Regret (Rp)":     [f"{regrets[s]:,.0f}" for s in scenarios],
            "Status":          [results[s]["status"] for s in scenarios],
        })

        # -------------------------------------------------------------------------
        # STEP 4.4 — SUMMARY METRICS
        # Metrik ringkasan untuk ditampilkan sebagai KPI cards di Streamlit.
        # -------------------------------------------------------------------------
        mod = results["moderate"]
        summary_metrics = {
            "expected_cost":      round(expected_cost, 0),
            "max_regret":         round(max_regret, 0),
            "cost_optimistic":    costs["optimistic"],
            "cost_moderate":      costs["moderate"],
            "cost_pessimistic":   costs["pessimistic"],
            "total_subcon_units": round(
                mod["df_plan"]["Subcontracting"].sum(), 1
            ) if not mod["df_plan"].empty else 0,
            "total_ot_hours":     round(
                mod["df_plan"]["OT (jam)"].sum(), 1
            ) if not mod["df_plan"].empty else 0,
            "avg_capacity_util":  round(
                mod["df_plan"]["Utilisasi (%)"].mean(), 1
            ) if not mod["df_plan"].empty else 0,
            "total_stockout":     round(
                mod["df_plan"]["Stockout"].sum(), 1
            ) if not mod["df_plan"].empty else 0,
            "safety_stock_rm":    mod.get("rm_ss_result", {}).get("safety_stock", 0),
            "reorder_point":      mod.get("rm_ss_result", {}).get("reorder_point", 0),
        }

    else:
        # ada skenario yang tidak optimal
        df_comparison  = pd.DataFrame()
        summary_metrics = {}
        expected_cost   = None
        max_regret      = None

    return {
        "results":         results,       # hasil lengkap per skenario
        "df_comparison":   df_comparison, # tabel perbandingan
        "summary_metrics": summary_metrics,
        "all_optimal":     all_ok,
        "strategy":        strategy,
    }


# =============================================================================
# BAGIAN 5 — ANALISIS SENSITIVITAS
# Jalankan model berulang dengan satu parameter diubah,
# parameter lain tetap. Hasilnya adalah data untuk tornado chart.
# =============================================================================

def sensitivity_analysis(
    demand_scenario: DemandScenario,
    cost: CostParams,
    cap: CapacityParams,
    init: InitialConditions,
    supply: SupplyParams,
    strategy: str = "mixed",
    variation_pct: float = 0.20,  # variasi ±20% dari nilai baseline
) -> pd.DataFrame:
    """
    Lakukan analisis sensitivitas one-at-a-time (OAT) terhadap parameter biaya.

    Untuk setiap parameter biaya:
      - Naikkan sebesar variation_pct (misal +20%)
      - Turunkan sebesar variation_pct (misal -20%)
      - Catat perubahan expected total cost

    Output digunakan untuk membuat tornado chart di Streamlit.

    Parameters
    ----------
    variation_pct : float — persentase variasi (0.20 = ±20%)

    Returns
    -------
    DataFrame dengan kolom: Parameter, Nilai_Baseline, Cost_Low, Cost_High, Swing
    """
    import copy

    # ambil expected cost baseline (skenario moderat, nilai parameter asli)
    baseline_result = solve_all_scenarios(
        demand_scenario, cost, cap, init, supply, strategy
    )
    baseline_cost = baseline_result["summary_metrics"].get("expected_cost", 0)

    # daftar parameter yang akan dianalisis (nama atribut, label untuk chart)
    params_to_test = [
        ("rt_labor_cost",    "RT Labor Cost"),
        ("ot_labor_cost",    "OT Labor Cost"),
        ("hiring_cost",      "Hiring Cost"),
        ("firing_cost",      "Firing Cost"),
        ("material_cost",    "Material Cost"),
        ("subcon_cost",      "Subcontracting Cost"),
        ("inventory_cost",   "Inventory Holding Cost"),
        ("stockout_cost",    "Stockout Penalty"),
        ("rm_holding_cost",  "RM Holding Cost"),
    ]

    rows = []
    for attr, label in params_to_test:
        base_val = getattr(cost, attr)

        # --- nilai parameter dinaikkan (+variation_pct) ---
        cost_high = copy.deepcopy(cost)
        setattr(cost_high, attr, base_val * (1 + variation_pct))
        result_high = solve_all_scenarios(
            demand_scenario, cost_high, cap, init, supply, strategy
        )
        tc_high = result_high["summary_metrics"].get("expected_cost", baseline_cost)

        # --- nilai parameter diturunkan (-variation_pct) ---
        cost_low = copy.deepcopy(cost)
        setattr(cost_low, attr, base_val * (1 - variation_pct))
        result_low = solve_all_scenarios(
            demand_scenario, cost_low, cap, init, supply, strategy
        )
        tc_low = result_low["summary_metrics"].get("expected_cost", baseline_cost)

        rows.append({
            "Parameter":        label,
            "Baseline Cost":    round(baseline_cost, 0),
            "Cost (-20%)":      round(tc_low, 0),
            "Cost (+20%)":      round(tc_high, 0),
            "Swing":            round(tc_high - tc_low, 0),  # lebar ayunan total
            "Impact Low (%)":   round((tc_low  - baseline_cost) / baseline_cost * 100, 2),
            "Impact High (%)":  round((tc_high - baseline_cost) / baseline_cost * 100, 2),
        })

    # urutkan dari swing terbesar ke terkecil (untuk tornado chart)
    df = pd.DataFrame(rows).sort_values("Swing", ascending=False).reset_index(drop=True)
    return df


# =============================================================================
# BAGIAN 6 — PERBANDINGAN TIGA STRATEGI
# Jalankan ketiga strategi (chase, level, mixed) pada skenario moderat
# untuk perbandingan langsung total cost dan karakteristik masing-masing.
# =============================================================================

def compare_strategies(
    demand_scenario: DemandScenario,
    cost: CostParams,
    cap: CapacityParams,
    init: InitialConditions,
    supply: SupplyParams,
) -> Dict[str, pd.DataFrame]:
    """
    Bandingkan tiga strategi AP pada demand skenario moderat.

    Returns
    -------
    dict berisi:
      - "df_strategy_comparison": tabel perbandingan total cost per strategi
      - "results_by_strategy"   : dict hasil lengkap per strategi
    """
    strategies = ["chase", "level", "mixed"]
    strat_results = {}

    for strat in strategies:
        strat_results[strat] = solve_single_scenario(
            scenario_name = "moderate",
            demand        = demand_scenario.moderate,
            cost          = cost,
            cap           = cap,
            init          = init,
            supply        = supply,
            strategy      = strat,
        )

    # buat tabel perbandingan
    rows = []
    for strat in strategies:
        r = strat_results[strat]
        if r["status"] == "Optimal":
            plan = r["df_plan"]
            rows.append({
                "Strategi":           strat.capitalize(),
                "Total Cost (Rp)":    r["total_cost"],
                "Total Stockout":     plan["Stockout"].sum(),
                "Total Hiring":       plan["Hiring"].sum(),
                "Total Firing":       plan["Firing"].sum(),
                "Total OT (jam)":     plan["OT (jam)"].sum(),
                "Total Subcon":       plan["Subcontracting"].sum(),
                "Avg Utilisasi (%)":  round(plan["Utilisasi (%)"].mean(), 1),
            })
        else:
            rows.append({
                "Strategi":           strat.capitalize(),
                "Total Cost (Rp)":    None,
                "Total Stockout":     None,
                "Total Hiring":       None,
                "Total Firing":       None,
                "Total OT (jam)":     None,
                "Total Subcon":       None,
                "Avg Utilisasi (%)":  None,
            })

    df_comparison = pd.DataFrame(rows)

    return {
        "df_strategy_comparison": df_comparison,
        "results_by_strategy":    strat_results,
    }


# =============================================================================
# BAGIAN 7 — FUNGSI HELPER UNTUK STREAMLIT
# Fungsi-fungsi kecil yang sering dipanggil dari Streamlit UI.
# =============================================================================

def make_demand_scenario(
    base_demand: List[float],
    pct_up: float = 0.25,
    pct_down: float = 0.25,
    prob_opt: float = 0.25,
    prob_mod: float = 0.50,
    prob_pes: float = 0.25,
) -> DemandScenario:
    """
    Buat DemandScenario dari demand baseline dengan variasi persentase.

    Parameters
    ----------
    base_demand : List[float] — demand moderat per periode
    pct_up      : float — persentase kenaikan untuk skenario optimis
    pct_down    : float — persentase penurunan untuk skenario pesimis

    Returns
    -------
    DemandScenario siap dipakai sebagai input solver
    """
    return DemandScenario(
        moderate    = [float(d)               for d in base_demand],
        optimistic  = [float(d) * (1 + pct_up)   for d in base_demand],
        pessimistic = [float(d) * (1 - pct_down)  for d in base_demand],
        prob_opt    = prob_opt,
        prob_mod    = prob_mod,
        prob_pes    = prob_pes,
    )


def format_currency(value: float, prefix: str = "Rp ") -> str:
    """Format angka sebagai string mata uang Rupiah."""
    if value is None:
        return "N/A"
    return f"{prefix}{value:,.0f}"


def get_cost_breakdown_summary(df_cost: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung total per komponen biaya sepanjang horizon.
    Dipakai untuk donut chart di Streamlit.
    """
    cost_cols = [c for c in df_cost.columns if c not in ("Periode", "Total Cost")]
    totals = {col: df_cost[col].sum() for col in cost_cols}
    grand  = sum(totals.values())
    rows   = [
        {
            "Komponen":    col,
            "Total (Rp)":  round(totals[col], 0),
            "Porsi (%)":   round(totals[col] / grand * 100, 1) if grand > 0 else 0,
        }
        for col in cost_cols
        if totals[col] > 0  # sembunyikan komponen dengan biaya nol
    ]
    return pd.DataFrame(rows).sort_values("Total (Rp)", ascending=False)


# =============================================================================
# BAGIAN 8 — DEMO / TEST RUN
# Blok ini hanya berjalan jika file dijalankan langsung: python lp_model.py
# Tidak akan berjalan saat diimpor oleh Streamlit (karena __name__ != "__main__")
# Ini memungkinkan file yang sama bekerja di Python langsung DAN di Streamlit.
# =============================================================================

if __name__ == "__main__":

    print("=" * 65)
    print("  DEMO: Aggregate Planning LP Model")
    print("  Skenario: Industri Manufaktur — 12 Periode")
    print("=" * 65)

    # --- Contoh data demand moderat (12 bulan) ---
    # Data ini akan diisi oleh user via Streamlit sidebar di aplikasi nyata
    base_demand = [
        420, 380, 400, 450, 500, 620,   # semester 1 (ada peak di bulan 6)
        580, 470, 440, 410, 490, 560,   # semester 2 (peak akhir tahun)
    ]

    # --- Buat skenario demand ---
    demand_sc = make_demand_scenario(
        base_demand = base_demand,
        pct_up      = 0.25,   # optimis: +25%
        pct_down    = 0.20,   # pesimis: -20%
        prob_opt    = 0.25,
        prob_mod    = 0.50,
        prob_pes    = 0.25,
    )

    # --- Parameter biaya (dalam Rupiah) ---
    cost_params = CostParams(
        rt_labor_cost    = 4_500_000,   # Rp 4.5 juta per orang per bulan
        ot_labor_cost    = 28_000,      # Rp 28 ribu per jam lembur
        hiring_cost      = 3_000_000,   # Rp 3 juta per rekrutan baru
        firing_cost      = 5_000_000,   # Rp 5 juta per PHK
        inventory_cost   = 15_000,      # Rp 15 ribu per unit per bulan
        stockout_cost    = 75_000,      # Rp 75 ribu per unit tidak terpenuhi
        material_cost    = 85_000,      # Rp 85 ribu per unit produksi
        subcon_cost      = 120_000,     # Rp 120 ribu per unit subcon
        rm_holding_cost  = 8_000,       # Rp 8 ribu per unit RM per bulan
        rm_ordering_cost = 500_000,     # Rp 500 ribu per order bahan baku
        rm_shortage_cost = 50_000,      # Rp 50 ribu per unit RM yang kurang
    )

    # --- Parameter kapasitas ---
    cap_params = CapacityParams(
        prod_time         = 2.5,   # 2.5 jam per unit
        rt_hours          = 160,   # 160 jam regular per orang per bulan (8 jam × 20 hari)
        ot_max_per_worker = 40,    # maks 40 jam lembur per orang per bulan
        capacity_max      = 600,   # maks 600 unit per bulan (batas mesin)
        W_min             = 5,     # minimal 5 pekerja di setiap waktu
        subcon_max        = 150,   # mitra bisa handle maks 150 unit per bulan
        rm_per_unit       = 3.0,   # 3 unit bahan baku per unit produk jadi
    )

    # --- Kondisi awal ---
    init_cond = InitialConditions(
        W0           = 10,     # 10 pekerja saat ini
        I0_fg        = 50,     # 50 unit produk jadi di gudang
        I0_rm        = 500,    # 500 unit bahan baku di gudang
        RM_on_order0 = 200,    # 200 unit RM sudah dipesan, belum tiba
    )

    # --- Parameter supply dan lead time ---
    supply_params = SupplyParams(
        lt_mean          = 2.0,   # rata-rata 2 bulan lead time
        lt_std           = 0.5,   # standar deviasi 0.5 bulan
        service_level_rm = 0.95,  # target 95% service level bahan baku
        rm_order_qty     = 1500,  # pesan 1500 unit setiap kali order
    )

    # =========================================================================
    # JALANKAN SOLVER — TIGA SKENARIO SEKALIGUS
    # =========================================================================
    print("\n[1] Menjalankan solver untuk 3 skenario demand (mixed strategy)...\n")

    hasil = solve_all_scenarios(
        demand_scenario = demand_sc,
        cost            = cost_params,
        cap             = cap_params,
        init            = init_cond,
        supply          = supply_params,
        strategy        = "mixed",
    )

    # tampilkan status
    for s in ["optimistic", "moderate", "pessimistic"]:
        r = hasil["results"][s]
        tc = format_currency(r["total_cost"])
        print(f"  Skenario {s:12s}: {r['status']:10s} | Total Cost = {tc}")

    # tampilkan expected cost
    sm = hasil["summary_metrics"]
    print(f"\n  Expected Total Cost  : {format_currency(sm.get('expected_cost'))}")
    print(f"  Max Regret           : {format_currency(sm.get('max_regret'))}")
    print(f"  Safety Stock RM      : {sm.get('safety_stock_rm', 0):.1f} unit")
    print(f"  Reorder Point        : {sm.get('reorder_point', 0):.1f} unit")
    print(f"  Avg Utilisasi        : {sm.get('avg_capacity_util', 0):.1f}%")

    # tampilkan tabel rencana produksi skenario moderat
    print("\n[2] Rencana Produksi — Skenario Moderat (Mixed Strategy):")
    print("-" * 65)
    df_plan = hasil["results"]["moderate"]["df_plan"]
    cols_show = [
        "Periode", "Demand (unit)", "Produksi Internal",
        "Subcontracting", "Inventory FG", "Stockout",
        "TK Aktif", "OT (jam)", "Utilisasi (%)", "Make-or-Buy"
    ]
    print(df_plan[cols_show].to_string(index=False))

    # tampilkan breakdown biaya
    print("\n[3] Breakdown Biaya — Skenario Moderat:")
    print("-" * 65)
    df_cost = hasil["results"]["moderate"]["df_cost"]
    print(df_cost.to_string(index=False))

    # =========================================================================
    # PERBANDINGAN TIGA STRATEGI
    # =========================================================================
    print("\n[4] Perbandingan Chase vs Level vs Mixed Strategy:")
    print("-" * 65)
    strat_result = compare_strategies(
        demand_sc, cost_params, cap_params, init_cond, supply_params
    )
    df_strat = strat_result["df_strategy_comparison"]
    print(df_strat.to_string(index=False))

    # =========================================================================
    # ANALISIS SENSITIVITAS
    # =========================================================================
    print("\n[5] Analisis Sensitivitas (±20% tiap parameter biaya):")
    print("-" * 65)
    df_sens = sensitivity_analysis(
        demand_sc, cost_params, cap_params, init_cond, supply_params,
        strategy="mixed", variation_pct=0.20
    )
    print(df_sens[["Parameter", "Cost (-20%)", "Baseline Cost", "Cost (+20%)", "Swing"]].to_string(index=False))

    print("\n" + "=" * 65)
    print("  Demo selesai. File ini siap diimpor oleh Streamlit.")
    print("  Import dengan: from lp_model import *")
    print("=" * 65)