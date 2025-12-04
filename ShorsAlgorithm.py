from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import QFT
from math import gcd, pi
from random import randint
from fractions import Fraction
import numpy as np
import time
import matplotlib.pyplot as plt


# HELPER FUNCTIONS
def mod_inverse(e, phi):
    """Returns d such that (d*e) % phi == 1"""
    def egcd(a, b):
        if a == 0:
            return b, 0, 1
        else:
            g, y, x = egcd(b % a, a)
            return g, x - (b // a) * y, y

    g, x, _ = egcd(e, phi)
    if g != 1:
        return None
    else:
        return x % phi


# RSA ENCRYPTION/DECRYPTION
def generate_rsa_keys(p, q):
    """Generate RSA public and private keys from primes p and q"""
    N = p * q
    phi = (p - 1) * (q - 1)
    
    # Choose e (commonly 65537, but we'll use smaller values for small N)
    e = 65537
    if e >= phi:
        e = 3
        while gcd(e, phi) != 1:
            e += 2
    
    d = mod_inverse(e, phi)
    
    return (e, N), (d, N)  # public_key, private_key


def rsa_encrypt(message, public_key):
    """Encrypt a message using RSA public key"""
    e, N = public_key
    return pow(message, e, N)


def rsa_decrypt(ciphertext, private_key):
    """Decrypt a ciphertext using RSA private key"""
    d, N = private_key
    return pow(ciphertext, d, N)


def rsa_attack_by_factoring(N):
    """Attack RSA by factoring N to recover the private key
    """
    if N % 2 == 0:
        p = 2
        q = N // 2
        return p, q
    
    for i in range(3, int(N**0.5) + 1, 2):
        if N % i == 0:
            return i, N // i
    return None


# CLASSICAL FACTORING ALGORITHMS

def trial_division(N):
    """Trial division - simplest classical factoring method"""
    if N % 2 == 0:
        return 2, N // 2
    
    for i in range(3, int(N**0.5) + 1, 2):
        if N % i == 0:
            return i, N // i
    return None


def classical_period_finding(a, N):
    """Classical period finding - used in classical Shor simulation"""
    r = 1
    current = a % N
    while current != 1 and r < N:
        current = (current * a) % N
        r += 1
    return r if current == 1 else None


def classical_shor_method(N):
    """Classical implementation of Shor's approach (without quantum speedup)
    
    This uses the SAME mathematical approach as quantum Shor's algorithm,
    but WITHOUT the quantum speedup for period finding. This shows how
    the algorithm works but loses the exponential advantage.
    """
    # STEP 1: Check if N is even (trivial case)
    if N % 2 == 0:
        return 2, N // 2

    # STEP 2: Try multiple random values of 'a' (base for modular exponentiation)
    for attempt in range(20):
        a = randint(2, N - 1)
        
        # STEP 3: Check if gcd(a, N) is already a factor
        g = gcd(a, N)
        if g != 1:
            return g, N // g
        
        # STEP 4: Find the period 'r' where a^r ≡ 1 (mod N)
        # THIS IS THE CRITICAL STEP where quantum provides speedup!
        # Classical: O(N) time - tries every power until we cycle back to 1
        # Quantum: O(log(N)^3) time - uses quantum Fourier transform
        r = classical_period_finding(a, N)
        
        # STEP 5: Check if period is valid (must be even and not None)
        if r is None or r % 2 != 0:
            continue
        
        # STEP 6: Calculate x = a^(r/2) mod N
        x = pow(a, r // 2, N)
        
        # STEP 7: Check for the trivial case x ≡ -1 (mod N)
        if x == N - 1: 
            continue
        
        # STEP 8: Extract potential factors using GCD
        p = gcd(x - 1, N) 
        q = gcd(x + 1, N)  
        
        # STEP 9: Validate that we found real factors
        if p > 1 and q > 1 and p != N and q != N and p * q == N:
            return p, q
    
    # If all 20 attempts failed, return None
    return None


# QUANTUM SHOR'S ALGORITHM

def c_amod15(a, power):
    """Controlled multiplication by a mod 15"""
    if a not in [2, 4, 7, 8, 11, 13]:
        raise ValueError("'a' must be 2,4,7,8,11 or 13")
    U = QuantumCircuit(4)
    for _ in range(power):
        if a in [2, 13]:
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        if a in [7, 8]:
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        if a in [4, 11]:
            U.swap(1, 3)
            U.swap(0, 2)
        if a in [7, 11, 13]:
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = f"{a}^{power} mod 15"
    c_U = U.control()
    return c_U


def qft_dagger(n):
    """n-qubit inverse QFT"""
    qc = QuantumCircuit(n)
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)
    for j in range(n):
        for m in range(j):
            qc.cp(-pi / float(2 ** (j - m)), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc


def quantum_period_finding(a, N, n_count=8):
    """Quantum period finding using Qiskit"""
    qc = QuantumCircuit(n_count + 4, n_count)
    
    for q in range(n_count):
        qc.h(q)
    
    qc.x(n_count)
    
    for q in range(n_count):
        qc.append(c_amod15(a, 2 ** q), [q] + [i + n_count for i in range(4)])
    
    qc.append(qft_dagger(n_count), range(n_count))
    qc.measure(range(n_count), range(n_count))
    
    backend = Aer.get_backend('aer_simulator')
    job = backend.run(transpile(qc, backend), shots=1024, memory=True)
    result = job.result()
    counts = result.get_counts()
    
    return counts


def quantum_shor(N):
    """Quantum Shor's algorithm (only works for N=15 in this implementation)"""
    if N != 15:
        return None
    
    if N % 2 == 0:
        return 2, N // 2

    n_count = 8
    
    for attempt in range(10):
        a = randint(2, N - 1)
        if a not in [2, 4, 7, 8, 11, 13]:
            continue
        
        g = gcd(a, N)
        if g != 1:
            return g, N // g
        
        counts = quantum_period_finding(a, N, n_count)
        
        for output in counts:
            decimal = int(output, 2)
            phase = decimal / (2 ** n_count)
            
            if phase == 0:
                continue
            
            frac = Fraction(phase).limit_denominator(N)
            r = frac.denominator
            
            if r % 2 == 0 and pow(a, r, N) == 1:
                x = pow(a, r // 2, N)
                
                if x == N - 1:
                    continue
                
                p = gcd(x - 1, N)
                q = gcd(x + 1, N)
                
                if p > 1 and q > 1 and p != N and q != N and p * q == N:
                    return p, q
    
    return None


# MAIN COMPARISON
print("=" * 70)
print(" PROPER COMPARISON: CLASSICAL VS QUANTUM RSA ATTACK")
print("=" * 70)
print("\nTHE CORRECT COMPARISON:")
print("  ✓ Both algorithms attacking the SAME task: factoring N")
print("  ✓ Both trying to break RSA encryption")
print("  ✓ Using the SAME modulus N for fair comparison")
print("\n" + "=" * 70)


# Test on multiple values of N
test_values = [15, 21, 33, 35, 55, 77, 91, 143]

results = []

print("\n" + "=" * 70)
print(" FACTORING RACE: CLASSICAL vs QUANTUM")
print("=" * 70)

for N in test_values:
    print(f"\n📊 Testing N = {N}")
    print("-" * 50)
    
    # Trial Division (simplest classical)
    start = time.perf_counter()
    trial_result = trial_division(N)
    trial_time = time.perf_counter() - start
    
    # RSA Attack by factoring
    start = time.perf_counter()
    rsa_result = rsa_attack_by_factoring(N)
    rsa_time = time.perf_counter() - start
    
    # Classical Shor method
    start = time.perf_counter()
    classical_shor_result = classical_shor_method(N)
    classical_shor_time = time.perf_counter() - start
    
    # Quantum Shor (only works for N=15)
    if N == 15:
        start = time.perf_counter()
        quantum_result = quantum_shor(N)
        quantum_time = time.perf_counter() - start
        
        print(f"  Trial Division:     {trial_time*1000:.3f} ms  → {trial_result}")
        print(f"  RSA Attack:         {rsa_time*1000:.3f} ms  → {rsa_result}")
        print(f"  Classical Shor:     {classical_shor_time*1000:.3f} ms  → {classical_shor_result}")
        print(f"  🔮 QUANTUM Shor:    {quantum_time*1000:.3f} ms  → {quantum_result}")
        
        if quantum_result:
            print(f"\n  ⚡ Quantum speedup vs classical Shor: {classical_shor_time/quantum_time:.2f}x")
    else:
        print(f"  Trial Division:     {trial_time*1000:.3f} ms  → {trial_result}")
        print(f"  RSA Attack:         {rsa_time*1000:.3f} ms  → {rsa_result}")
        print(f"  Classical Shor:     {classical_shor_time*1000:.3f} ms  → {classical_shor_result}")
        print(f"  🔮 Quantum Shor:    N/A (only implemented for N=15)")
        quantum_time = None
        quantum_result = None
    
    results.append({
        'N': N,
        'trial_time': trial_time * 1000,
        'rsa_time': rsa_time * 1000,
        'classical_shor_time': classical_shor_time * 1000,
        'quantum_time': quantum_time * 1000 if quantum_time else None
    })


# THEORETICAL COMPLEXITY COMPARISON
print("\n" + "=" * 70)
print(" THEORETICAL COMPLEXITY ANALYSIS")
print("=" * 70)

key_sizes = [512, 1024, 2048, 4096]

print("\nEstimated time to factor RSA keys of different sizes:")
print("-" * 70)
print(f"{'Key Size':<12} {'Trial Division':<20} {'Best Classical':<20} {'Quantum Shor'}")
print("-" * 70)

for bits in key_sizes:
    # These are rough estimates based on complexity classes
    trial_log = (bits / 2) * np.log(2) - np.log(1e15)
    trial_est = np.exp(trial_log) if trial_log < 700 else float('inf')
    gnfs_exponent = 1.923 * (bits ** (1/3)) * ((np.log(bits)) ** (2/3)) - np.log(1e9)
    gnfs_est = np.exp(gnfs_exponent) if gnfs_exponent < 700 else float('inf')
    
    # Quantum Shor: O(n^3 * log(n))
    quantum_est = (bits ** 3) * np.log(bits) / 1e9  # seconds
    
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.2f} sec"
        elif seconds < 3600:
            return f"{seconds/60:.2f} min"
        elif seconds < 86400:
            return f"{seconds/3600:.2f} hours"
        elif seconds < 31536000:
            return f"{seconds/86400:.2f} days"
        elif seconds < 31536000000:
            return f"{seconds/31536000:.2e} years"
        else:
            return f"{seconds/31536000:.2e} years"
    
    print(f"{bits}-bit{'':<6} {format_time(trial_est):<20} {format_time(gnfs_est):<20} {format_time(quantum_est)}")

print("-" * 70)


# VISUALIZATION
print("\n" + "=" * 70)
print(" GENERATING VISUALIZATIONS")
print("=" * 70)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Actual measured times for small N
ax1 = fig.add_subplot(gs[0, :])
n_values = [r['N'] for r in results]
trial_times = [r['trial_time'] for r in results]
rsa_times = [r['rsa_time'] for r in results]
classical_shor_times = [r['classical_shor_time'] for r in results]

ax1.plot(n_values, trial_times, 'o-', label='Trial Division', linewidth=2, markersize=8, color='red')
ax1.plot(n_values, rsa_times, 's-', label="RSA Attack", linewidth=2, markersize=8, color='orange')
ax1.plot(n_values, classical_shor_times, '^-', label='Classical Shor Method', linewidth=2, markersize=8, color='blue')

# Add quantum point for N=15
quantum_point = next((r for r in results if r['N'] == 15 and r['quantum_time']), None)
if quantum_point:
    ax1.plot([15], [quantum_point['quantum_time']], 'D', label='Quantum Shor (N=15)', 
             markersize=15, color='green', markeredgecolor='black', markeredgewidth=2)

ax1.set_xlabel('N (number to factor)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Time (milliseconds)', fontsize=12, fontweight='bold')
ax1.set_title('ACTUAL MEASURED TIMES: Classical vs Quantum Factoring\n(Same Task, Same Numbers)', 
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Plot 2: Theoretical complexity for large keys
ax2 = fig.add_subplot(gs[1, 0])
key_bits = np.array([256, 512, 768, 1024, 1536, 2048, 3072, 4096])

# Calculate theoretical times (normalized)
classical_times = np.exp(1.923 * (key_bits ** (1/3)) * ((np.log(key_bits)) ** (2/3)))
quantum_times = (key_bits ** 3) * np.log(key_bits)

# Normalize to make comparison visible
classical_times = classical_times / classical_times[0]
quantum_times = quantum_times / quantum_times[0]

ax2.plot(key_bits, classical_times, 'o-', label='Best Classical (GNFS)', 
         linewidth=3, markersize=8, color='red')
ax2.plot(key_bits, quantum_times, 's-', label='Quantum Shor', 
         linewidth=3, markersize=8, color='green')
ax2.set_xlabel('RSA Key Size (bits)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Relative Time (normalized)', fontsize=11, fontweight='bold')
ax2.set_title('Theoretical Complexity Growth\n(Exponential vs Polynomial)', 
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# Plot 3: Speedup ratio
ax3 = fig.add_subplot(gs[1, 1])
speedup_ratio = classical_times / quantum_times
ax3.plot(key_bits, speedup_ratio, 'o-', linewidth=3, markersize=10, color='purple')
ax3.fill_between(key_bits, speedup_ratio, alpha=0.3, color='purple')
ax3.set_xlabel('RSA Key Size (bits)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Speedup Factor (Classical / Quantum)', fontsize=11, fontweight='bold')
ax3.set_title('Quantum Advantage Growth\n(Why RSA is Doomed)', 
              fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

for i, (bits, speedup) in enumerate(zip(key_bits, speedup_ratio)):
    if i % 2 == 0:  # Label every other point
        ax3.annotate(f'{speedup:.0e}x', 
                    xy=(bits, speedup), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8,
                    fontweight='bold')

# Plot 4: Attack comparison summary
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

summary_text = """
"""

ax4.text(0.05, 0.95, summary_text, 
         transform=ax4.transAxes,
         fontsize=10,
         verticalalignment='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('PROPER COMPARISON: Classical vs Quantum RSA Attack\n(Why Quantum Computers Threaten Modern Encryption)', 
             fontsize=15, fontweight='bold', y=0.995)

plt.show()