from qiskit_aer import Aer
from qiskit import QuantumCircuit
from math import gcd
from random import randint
from Crypto.Util.number import inverse, long_to_bytes, bytes_to_long
import numpy as np
import time
import timeit
import matplotlib.pyplot as plt


# HELPER FUNCTIONS
def mod_inverse(e, phi):
    """
    Returns d such that (d*e) % phi == 1
    Uses extended Euclidean algorithm
    """
    def egcd(a, b):
        if a == 0:
            return b, 0, 1
        else:
            g, y, x = egcd(b % a, a)
            return g, x - (b // a) * y, y

    g, x, _ = egcd(e, phi)
    if g != 1:
        return None  # no inverse
    else:
        return x % phi
    
def text_to_num(text):
    return int.from_bytes(text.encode(), 'big')

def num_to_text(num):
    try:
        byte_length = (num.bit_length() + 7) // 8
        return num.to_bytes(byte_length, 'big').decode('utf-8', errors='ignore')
    except Exception as e:
        return f"[Error: {num}]"


# SHOR SIMULATION
def find_period(a, N):
    """Find the period r where a^r ≡ 1 (mod N)"""
    r = 1
    current = a % N
    while current != 1 and r < N:
        current = (current * a) % N
        r += 1
    return r if current == 1 else None


def shor_classical_sim(N):
    """Classical simulation of Shor's algorithm"""
    # Check if N is even
    if N % 2 == 0:
        return 2, N//2

    # Try multiple random values of a
    for attempt in range(20):
        a = randint(2, N-1)
        
        # Check if gcd(a, N) > 1
        g = gcd(a, N)
        if g != 1:
            return g, N//g
        
        # Find the period r
        r = find_period(a, N)
        
        if r is None or r % 2 != 0:
            continue
        
        # Calculate potential factors
        x = pow(a, r//2, N)
        
        if x == N - 1:  # x ≡ -1 (mod N)
            continue
        
        p = gcd(x - 1, N)
        q = gcd(x + 1, N)
        
        # Validate factors
        if p > 1 and q > 1 and p != N and q != N and p * q == N:
            return p, q
    
    return None


# RSA SETUP
print("\n================ RSA (CLASSICAL) ================")

start_keygen = time.time()

# HELLO = 310400273487 in numeric form
p = 1000003  
q = 1000033  
n = p * q    
phi = (p - 1) * (q - 1)
e = 65537  
while gcd(e, phi) != 1:
    e += 2
d = mod_inverse(e, phi)

end_keygen = time.time()
keygen_time = end_keygen - start_keygen

message = "HELLO"
m = text_to_num(message)

# Verify message is smaller than modulus
if m >= n:
    print(f"Warning: Message value {m} is too large for modulus {n}")
    message = "HI"
    m = text_to_num(message)
    print(f"Using shorter message: {message}")
else:
    print(f"Message '{message}' = {m} (fits in modulus {n})")


# RSA ENCRYPT 
start_encrypt = time.time()
cipher = pow(m, e, n)
end_encrypt = time.time()
encrypt_time = end_encrypt - start_encrypt


# RSA DECRYPT 
start_decrypt = time.time()
original = pow(cipher, d, n)
decoded = num_to_text(original)
end_decrypt = time.time()
decrypt_time = end_decrypt - start_decrypt

print(f"Original message: {message}")
print(f"Public Key (n, e): ({n}, {e})")
print(f"Private Key d: {d}")
print(f"Encrypted message: {cipher}")
print(f"Decrypted message: {decoded}")

def format_time(seconds):
    """Format time in the most appropriate unit"""
    if seconds == 0:
        return "< 0.001 μs"
    elif seconds < 1e-6:  
        return f"{seconds*1e9:.3f} ns"
    elif seconds < 1e-3:  
        return f"{seconds*1e6:.3f} μs"
    elif seconds < 1:  
        return f"{seconds*1e3:.3f} ms"
    else:
        return f"{seconds:.3f} s"

print(f"\nRSA Key generation time: {format_time(keygen_time)}")
print(f"RSA Encryption time:     {format_time(encrypt_time)}")
print(f"RSA Decryption time:     {format_time(decrypt_time)}")


# SHOR ATTACK 
print("\n============ SHOR'S ALGORITHM (ATTACK) ============")

# Use a smaller n for the attack demo
small_n = 15  
start_shor = time.perf_counter()

print(f"Attempting to factor n = {small_n}")
factors = shor_classical_sim(small_n)

if factors:
    p_f, q_f = factors
    print(f"✓ Shor found factors of n: {p_f} × {q_f} = {p_f * q_f}")
    
    phi_f = (p_f - 1) * (q_f - 1)
    
    # Use a smaller e that works with small_n
    e_small = 3
    while gcd(e_small, phi_f) != 1:
        e_small += 2
    
    hacked_d = mod_inverse(e_small, phi_f)
    
    # Encrypt a test message with small_n
    test_msg_text = "A"
    test_msg = text_to_num(test_msg_text)
    
    # Make sure message fits
    if test_msg >= small_n:
        test_msg = ord("A")  # Use ASCII value instead
        test_msg_text = f"A (ASCII={test_msg})"
    
    test_cipher = pow(test_msg, e_small, small_n)
    cracked_message = pow(test_cipher, hacked_d, small_n)
    
    print(f"✓ Recovered private key d: {hacked_d}")
    print(f"✓ Test encryption: '{test_msg_text}' → {test_cipher}")
    print(f"✓ Successfully decrypted: {test_cipher} → {cracked_message}")
    print(f"✓ Attack successful! Attacker can now decrypt any message encrypted with n={small_n}")
else:
    print("✗ Shor attack failed.")

end_shor = time.perf_counter()
shor_time = end_shor - start_shor

print(f"\nShor attack time: {format_time(shor_time)}")


# SUMMARY 
print("\n================= FINAL COMPARISON =================")

total_rsa_time = keygen_time + encrypt_time + decrypt_time
print(f"RSA Key Generation:      {format_time(keygen_time)}")
print(f"RSA Encryption:          {format_time(encrypt_time)}")
print(f"RSA Decryption:          {format_time(decrypt_time)}")
print(f"RSA TOTAL TIME:          {format_time(total_rsa_time)}")
print(f"\nShor factoring attack:   {format_time(shor_time)}")

if total_rsa_time > 0 and shor_time > 0:
    if shor_time > total_rsa_time:
        ratio = shor_time / total_rsa_time
        print(f"\nSpeed comparison: Shor is {ratio:.1f}x SLOWER than RSA (classical simulation)")
    else:
        ratio = total_rsa_time / shor_time
        print(f"\nSpeed comparison: RSA is {ratio:.1f}x slower than Shor (unexpected!)")
else:
    print("\nSpeed comparison: Times too small to compare accurately")

print("\nConclusion:")
print("• RSA is fast for encryption/decryption, but its security depends on hard factoring.")
print("• Shor's algorithm successfully factored N and broke RSA.")
print("• This demonstrates why quantum computers threaten classical encryption.")
print("• NOTE: This is a classical simulation. Real quantum computers would be much faster.")


# TIMING GRAPH
print("\n================= GENERATING TIMING GRAPH =================")

# Use a wider range of composite numbers for better visualization
n_values = [15, 21, 33, 35, 55, 77, 91, 143, 187, 221, 323, 437]
rsa_times = []
shor_times = []
successful_n = []

for n_val in n_values:
    print(f"Testing n = {n_val}...", end=" ")
    
    # Get factors for RSA setup
    result = shor_classical_sim(n_val)
    if result is None:
        print(f"Could not factor, skipping...")
        continue
    
    p_val, q_val = result
    phi_val = (p_val - 1) * (q_val - 1)
    e_val = 3
    while gcd(e_val, phi_val) != 1:
        e_val += 2
    d_val = mod_inverse(e_val, phi_val)
    m_val = 65  # ASCII 'A'
    
    # Time RSA multiple times for better measurement
    rsa_iterations = 10000  
    start_rsa = time.time()
    for _ in range(rsa_iterations):
        c_val = pow(m_val, e_val, n_val)
        o_val = pow(c_val, d_val, n_val)
    end_rsa = time.time()
    avg_rsa_time = (end_rsa - start_rsa) / rsa_iterations
    
    # Time Shor (just once as it's slower)
    start_shor = time.time()
    shor_classical_sim(n_val)
    end_shor = time.time()
    shor_time = end_shor - start_shor
    
    rsa_times.append(avg_rsa_time * 1000000)  
    shor_times.append(shor_time * 1000)  
    successful_n.append(n_val)
    print(f"✓ RSA: {avg_rsa_time*1000000:.3f}μs, Shor: {shor_time*1000:.3f}ms")

if len(rsa_times) > 0:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: RSA Time Only (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(successful_n, rsa_times, marker='o', linewidth=2, markersize=8, color='green', label='RSA')
    ax1.fill_between(successful_n, rsa_times, alpha=0.3, color='green')
    ax1.set_xlabel('RSA Modulus (n)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Time (microseconds)', fontsize=11, fontweight='bold')
    ax1.set_title('RSA Encryption + Decryption Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Shor Time Only (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(successful_n, shor_times, marker='s', linewidth=2, markersize=8, color='red', label='Shor')
    ax2.fill_between(successful_n, shor_times, alpha=0.3, color='red')
    ax2.set_xlabel('RSA Modulus (n)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Time (milliseconds)', fontsize=11, fontweight='bold')
    ax2.set_title('Shor\'s Algorithm Attack Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Both on same graph with dual y-axis (Middle Left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(successful_n, rsa_times, marker='o', label='RSA Enc+Dec', linewidth=2, markersize=8, color='green')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(successful_n, shor_times, marker='s', label='Shor Attack', linewidth=2, markersize=8, color='red')
    ax3.set_xlabel('RSA Modulus (n)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('RSA Time (μs)', fontsize=10, color='green', fontweight='bold')
    ax3_twin.set_ylabel('Shor Time (ms)', fontsize=10, color='red', fontweight='bold')
    ax3.set_title('Direct Comparison (Dual Axis)', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='green')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left', fontsize=9)
    ax3_twin.legend(loc='upper right', fontsize=9)
    
    # Plot 4: Performance Breakdown Bar Chart (Middle Right)
    ax4 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(successful_n))
    width = 0.35
    bars1 = ax4.bar(x - width/2, rsa_times, width, label='RSA (μs)', color='green', alpha=0.7)
    ax4_twin = ax4.twinx()
    bars2 = ax4_twin.bar(x + width/2, shor_times, width, label='Shor (ms)', color='red', alpha=0.7)
    ax4.set_xlabel('RSA Modulus (n)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('RSA Time (μs)', fontsize=10, color='green', fontweight='bold')
    ax4_twin.set_ylabel('Shor Time (ms)', fontsize=10, color='red', fontweight='bold')
    ax4.set_title('Side-by-Side Performance', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([str(n) for n in successful_n], rotation=45, ha='right', fontsize=8)
    ax4.tick_params(axis='y', labelcolor='green')
    ax4_twin.tick_params(axis='y', labelcolor='red')
    ax4.legend(loc='upper left', fontsize=9)
    ax4_twin.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Ratio showing time difference (Bottom Span)
    ax5 = fig.add_subplot(gs[2, :])
    ratios = []
    for i in range(len(rsa_times)):
        if rsa_times[i] > 0:
            ratios.append(shor_times[i] * 1000 / rsa_times[i])
        else:
            ratios.append(0)
    
    colors = ['purple' if r > 100 else 'orange' if r > 50 else 'yellow' for r in ratios]
    bars = ax5.bar(range(len(successful_n)), ratios, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax5.set_xlabel('Test Cases', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Speed Ratio (Shor Time / RSA Time)', fontsize=11, fontweight='bold')
    ax5.set_title('How Much Slower is Shor vs RSA? (Classical Simulation)', fontsize=12, fontweight='bold')
    ax5.set_xticks(range(len(successful_n)))
    ax5.set_xticklabels([f'n={n}' for n in successful_n], rotation=45, ha='right')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Equal Performance')
    
    # Add value labels on bars
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(ratio)}x',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax5.legend(fontsize=10)
    
    plt.suptitle('RSA vs Shor\'s Algorithm: Comprehensive Performance Analysis', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.show()
    
    print(f"\n✓ Comprehensive graphs generated with {len(successful_n)} data points!")
    print("\nKey Insights:")
    print(f"  • RSA Time Range: {min(rsa_times):.3f}-{max(rsa_times):.3f} μs (microseconds)")
    print(f"  • RSA Average Time: {np.mean(rsa_times):.3f} μs")
    print(f"  • Shor Time Range: {min(shor_times):.3f}-{max(shor_times):.3f} ms (milliseconds)")
    print(f"  • Shor Average Time: {np.mean(shor_times):.3f} ms")
    valid_ratios = [r for r in ratios if r > 0]
    if valid_ratios:
        print(f"  • Classical Shor is ~{int(np.mean(valid_ratios))}x slower than RSA on average")
    print(f"  • IMPORTANT: Real quantum Shor would be exponentially faster for large keys!")
    print(f"  • This demonstrates the quantum threat to current RSA encryption.")
else:
    print("✗ Not enough data points to generate graph.")