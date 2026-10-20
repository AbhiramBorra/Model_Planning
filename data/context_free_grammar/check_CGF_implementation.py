# check_CFG_implementation.py
import random
import sys
import os
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Import the CFG class from the CFG_data_generation module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from CFG_data_generation import CFG

def run_cfg_test(size, name=""):
    print(f"\n{'='*50}")
    print(f"Testing CFG with size {size} {name}")
    print(f"{'='*50}")
    
    # Create CFG with specific parameters
    rule_degrees = [4,5,6]
    rule_lengths = [3,4,5]  # Common rule lengths
    
    # Initialize the CFG
    cfg = CFG(size=size, rule_degrees=rule_degrees, rule_lengths=rule_lengths)
    
    print(f"CFG created with:")
    print(f"- Symbols: {cfg.symbols}")
    print(f"- Rule count: {len(cfg.rules)}")
    
    # Generate 100 valid sequences
    print("Generating 100 valid sequences...")
    valid_sequences, _ = cfg.generate_multiple_sequences_parallel(
        num_sequences=100, history=False, max_workers=8
    )

    # Try to add 10 random permutations and acc should be %90.9
    # Add 10 random permutations of an existing sequence
    print("Adding 10 random permutations of an existing sequence...")
    # Take the first valid sequence as reference
    reference_seq = valid_sequences[0]
    unique_chars = list(set(reference_seq))  # Get unique characters
    # Generate 10 random permutations
    for _ in range(10):
        permuted_seq = [random.choice(unique_chars) for _ in range(len(reference_seq))]
        valid_sequences.append(permuted_seq)
    
    # Extract terminal symbols (last layer in the grammar)
    terminal_symbols = cfg.symbols[-1]
    print(f"Terminal symbols: {terminal_symbols}")
    
    # Generate 100 random sequences with same lengths but random terminal symbols
    print("Generating 100 random sequences with same lengths...")
    random_sequences = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        def generate_random_seq(seq):
            return [random.choice(terminal_symbols) for _ in range(len(seq))]
        random_sequences = list(executor.map(generate_random_seq, valid_sequences))

    # Test valid sequences
    print("Testing valid sequences...")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        valid_results = list(tqdm(executor.map(cfg.is_valid_sequence, valid_sequences), 
                                total=len(valid_sequences)))
        valid_correct = sum(valid_results)

    # Test random (likely invalid) sequences  
    print("Testing random sequences...")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        random_results = list(tqdm(executor.map(cfg.is_valid_sequence, random_sequences),
                                 total=len(random_sequences)))
        random_correct = sum(not result for result in random_results)
    
    # Calculate accuracy
    valid_accuracy = (valid_correct / len(valid_sequences)) * 100
    random_accuracy = (random_correct / len(random_sequences)) * 100
    overall_accuracy = ((valid_correct + random_correct) / (len(valid_sequences) + len(random_sequences))) * 100
    
    # Print results
    print(f"\nResults:")
    print(f"- Valid sequences correctly identified: {valid_correct}/{len(valid_sequences)} ({valid_accuracy:.2f}%) (which should be 90.91%)")
    print(f"- Random sequences correctly identified as invalid: {random_correct}/{len(random_sequences)} ({random_accuracy:.2f}%) (which should be 100%)")
    print(f"- Overall accuracy: {overall_accuracy:.2f}%")
    
    # Calculate sequence statistics
    lengths = [len(seq) for seq in valid_sequences]
    min_len = min(lengths)
    max_len = max(lengths)
    avg_len = sum(lengths) / len(lengths)
    
    print(f"\nSequence statistics:")
    print(f"- Min length: {min_len}")
    print(f"- Max length: {max_len}")
    print(f"- Average length: {avg_len:.2f}")
    
    return {
        "valid_accuracy": valid_accuracy,
        "random_accuracy": random_accuracy,
        "overall_accuracy": overall_accuracy
    }

def main():
    print("Starting CFG implementation check...")
    
    # Test first CFG
    cfg1_results = run_cfg_test(size=(1,7,6,7,6), name="(4 layers)")
    
    # Test second CFG
    # cfg2_results = run_cfg_test(size=(1,3,3,3,3), name="(5 layers)")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"CFG (1,3,4,3): Overall accuracy: {cfg1_results['overall_accuracy']:.2f}% (which should be 95.45%)")
    # print(f"CFG (1,3,3,3,3): Overall accuracy: {cfg2_results['overall_accuracy']:.2f}%")

if __name__ == "__main__":
    main()