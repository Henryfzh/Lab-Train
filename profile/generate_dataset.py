def generate_dataset(prob_head=0.666, sequence_length=10, num_sequences=1000):
    dataset = []
    for _ in range(num_sequences):
        sequence = np.random.choice(
            [0, 1], size=sequence_length, p=[1 - prob_head, prob_head]
        )
        dataset.append(" ".join(map(str, sequence)))
    return dataset
